import numpy as np
import time
import json
import os
import sys
import struct
import gzip
import pickle
import threading
import glob
from datetime import datetime


# ============================================================
# INT4 nibble packing/unpacking for VTA
#
# VTA int4 stores two values per byte: lo nibble = even element,
# hi nibble = odd element.  DMA transfers INP_ELEM_BYTES = 8 bytes
# per 16-element vector (16 × 4 bits = 64 bits).
#
# numpy int8 arrays store one value per byte.  When used as VTA int4
# tensors, the first half of the bytes are read by DMA and each byte
# is interpreted as two packed nibbles.  Without explicit packing,
# every odd-indexed element is silently zero.
#
# These helpers convert between the two representations.  They are
# no-ops for int8 (called only on the vta_native int4 path).
# ============================================================

def pack_int4_for_vta(vals_int8):
    """Pack int8 array of int4 values into VTA nibble format.

    VTA int4 DMA reads the buffer flat — byte[flat_index / 2] for element
    at flat_index.  Two values per byte: lo nibble = even flat index,
    hi nibble = odd flat index.  Packing must be contiguous across the
    ENTIRE flattened tensor, not per-row, because VTA's byte addressing
    spans all dimensions without gaps.

    Input:  int8 array, any shape.  Each element is one int4 value.
    Output: int8 array, same shape.  First half of flat buffer holds
            packed nibble pairs; second half is zero-padded.
    """
    vals = np.asarray(vals_int8, dtype=np.int8)
    flat = vals.flatten()
    n = len(flat)
    lo = flat[0::2].view(np.uint8) & 0xF
    hi = flat[1::2].view(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[:n // 2] = packed
    return out.reshape(vals.shape)


def unpack_int4_from_vta(packed_int8):
    """Unpack VTA nibble-packed int4 output to one-value-per-element int8.

    Inverse of pack_int4_for_vta.  First half of flat buffer holds packed
    nibble pairs.  Each pair is split into two sign-extended int4 values.

    Input:  int8 array, any shape.
    Output: int8 array, same shape.  Each element is one sign-extended
            int4 value in [-8, 7].
    """
    raw = np.asarray(packed_int8, dtype=np.int8)
    flat = raw.flatten()
    n = len(flat)
    packed_bytes = flat[:n // 2].view(np.uint8)
    lo = (packed_bytes & 0xF).astype(np.int8)
    hi = ((packed_bytes >> 4) & 0xF).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.int8)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out.reshape(raw.shape)

# ============================================================
# Power measurement
# Three modes (in priority order):
#   1. INA260 via hwmon sysfs (e.g. KV260)
#   2. FNB58 USB power meter  -- TODO: implement when meter arrives
#   3. None                   -- no power data, log null + warn
# SYSMON (on-chip temp/voltage) always logged as supplementary data.
# ============================================================

def find_power_file():
    for f in glob.glob('/sys/class/hwmon/hwmon*/name'):
        with open(f) as fh:
            if 'ina260' in fh.read():
                return f.rsplit('/', 1)[0] + '/power1_input'
    return None

POWER_FILE = find_power_file()
POWER_AVAILABLE = POWER_FILE is not None

if not POWER_AVAILABLE:
    print("NOTE: No on-board power sensor (INA260). Board power collected host-side via FNB58.")
    print("      SYSMON temperature/voltage will be logged as supplementary data.")

def read_power():
    if not POWER_AVAILABLE:
        return None
    with open(POWER_FILE) as f:
        return int(f.read().strip()) / 1e6

# ============================================================
# SYSMON (on-chip temperature and voltage)
# Read via Linux IIO sysfs interface (xilinx-ams driver).
# No overlay loading required -- safe to use alongside any bitfile.
#
# ZynqMP AMS channel mapping (verified on AUP-ZU3):
#   in_temp7    = PS die temperature
#   in_temp8    = PL die temperature
#   in_voltage6 = VCCINT  (PL core, nominal 0.85V)
#   in_voltage9 = VCCBRAM (PL BRAM, nominal 0.85V)
#   in_voltage11= VCCAUX  (PL aux,  nominal 1.8V)
# ============================================================

IIO_BASE = '/sys/bus/iio/devices/iio:device0'
SYSMON_AVAILABLE = os.path.exists(IIO_BASE)

if not SYSMON_AVAILABLE:
    print("WARNING: IIO SYSMON interface not found. Thermal data will not be logged.")

def _iio_read(channel):
    """Read a single IIO channel, return converted float or None."""
    try:
        raw   = int(open(f'{IIO_BASE}/{channel}_raw').read().strip())
        scale = float(open(f'{IIO_BASE}/{channel}_scale').read().strip())
        # Temperature channels also have an offset
        if 'temp' in channel:
            offset = float(open(f'{IIO_BASE}/{channel}_offset').read().strip())
            return (raw + offset) * scale / 1000.0
        return raw * scale / 1000.0
    except Exception:
        return None

def read_sysmon():
    """Read IIO SYSMON channels. Returns dict or None."""
    if not SYSMON_AVAILABLE:
        return None
    temp_ps  = _iio_read('in_temp7')
    temp_pl  = _iio_read('in_temp8')
    vccint   = _iio_read('in_voltage6')
    vccbram  = _iio_read('in_voltage9')
    vccaux   = _iio_read('in_voltage11')
    if all(v is None for v in [temp_ps, temp_pl, vccint, vccbram, vccaux]):
        return None
    return {
        'temp_ps_c':  temp_ps,
        'temp_pl_c':  temp_pl,
        'vccint_v':   vccint,
        'vccbram_v':  vccbram,
        'vccaux_v':   vccaux,
    }

# ============================================================
# Data loaders
# ============================================================
def _default_mnist_path():
    for p in ['/home/petalinux/MNIST/raw', '/home/xilinx/MNIST/raw']:
        if os.path.exists(p):
            return p
    return '/home/xilinx/MNIST/raw'

def load_mnist(path=None):
    if path is None:
        path = _default_mnist_path()
    def load_images(p):
        with gzip.open(p, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    def load_labels(p):
        with gzip.open(p, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    images = load_images(f'{path}/t10k-images-idx3-ubyte.gz')
    labels = load_labels(f'{path}/t10k-labels-idx1-ubyte.gz')
    images = images.astype(np.float32) / 255.0
    images = np.expand_dims(images, axis=1)
    return images, labels

def _default_cifar10_path():
    for p in ['/home/petalinux/cifar-10-batches-py/test_batch', '/home/xilinx/cifar-10-batches-py/test_batch']:
        if os.path.exists(p):
            return p
    return '/home/xilinx/cifar-10-batches-py/test_batch'

def load_cifar10(path=None):
    if path is None:
        path = _default_cifar10_path()
    with open(path, 'rb') as f:
        test = pickle.load(f, encoding='bytes')
    images = test[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(test[b'labels'])
    return images, labels

# ============================================================
# Sampling thread
# ============================================================
def make_sampler(power_log, sysmon_log, sampling_flag):
    def sampler():
        while sampling_flag[0]:
            t = time.time()
            pwr = read_power()
            smon = read_sysmon()
            if pwr is not None:
                power_log.append((t, pwr))
            if smon is not None:
                sysmon_log.append((t, smon))
            time.sleep(0.01)
    return sampler

# ============================================================
# Idle measurement
# ============================================================
def measure_idle(idle_seconds):
    print(f"Measuring idle ({idle_seconds}s)...")
    t_start = time.time()
    power_samples = []
    sysmon_samples = []
    n = int(idle_seconds / 0.02)
    for _ in range(n):
        pwr = read_power()
        smon = read_sysmon()
        if pwr is not None:
            power_samples.append(pwr)
        if smon is not None:
            sysmon_samples.append(smon)
        time.sleep(0.02)
    t_end = time.time()

    if power_samples:
        idle_power = float(np.mean(power_samples))
        idle_std = float(np.std(power_samples))
        print(f"  Idle power: {idle_power:.3f} +/- {idle_std:.3f} W")
    else:
        idle_power = None
        idle_std = None
        print("  Idle power: collected host-side (FNB58)")

    if sysmon_samples:
        idle_temp_ps = float(np.mean([s['temp_ps_c'] for s in sysmon_samples if s['temp_ps_c'] is not None]))
        idle_temp_pl = float(np.mean([s['temp_pl_c'] for s in sysmon_samples if s['temp_pl_c'] is not None]))
        idle_vccint  = float(np.mean([s['vccint_v']  for s in sysmon_samples if s['vccint_v']  is not None]))
        print(f"  Idle temp:  PS={idle_temp_ps:.1f}C  PL={idle_temp_pl:.1f}C  VCCINT={idle_vccint:.3f}V")
    else:
        idle_temp_ps = None
        idle_temp_pl = None
        idle_vccint  = None

    return {
        't_start': t_start,
        't_end': t_end,
        'power':  {'mean': idle_power, 'std': idle_std, 'n_samples': len(power_samples)},
        'sysmon': {'temp_ps_c': idle_temp_ps, 'temp_pl_c': idle_temp_pl,
                   'vccint_v': idle_vccint, 'n_samples': len(sysmon_samples)},
    }

# ============================================================
# Per-run result builder
# ============================================================
def build_run_result(run_num, correct, total, elapsed, power_log, sysmon_log,
                     t_start=None, t_end=None):
    has_power  = len(power_log)  > 0
    has_sysmon = len(sysmon_log) > 0

    avg_power      = float(np.mean([s[1] for s in power_log])) if has_power else None
    energy_total   = avg_power * elapsed if avg_power is not None else None
    energy_per_img = 1000 * energy_total / total if energy_total is not None else None

    sysmon_summary = None
    if has_sysmon:
        sysmon_summary = {
            'temp_ps_c_mean':  float(np.mean([s[1]['temp_ps_c'] for s in sysmon_log if s[1]['temp_ps_c'] is not None])),
            'temp_pl_c_mean':  float(np.mean([s[1]['temp_pl_c'] for s in sysmon_log if s[1]['temp_pl_c'] is not None])),
            'temp_pl_c_max':   float(np.max( [s[1]['temp_pl_c'] for s in sysmon_log if s[1]['temp_pl_c'] is not None])),
            'vccint_v_mean':   float(np.mean([s[1]['vccint_v']  for s in sysmon_log if s[1]['vccint_v']  is not None])),
            'vccbram_v_mean':  float(np.mean([s[1]['vccbram_v'] for s in sysmon_log if s[1]['vccbram_v'] is not None])),
            'n_samples':       len(sysmon_log),
        }

    result = {
        'run': run_num,
        't_start': t_start,
        't_end': t_end,
        'accuracy': 100 * correct / total,
        'time_s': elapsed,
        'throughput_fps': total / elapsed,
        'latency_ms': 1000 * elapsed / total,
        'avg_power_w': avg_power,
        'energy_total_j': energy_total,
        'energy_per_image_mj': energy_per_img,
        'power_samples': len(power_log),
        'sysmon': sysmon_summary,
    }

    pwr_str  = f"{avg_power:.3f} W"       if avg_power      is not None else "N/A"
    enrg_str = f"{energy_per_img:.4f} mJ/img" if energy_per_img is not None else "N/A"
    temp_str = f"{sysmon_summary['temp_pl_c_mean']:.1f}C" if sysmon_summary else "N/A"
    print(f"  Run {run_num}: {result['throughput_fps']:.1f} FPS, "
          f"pwr={pwr_str}, {enrg_str}, temp={temp_str}")
    return result

# ============================================================
# Vitis AI benchmark
# ============================================================
def run_vitisai_benchmark(model_path, dataset, batch_size, num_runs,
                          warmup_batches, idle_seconds, stabilize_seconds,
                          results_dir, run_name):
    from pynq_dpu import DpuOverlay

    print(f"Loading {dataset} dataset...")
    images, labels = load_mnist() if dataset == 'mnist' else load_cifar10()
    print(f"  {len(images)} images, shape {images[0].shape}")

    print(f"Loading model: {model_path}")
    overlay = DpuOverlay('dpu.bit')
    overlay.load_model(model_path)
    dpu = overlay.runner

    inputTensors  = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_shape   = tuple(inputTensors[0].dims)
    output_shape  = tuple(outputTensors[0].dims)
    print(f"  DPU input: {input_shape}, output: {output_shape}")

    single_image_size = int(np.prod(images[0].shape))
    dpu_input_size    = int(np.prod(input_shape[1:]))
    is_batched        = dpu_input_size > single_image_size

    if is_batched:
        actual_batch = batch_size
        num_batches  = len(images) // actual_batch
        print(f"  Batched mode: {actual_batch} images/call, {num_batches} batches")
    else:
        actual_batch = 1
        num_batches  = len(images)
        print(f"  Single image mode: {num_batches} images")

    config = {
        'toolchain': 'vitis_ai',
        'model_path': model_path,
        'dataset': dataset,
        'batch_size': actual_batch,
        'num_runs': num_runs,
        'num_images': len(images),
        'image_shape': list(images[0].shape),
        'dpu_input_shape': list(input_shape),
        'dpu_output_shape': list(output_shape),
        'timestamp': datetime.now().isoformat(),
        'board': 'AUP-ZU3',
        'dpu': 'DPUCZDX8G_ISA1_B2304',
        'power_method': 'ina260' if POWER_AVAILABLE else 'none',
    }

    print(f"Thermal stabilization ({stabilize_seconds}s)...")
    time.sleep(stabilize_seconds)

    idle = measure_idle(idle_seconds)

    print(f"Warmup ({warmup_batches} batches)...")
    for b in range(warmup_batches):
        if is_batched:
            batch_imgs = images[b*actual_batch:(b+1)*actual_batch]
            input_data = [np.ascontiguousarray(batch_imgs.flatten().reshape(input_shape), dtype=np.float32)]
        else:
            input_data = [np.empty(input_shape, dtype=np.float32, order='C')]
            input_data[0][0] = images[b]
        output_data = [np.empty(output_shape, dtype=np.float32, order='C')]
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)

    print(f"Running {num_runs} measured runs...")
    all_runs = []
    for run in range(num_runs):
        power_log   = []
        sysmon_log  = []
        sampling_flag = [True]
        thread = threading.Thread(target=make_sampler(power_log, sysmon_log, sampling_flag))
        correct = 0
        total   = 0
        thread.start()
        start_time = time.time()

        if is_batched:
            for b in range(num_batches):
                batch_imgs   = images[b*actual_batch:(b+1)*actual_batch]
                batch_labels = labels[b*actual_batch:(b+1)*actual_batch]
                input_data   = [np.ascontiguousarray(batch_imgs.flatten().reshape(input_shape), dtype=np.float32)]
                output_data  = [np.empty(output_shape, dtype=np.float32, order='C')]
                job_id = dpu.execute_async(input_data, output_data)
                dpu.wait(job_id)
                preds   = np.argmax(output_data[0].reshape(actual_batch, -1), axis=1)
                correct += int(np.sum(preds == batch_labels))
                total   += actual_batch
        else:
            for i in range(len(images)):
                input_data  = [np.empty(input_shape,  dtype=np.float32, order='C')]
                output_data = [np.empty(output_shape, dtype=np.float32, order='C')]
                input_data[0][0] = images[i]
                job_id = dpu.execute_async(input_data, output_data)
                dpu.wait(job_id)
                pred = int(np.argmax(output_data[0][0]))
                if pred == labels[i]:
                    correct += 1
                total += 1

        elapsed = time.time() - start_time
        end_time = time.time()
        sampling_flag[0] = False
        thread.join()
        all_runs.append(build_run_result(run + 1, correct, total, elapsed, power_log, sysmon_log,
                                         t_start=start_time, t_end=end_time))

    return config, idle, all_runs

# ============================================================
# FINN benchmark
# ============================================================
def run_finn_benchmark(deploy_dir, dataset, batch_size, num_runs,
                       warmup_batches, idle_seconds, stabilize_seconds,
                       results_dir, run_name,
                       finn_runtime='python'):

    driver_dir = os.path.join(deploy_dir, 'driver')
    bitfile    = os.path.join(deploy_dir, 'bitfile', 'finn-accel.bit')
    sys.path.insert(0, driver_dir)

    from driver import io_shape_dict
    from driver_base import FINNExampleOverlay
    from finn.util.data_packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy

    print(f"Loading {dataset} dataset...")
    images, labels = load_mnist() if dataset == 'mnist' else load_cifar10()
    print(f"  {len(images)} images, shape {images[0].shape}")

    images_uint8 = (images * 255).clip(0, 255).astype(np.uint8)
    n_inputs     = int(np.prod(images_uint8[0].shape))
    images_flat  = images_uint8.reshape(len(images_uint8), n_inputs)

    ishape_normal = io_shape_dict['ishape_normal'][0]
    oshape_normal = io_shape_dict['oshape_normal'][0]
    ishape_packed = io_shape_dict['ishape_packed'][0]
    oshape_packed = io_shape_dict['oshape_packed'][0]
    idt = io_shape_dict['idt'][0]
    odt = io_shape_dict['odt'][0]

    print(f"Loading FINN bitfile: {bitfile}")
    ol = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform='zynq-iodma',
        io_shape_dict=io_shape_dict,
        batch_size=1,
        runtime_weight_dir=os.path.join(driver_dir, 'runtime_weights')
    )
    print(f"  Input shape: {ishape_normal}, Output shape: {oshape_normal}")

    # Load CPU-side weights if present
    has_mlp_pre = os.path.exists(os.path.join(deploy_dir, 'mlp_MatMul_0_param0.npy'))
    has_cnn_pre = os.path.exists(os.path.join(deploy_dir, 'cnn_MatMul_0_param0.npy'))

    if has_mlp_pre:
        print("  CPU pre-processing: MLP first layer (partial hardware mapping)")
        W0      = np.load(os.path.join(deploy_dir, 'mlp_MatMul_0_param0.npy'))
        thres   = np.load(os.path.join(deploy_dir, 'mlp_MultiThreshold_0_param0.npy'))
        mul_out = np.load(os.path.join(deploy_dir, 'mlp_Mul_0_param0.npy'))
        add_out = np.load(os.path.join(deploy_dir, 'mlp_Add_0_param0.npy'))
    elif has_cnn_pre:
        print("  CPU pre/post processing: CNN first+last layers (partial hardware mapping)")
        W_conv  = np.load(os.path.join(deploy_dir, 'cnn_MatMul_0_param0.npy'))   # (27, 8)
        thres   = np.load(os.path.join(deploy_dir, 'cnn_MultiThreshold_0_param0.npy'))  # (8, 255)
        W_cls   = np.load(os.path.join(deploy_dir, 'cnn_MatMul_2_param0.npy'))   # (16, 10)
        mul_out = np.load(os.path.join(deploy_dir, 'cnn_Mul_0_param0.npy'))
        add_out = np.load(os.path.join(deploy_dir, 'cnn_Add_0_param0.npy'))
    else:
        print("  Full hardware mapping")


    config = {
        'toolchain': 'finn',
        'deploy_dir': deploy_dir,
        'dataset': dataset,
        'batch_size': 1,
        'num_runs': num_runs,
        'num_images': len(images),
        'image_shape': list(images[0].shape),
        'finn_input_shape': list(ishape_normal),
        'finn_output_shape': list(oshape_normal),
        'timestamp': datetime.now().isoformat(),
        'board': 'AUP-ZU3',
        'fpga_part': 'xczu3eg-sbva484-1-e',
        'power_method': 'ina260' if POWER_AVAILABLE else 'none',
        'cpu_pre_layer': has_mlp_pre or has_cnn_pre,
        'cpu_split_type': 'mlp' if has_mlp_pre else 'cnn' if has_cnn_pre else 'none',
    }

    print(f"Thermal stabilization ({stabilize_seconds}s)...")
    time.sleep(stabilize_seconds)

    idle = measure_idle(idle_seconds)

    def im2col(x, kernel_size=3, stride=1, pad=1):
        """Extract sliding window patches. x: (H, W, C) -> (H, W, k*k*C)"""
        H, W, C = x.shape
        kH, kW = kernel_size, kernel_size
        x_pad = np.pad(x, ((pad,pad),(pad,pad),(0,0)), mode='constant')
        out = np.zeros((H, W, kH*kW*C), dtype=x.dtype)
        for i in range(H):
            for j in range(W):
                patch = x_pad[i:i+kH, j:j+kW, :]
                out[i, j, :] = patch.flatten()
        return out

    def multithreshold(x, thresholds):
        """x: (..., C), thresholds: (C, 255) -> (..., C) uint8"""
        # Use >= to match qonnx.custom_op.general.multithreshold (line 78), the
        # reference the FINN compiler/training pipeline uses. The previous
        # strict-> form differed only at exact ties (~1 in 64k channel-evals on
        # MNIST) but is wrong against the trained-model reference.
        # Was:
        #   return np.sum(x[..., np.newaxis] > thresholds, axis=-1).astype(np.uint8)
        return np.sum(x[..., np.newaxis] >= thresholds, axis=-1).astype(np.uint8)

    def infer(img_flat):
        if has_mlp_pre:
            x = img_flat.astype(np.float32) / 255.0 @ W0
            x = multithreshold(x, thres)
            hw_out = ol.execute([x.reshape(ishape_normal)])
            out = hw_out.flatten().astype(np.float32) * mul_out + add_out
            return int(np.argmax(out))
        elif has_cnn_pre:
            # Auto-detect spatial dims from weight shape
            # W_conv is (k*k*C_in, C_out) — MNIST: (9,8), CIFAR-10: (27,8)
            kkc = W_conv.shape[0]
            c_out = W_conv.shape[1]
            n_inputs = len(img_flat)
            if n_inputs == 784:  # MNIST 28*28
                img = img_flat.reshape(28, 28, 1).astype(np.float32) / 255.0
            else:  # CIFAR-10 3*32*32
                img = img_flat.reshape(3, 32, 32).astype(np.float32) / 255.0
                img = img.transpose(1, 2, 0)  # (32, 32, 3)
            H, W, C = img.shape
            patches = im2col(img)               # (H, W, k*k*C)
            x = patches @ W_conv                # (H, W, c_out)
            x = multithreshold(x, thres)        # (H, W, c_out) uint8
            hw_out = ol.execute([x.reshape(ishape_normal)])
            # oshape_normal tells us the output spatial dims
            feat = hw_out.reshape(oshape_normal[1], oshape_normal[2], oshape_normal[3])
            feat = feat.mean(axis=(0, 1))       # (C,) global average pool
            out = feat @ W_cls
            out = out.astype(np.float32) * mul_out + add_out
            return int(np.argmax(out))

    # ---------- Optional C hot-path ----------
    # Mirrors the libfinn_t_infer.so pattern: Python owns bitstream / buffer
    # allocation / MMIO mapping, and hands pointers into a shared library
    # that runs the per-image hot loop. Two variants are supported, selected
    # by the deploy type:
    #   has_mlp_pre -> libfinn_mlp_infer.so  (6-stage profile)
    #   has_cnn_pre -> libfinn_cnn_infer.so  (10-stage profile, MNIST only)
    # The warmup / profiled / measured-loop code downstream is identical for
    # both — c_state stashes the runner's batch/profile/destroy callables and
    # stage-name list so the shared code dispatches without caring which.
    c_runner = None
    c_state  = None
    if finn_runtime == 'c' and has_mlp_pre:
        _so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'libfinn_mlp_infer.so')
        try:
            import ctypes
            _lib = ctypes.CDLL(_so_path)
            _lib.finn_mlp_runner_init.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int,                # ibuf_bytes (caller-supplied)
                ctypes.c_int,                # use_cache_ops
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
            ]
            _lib.finn_mlp_runner_init.restype = ctypes.c_int
            _lib.finn_mlp_runner_destroy.argtypes = []
            _lib.finn_mlp_runner_destroy.restype  = ctypes.c_int
            _lib.finn_mlp_infer_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ]
            _lib.finn_mlp_infer_batch.restype  = ctypes.c_int
            _lib.finn_mlp_infer_one_profiled.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _lib.finn_mlp_infer_one_profiled.restype  = ctypes.c_int

            precision = int(idt.bitwidth())
            assert not idt.signed(), (
                "C runner expects unsigned activation (UINTn); got signed idt")
            nthres = (1 << precision) - 1  # full threshold count for the dtype

            # Reuse ol's auto-allocated buffers: they're already cacheable=True,
            # matching use_cache_ops=1 below. Avoids a duplicate allocation.
            ibuf_dev = ol.ibuf_packed_device[0]
            obuf_dev = ol.obuf_packed_device[0]
            idma_mmio = ol.idma[0].mmio.array.ctypes.data
            odma_mmio = ol.odma[0].mmio.array.ctypes.data

            W0_c    = np.ascontiguousarray(W0.astype(np.float32))
            thres_c = np.ascontiguousarray(thres.astype(np.float32))
            add_c   = np.ascontiguousarray(add_out.astype(np.float32))
            mul_v   = float(np.asarray(mul_out).flatten()[0])

            # Per-image packed input byte count from FINN's driver. Equals
            # mid_dim at PE=SIMD=1 (1-per-byte INT4 or INT8); halves to
            # mid_dim/2 when FINN folds with high SIMD on INT4 (2-per-byte
            # convention, e.g. mlp_int4_fps500000 with SIMD=16).
            ibuf_bytes_per_image = int(np.prod(ol.ishape_packed(0)[1:]))
            rc = _lib.finn_mlp_runner_init(
                precision, n_inputs, W0.shape[1], add_out.shape[0], nthres,
                ibuf_bytes_per_image,
                1,  # use_cache_ops=1 (ol buffers are cacheable)
                ibuf_dev.ctypes.data, int(ibuf_dev.device_address),
                obuf_dev.ctypes.data, int(obuf_dev.device_address),
                idma_mmio, odma_mmio,
                W0_c.ctypes.data, thres_c.ctypes.data, mul_v, add_c.ctypes.data)
            if rc != 0:
                print(f"  finn_mlp_runner_init returned {rc}; falling back to Python")
            else:
                images_c = np.ascontiguousarray(images_flat.astype(np.uint8))
                labels_c = np.ascontiguousarray(labels.astype(np.int32))
                preds_c  = np.zeros(len(images_c), dtype=np.int32)
                c_runner = _lib
                c_state  = {
                    'precision': precision,
                    'W0_c':    W0_c,      'thres_c': thres_c,
                    'add_c':   add_c,     'mul_v':   mul_v,
                    'images':  images_c,  'labels':  labels_c,
                    'preds':   preds_c,
                    'batch_fn':    _lib.finn_mlp_infer_batch,
                    'profile_fn':  _lib.finn_mlp_infer_one_profiled,
                    'destroy_fn':  _lib.finn_mlp_runner_destroy,
                    'stage_names': ['MatMul', 'MultiThreshold', 'Pack',
                                    'DMA trig+wait', 'Unpack',
                                    'Post dequant+argmax'],
                }
                print(f"  MLP C runner loaded: {_so_path} "
                      f"(INT{precision}, nthres={nthres})")
        except OSError as e:
            print(f"  C runner not available ({e}); falling back to Python infer")
    elif finn_runtime == 'c' and has_cnn_pre:
        if n_inputs != 784:
            # CIFAR-10 (n_inputs = 3072) falls here. Shapes would differ and
            # no CNN CIFAR-10 deploy has been validated against C yet.
            print("  CNN C runner is MNIST-only (28x28x1); CIFAR-10 uses Python")
        else:
            _so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'libfinn_cnn_infer.so')
            try:
                import ctypes
                _lib = ctypes.CDLL(_so_path)
                _lib.finn_cnn_runner_init.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p, ctypes.c_uint64,
                    ctypes.c_void_p, ctypes.c_uint64,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_float, ctypes.c_void_p,
                ]
                _lib.finn_cnn_runner_init.restype = ctypes.c_int
                _lib.finn_cnn_runner_destroy.argtypes = []
                _lib.finn_cnn_runner_destroy.restype  = ctypes.c_int
                _lib.finn_cnn_infer_batch.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                ]
                _lib.finn_cnn_infer_batch.restype  = ctypes.c_int
                _lib.finn_cnn_infer_one_profiled.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p]
                _lib.finn_cnn_infer_one_profiled.restype  = ctypes.c_int

                precision = int(idt.bitwidth())
                assert not idt.signed(), (
                    "C runner expects unsigned activation (UINTn); got signed idt")
                nthres = (1 << precision) - 1

                # Geometry derived from weight + io_shape_dict.  W_conv shape
                # is (kernel*kernel*img_c, fpga_in_c); for MNIST that's (9, 8).
                img_h, img_w, img_c = 28, 28, 1
                patch_dim = W_conv.shape[0]
                kernel_size = int(round((patch_dim // img_c) ** 0.5))
                pad = 1  # FINN MNIST-tiny convention; same=1 for 3x3
                fpga_in_c = int(W_conv.shape[1])
                fpga_out_h = int(oshape_normal[1])
                fpga_out_w = int(oshape_normal[2])
                fpga_out_c = int(oshape_normal[3])
                num_classes = int(add_out.shape[0])

                ibuf_dev = ol.ibuf_packed_device[0]
                obuf_dev = ol.obuf_packed_device[0]
                idma_mmio = ol.idma[0].mmio.array.ctypes.data
                odma_mmio = ol.odma[0].mmio.array.ctypes.data

                W_conv_c = np.ascontiguousarray(W_conv.astype(np.float32))
                thres_c  = np.ascontiguousarray(thres.astype(np.float32))
                W_cls_c  = np.ascontiguousarray(W_cls.astype(np.float32))
                add_c    = np.ascontiguousarray(add_out.astype(np.float32))
                mul_v    = float(np.asarray(mul_out).flatten()[0])

                rc = _lib.finn_cnn_runner_init(
                    precision,
                    img_h, img_w, img_c,
                    kernel_size, pad,
                    fpga_in_c,
                    fpga_out_h, fpga_out_w, fpga_out_c,
                    num_classes, nthres,
                    1,  # use_cache_ops=1
                    ibuf_dev.ctypes.data, int(ibuf_dev.device_address),
                    obuf_dev.ctypes.data, int(obuf_dev.device_address),
                    idma_mmio, odma_mmio,
                    W_conv_c.ctypes.data, thres_c.ctypes.data, W_cls_c.ctypes.data,
                    mul_v, add_c.ctypes.data)
                if rc != 0:
                    print(f"  finn_cnn_runner_init returned {rc}; "
                          f"falling back to Python")
                else:
                    # Flatten MNIST to uint8 [N, 784] for the C batch call.
                    images_c = np.ascontiguousarray(images_flat.astype(np.uint8))
                    labels_c = np.ascontiguousarray(labels.astype(np.int32))
                    preds_c  = np.zeros(len(images_c), dtype=np.int32)
                    c_runner = _lib
                    c_state  = {
                        'precision': precision,
                        'W_conv_c': W_conv_c, 'thres_c': thres_c,
                        'W_cls_c':  W_cls_c,  'add_c':   add_c,  'mul_v': mul_v,
                        'images':   images_c, 'labels':  labels_c,
                        'preds':    preds_c,
                        'batch_fn':    _lib.finn_cnn_infer_batch,
                        'profile_fn':  _lib.finn_cnn_infer_one_profiled,
                        'destroy_fn':  _lib.finn_cnn_runner_destroy,
                        'stage_names': ['CastNorm', 'im2col', 'MatMul1',
                                        'MultiThreshold', 'Pack',
                                        'DMA trig+wait', 'CacheInv', 'GAP',
                                        'MatMul2', 'Post dequant+argmax'],
                    }
                    print(f"  CNN C runner loaded: {_so_path} "
                          f"(INT{precision}, geometry "
                          f"{img_h}x{img_w}x{img_c} k={kernel_size} pad={pad} "
                          f"-> {fpga_out_h}x{fpga_out_w}x{fpga_out_c})")
            except OSError as e:
                print(f"  C runner not available ({e}); falling back to Python")
    elif finn_runtime == 'c':
        print("  --finn-runtime=c requires an MLP or CNN deploy "
              "(has_mlp_pre or has_cnn_pre); falling back to Python")

    config['finn_runtime'] = ('c' if c_runner is not None else 'python')

    if c_runner is not None:
        print(f"Warmup ({warmup_batches} images via C batch)...")
        _warm_n = min(warmup_batches, len(c_state['images']))
        c_state['batch_fn'](
            c_state['images'].ctypes.data, None, _warm_n,
            c_state['preds'].ctypes.data)

        # One-shot per-stage timing breakdown on image[0]. Stage count is
        # runner-specific: 6 for MLP, 10 for CNN.
        stage_names = c_state['stage_names']
        ns_arr = np.zeros(len(stage_names), dtype=np.uint64)
        c_state['profile_fn'](
            c_state['images'][0].ctypes.data, ns_arr.ctypes.data)
        total_ns = int(ns_arr.sum())
        print('[C runner timing breakdown — one inference on images_flat[0]]:')
        for name, ns in zip(stage_names, ns_arr.tolist()):
            pct = 100.0 * ns / max(1, total_ns)
            print(f'  {name:22s}: {ns/1000.0:9.2f} µs   ({pct:5.1f} %)')
        print(f'  {"total":22s}: {total_ns/1000.0:9.2f} µs   '
              f'(steady-state upper bound ~ {1e9 / max(1, total_ns):.0f} FPS)')
    else:
        print(f"Warmup ({warmup_batches} images)...")
        for b in range(min(warmup_batches, len(images_flat))):
            infer(images_flat[b])

    print(f"Running {num_runs} measured runs...")
    all_runs = []
    for run in range(num_runs):
        power_log     = []
        sysmon_log    = []
        sampling_flag = [True]
        thread  = threading.Thread(target=make_sampler(power_log, sysmon_log, sampling_flag))
        thread.start()
        start_time = time.time()

        if c_runner is not None:
            correct = c_state['batch_fn'](
                c_state['images'].ctypes.data,
                c_state['labels'].ctypes.data,
                len(c_state['images']),
                c_state['preds'].ctypes.data)
            total = len(c_state['images'])
        else:
            correct = 0
            total   = 0
            for i in range(len(images_flat)):
                pred = infer(images_flat[i])
                if pred == labels[i]:
                    correct += 1
                total += 1

        elapsed = time.time() - start_time
        end_time = time.time()
        sampling_flag[0] = False
        thread.join()
        all_runs.append(build_run_result(run + 1, correct, total, elapsed, power_log, sysmon_log,
                                         t_start=start_time, t_end=end_time))

    if c_runner is not None:
        c_state['destroy_fn']()

    return config, idle, all_runs

# ============================================================
# FINN-T transformer benchmark (RadioML 2018)
# ============================================================
def load_radioml(hdf5_path, seed=12, split_fractions=(0.80, 0.10, 0.10),
                 snr_range=None, reshape=(1, 1024, 2)):
    """Load RadioML 2018 eval split. Returns (signals, labels)."""
    if hdf5_path.endswith('.npz'):
        data = np.load(hdf5_path)
        return data['signals'], data['labels']
        
    import h5py
    f = h5py.File(hdf5_path, 'r')
    cls = f['Y'][:].argmax(axis=-1).squeeze()
    snr = f['Z'][:].squeeze()
    if snr_range is None:
        snr_range = set(np.unique(snr))
    else:
        snr_range = set(snr_range)
    indices = [i for i, (c, s) in enumerate(zip(cls, snr))
               if s in snr_range]
    rng = np.random.default_rng(seed)
    indices = rng.permuted(indices)
    n_train = int(split_fractions[0] * len(indices))
    n_val = int(split_fractions[1] * len(indices))
    eval_indices = indices[n_train + n_val:]
    signals = np.stack([f['X'][int(i)].reshape(reshape) for i in eval_indices])
    labels = cls[eval_indices]
    f.close()
    return signals.astype(np.float32), labels


def run_finn_t_benchmark(deploy_dir, weights_dir, hdf5_path,
                         num_runs, warmup_batches, idle_seconds,
                         stabilize_seconds, results_dir, run_name):
    """Benchmark FINN-T RadioML transformer on ZU3EG."""
    driver_dir = os.path.join(deploy_dir, 'driver')
    bitfile    = os.path.join(deploy_dir, 'bitfile', 'finn-accel.bit')
    sys.path.insert(0, driver_dir)

    with open(os.path.join(driver_dir, 'settings.json')) as f:
        settings = json.load(f)['driver_information']

    io_shape_dict = settings['io_shape_dict']
    # Read shapes directly from settings — do NOT use the overlay's
    # ishape_normal()/oshape_normal() methods because those clobber dim 0
    # with batch_size.  The FINN-plus driver assumes dim 0 = batch, but
    # the transformer output [64, 96] has dim 0 = seq_len.  Calling
    # ol.execute() would crash in unfold_output() (tries to reshape 6144
    # elements into (1, 96)).  We bypass execute() and drive DMA manually.
    ishape_normal = tuple(io_shape_dict['ishape_normal'][0])
    ishape_folded = tuple(io_shape_dict['ishape_folded'][0])
    ishape_packed = tuple(io_shape_dict['ishape_packed'][0])
    oshape_normal = tuple(io_shape_dict['oshape_normal'][0])
    oshape_folded = tuple(io_shape_dict['oshape_folded'][0])
    oshape_packed = tuple(io_shape_dict['oshape_packed'][0])
    idt_str = io_shape_dict['idt'][0]
    odt_str = io_shape_dict['odt'][0]

    from driver import FINNDMAOverlay
    from finn.util.data_packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy
    from qonnx.core.datatype import DataType
    from pynq import allocate

    idt = DataType[idt_str.replace("DataType['", "").replace("']", "")]
    odt = DataType[odt_str.replace("DataType['", "").replace("']", "")]

    print(f"Loading RadioML 2018 eval set from {hdf5_path}...")
    signals, labels = load_radioml(hdf5_path)
    print(f"  {len(signals)} eval samples, signal shape {signals[0].shape}")

    print(f"Loading FINN-T bitfile: {bitfile}")
    ol = FINNDMAOverlay(
        bitfile_name=bitfile,
        platform=settings.get('platform', 'zynq-iodma'),
        io_shape_dict=io_shape_dict,
        batch_size=1,
        fclk_mhz=settings.get('fclk_mhz', 100.0),
    )
    print(f"  Accel input: {ishape_normal}  output: {oshape_normal}")

    # ---- Buffer allocation (C runner v2 selectable via FINN_T_OPT env var) ----
    #   baseline    -> v1 path: cached, single-buffered
    #   uncached    -> v2 with cacheable=False buffers, single-buffered (opt 1 only)
    #   doublebuf   -> v2 with cached buffers, two pairs (opt 2 only)
    #   both / auto -> v2 with cacheable=False buffers, two pairs (default)
    # Anything else logs a warning and uses 'both'.
    _finn_t_opt = os.environ.get('FINN_T_OPT', 'both').strip().lower()
    if _finn_t_opt in ('baseline', 'v1', 'cached'):
        _v2_requested, _v2_uncached, _v2_double = False, False, False
    elif _finn_t_opt == 'uncached':
        _v2_requested, _v2_uncached, _v2_double = True, True, False
    elif _finn_t_opt in ('doublebuf', 'double', 'db'):
        _v2_requested, _v2_uncached, _v2_double = True, False, True
    elif _finn_t_opt in ('both', 'optimized', 'auto'):
        _v2_requested, _v2_uncached, _v2_double = True, True, True
    else:
        print(f"  FINN_T_OPT={_finn_t_opt!r} not recognized; using 'both'")
        _v2_requested, _v2_uncached, _v2_double = True, True, True

    _alloc_kwargs = {'shape': ishape_packed, 'dtype': np.uint8}
    _alloc_kwargs_o = {'shape': oshape_packed, 'dtype': np.uint8}
    if _v2_requested and _v2_uncached:
        _alloc_kwargs['cacheable']  = False
        _alloc_kwargs_o['cacheable'] = False
    try:
        ibuf_device = allocate(**_alloc_kwargs)
        obuf_device = allocate(**_alloc_kwargs_o)
        ibuf_device_b = allocate(**_alloc_kwargs)  if _v2_double else None
        obuf_device_b = allocate(**_alloc_kwargs_o) if _v2_double else None
    except TypeError:
        # Older PYNQ that doesn't accept cacheable= kwarg — fall back to defaults.
        print("  pynq.allocate(cacheable=...) not supported here; "
              "falling back to default cached allocation")
        _v2_uncached = False
        ibuf_device = allocate(shape=ishape_packed, dtype=np.uint8)
        obuf_device = allocate(shape=oshape_packed, dtype=np.uint8)
        ibuf_device_b = allocate(shape=ishape_packed, dtype=np.uint8) if _v2_double else None
        obuf_device_b = allocate(shape=oshape_packed, dtype=np.uint8) if _v2_double else None

    # Load CPU tail weights (extracted from dataflow_parent.onnx)
    W_cls = np.load(os.path.join(weights_dir, 'finn_t_MatMul_7_param0.npy'))
    thres = np.load(os.path.join(weights_dir, 'finn_t_MultiThreshold_15_param0.npy'))
    dequant_scale = np.load(os.path.join(weights_dir, 'finn_t_Mul_12_param0.npy'))
    print(f"  CPU tail: MatMul {W_cls.shape}, MT {thres.shape}, scale {float(dequant_scale):.4f}")

    def multithreshold(x, thresholds):
        """x: (..., C), thresholds: (C, T) -> (..., C) int"""
        return np.sum(x[..., np.newaxis] > thresholds, axis=-1)

    def cpu_tail(accel_out):
        """accel_out: [64, 96] INT5 from FPGA → [24] float32 logits"""
        x = accel_out.astype(np.float32)
        x = x.T                                 # [96, 64]
        x = x[np.newaxis, ...]                  # [1, 96, 64]
        x = x.mean(axis=-1, keepdims=True)      # [1, 96, 1] GlobalAveragePool
        x = x.reshape(1, -1)                    # [1, 96]
        x = x @ W_cls                           # [1, 24]
        x = multithreshold(x, thres)            # [1, 24] int
        x = x[np.newaxis, ...]                  # [1, 1, 24]
        x = x.astype(np.float32) * dequant_scale  # [1, 1, 24] float
        return x.squeeze()                      # [24]

    def infer(signal):
        # Pack input (bypass ol.execute which clobbers dim 0)
        ibuf_normal = signal.reshape(ishape_normal)
        ibuf_folded = ibuf_normal.reshape(ishape_folded)
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded, idt, reverse_endian=True, reverse_inner=True, fast_mode=True)
        np.copyto(ibuf_device, ibuf_packed)
        ibuf_device.flush()
        # Raw DMA transfer
        ol.odma[0].write(0x10, obuf_device.device_address & 0xFFFFFFFF)
        ol.odma[0].write(0x14, (obuf_device.device_address >> 32) & 0xFFFFFFFF)
        ol.odma[0].write(0x1C, 1)
        ol.odma[0].write(0x00, 1)
        ol.idma[0].write(0x10, ibuf_device.device_address & 0xFFFFFFFF)
        ol.idma[0].write(0x14, (ibuf_device.device_address >> 32) & 0xFFFFFFFF)
        ol.idma[0].write(0x1C, 1)
        ol.idma[0].write(0x00, 1)
        # Wait for output DMA done
        status = ol.odma[0].read(0x00)
        while status & 0x2 == 0:
            status = ol.odma[0].read(0x00)
        # Unpack output with correct shapes
        obuf_device.invalidate()
        obuf_packed = np.array(obuf_device)
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed, odt, oshape_folded, reverse_endian=True, reverse_inner=True)
        accel_out = obuf_folded.reshape(oshape_normal)
        logits = cpu_tail(accel_out)
        return int(np.argmax(logits))

    # ---------- Optional C hot-path ----------
    # Try to load libfinn_t_infer.so from this script's dir. The C library
    # implements the per-iteration sequence (memcpy signal -> ibuf, DMA
    # trigger + poll, INT5 sign-extend, GAP, MatMul, argmax) so the Python
    # loop overhead doesn't cap the FPGA's measured throughput.
    # Falls back to the Python `infer` defined above if the .so isn't
    # present (i.e., on a host machine, or before the user has compiled it
    # on the board).
    c_runner = None
    c_runner_state = None
    use_v2 = False  # set if finn_t_runner_init succeeds for the v2 path
    _so_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'libfinn_t_infer.so')
    try:
        import ctypes
        _lib = ctypes.CDLL(_so_path)
        # v1 (cached, single-buffered) — always present
        _lib.finn_t_infer_one.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _lib.finn_t_infer_one.restype = ctypes.c_int
        _lib.finn_t_infer_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _lib.finn_t_infer_batch.restype = ctypes.c_int
        # v2 (configurable: uncached / double-buffered) — present in updated .so
        _has_v2 = True
        try:
            _lib.finn_t_runner_init.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            _lib.finn_t_runner_init.restype = ctypes.c_int
            _lib.finn_t_runner_destroy.argtypes = []
            _lib.finn_t_runner_destroy.restype = ctypes.c_int
            _lib.finn_t_infer_batch_v2.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            _lib.finn_t_infer_batch_v2.restype = ctypes.c_int
        except AttributeError:
            _has_v2 = False
            print("  C runner v2 symbols missing — rebuild libfinn_t_infer.so")

        _idma_mmio = ol.idma[0].mmio.array.ctypes.data
        _odma_mmio = ol.odma[0].mmio.array.ctypes.data

        _W_cls_int8 = np.ascontiguousarray(W_cls.astype(np.int8))
        _signals_c = np.ascontiguousarray(signals.astype(np.float32))
        _labels_c = np.ascontiguousarray(labels.astype(np.int32))
        _preds_c = np.zeros(len(signals), dtype=np.int32)

        c_runner = _lib
        c_runner_state = {
            'ibuf_virt':  ibuf_device.ctypes.data,
            'ibuf_phys':  int(ibuf_device.device_address),
            'obuf_virt':  obuf_device.ctypes.data,
            'obuf_phys':  int(obuf_device.device_address),
            'idma_mmio':  _idma_mmio,
            'odma_mmio':  _odma_mmio,
            'W_cls_int8': _W_cls_int8,
            'signals_c':  _signals_c,
            'labels_c':   _labels_c,
            'preds_c':    _preds_c,
        }
        print(f"  C runner loaded: {_so_path}")

        # Try the v2 path if requested and supported.
        if _v2_requested and _has_v2:
            _ib_b_virt = ibuf_device_b.ctypes.data if ibuf_device_b is not None else 0
            _ib_b_phys = int(ibuf_device_b.device_address) if ibuf_device_b is not None else 0
            _ob_b_virt = obuf_device_b.ctypes.data if obuf_device_b is not None else 0
            _ob_b_phys = int(obuf_device_b.device_address) if obuf_device_b is not None else 0
            _rc = _lib.finn_t_runner_init(
                2 if _v2_double else 1,
                0 if _v2_uncached else 1,
                ibuf_device.ctypes.data, int(ibuf_device.device_address),
                _ib_b_virt, _ib_b_phys,
                obuf_device.ctypes.data, int(obuf_device.device_address),
                _ob_b_virt, _ob_b_phys,
                _idma_mmio, _odma_mmio)
            if _rc == 0:
                use_v2 = True
                print(f"  C runner v2 initialized: "
                      f"buffers={'double' if _v2_double else 'single'}, "
                      f"cache_ops={'off' if _v2_uncached else 'on'}")
            else:
                print(f"  finn_t_runner_init returned {_rc}; falling back to v1 batch")
        elif not _v2_requested:
            print("  FINN_T_OPT=baseline -> using v1 batch")
    except OSError as e:
        print(f"  C runner not available ({e}); falling back to Python infer")

    config = {
        'toolchain': 'finn-t',
        'deploy_dir': deploy_dir,
        'weights_dir': weights_dir,
        'dataset': 'radioml2018',
        'batch_size': 1,
        'num_runs': num_runs,
        'num_images': len(signals),
        'signal_shape': list(signals[0].shape),
        'accel_input_shape': list(ishape_normal),
        'accel_output_shape': list(oshape_normal),
        # When running via the C path the CPU-tail ops are: sign-extend,
        # transpose, GAP, MatMul[96,24], argmax (no MultiThreshold or
        # dequant scale — they only affect logit calibration, not argmax).
        # Python fallback path retains the full tail.
        'cpu_tail_ops': (
            ['SignExtend', 'Transpose', 'GAP', 'MatMul[96,24]', 'argmax']
            if c_runner is not None else
            ['Transpose', 'Unsqueeze', 'GAP', 'Flatten',
             'MatMul[96,24]', 'MultiThreshold[24,255]',
             'Unsqueeze', 'Mul(scale)']),
        'runtime': (
            'c-v2-uncached-double' if (use_v2 and _v2_uncached and _v2_double) else
            'c-v2-uncached'        if (use_v2 and _v2_uncached) else
            'c-v2-double'          if (use_v2 and _v2_double) else
            'c-v2'                 if use_v2 else
            'c-v1'                 if c_runner is not None else
            'python'),
        'finn_t_opt_env': _finn_t_opt,
        'num_classes': 24,
        'timestamp': datetime.now().isoformat(),
        'board': 'AUP-ZU3',
        'fpga_part': 'xczu3eg-sbva484-1-e',
        'power_method': 'ina260' if POWER_AVAILABLE else 'none',
    }

    print(f"Thermal stabilization ({stabilize_seconds}s)...")
    time.sleep(stabilize_seconds)

    idle = measure_idle(idle_seconds)

    def _infer_c(signal):
        """Single-sample C inference, returns predicted class. Used for
        warmup (Python loop) so we can reuse the same code path."""
        s = c_runner_state
        return c_runner.finn_t_infer_one(
            np.ascontiguousarray(signal.astype(np.float32)).ctypes.data,
            s['ibuf_virt'], s['ibuf_phys'],
            s['obuf_virt'], s['obuf_phys'],
            s['idma_mmio'], s['odma_mmio'],
            s['W_cls_int8'].ctypes.data)

    print(f"Warmup ({warmup_batches} samples)...")
    _warmup_infer = _infer_c if c_runner is not None else infer
    for b in range(min(warmup_batches, len(signals))):
        _warmup_infer(signals[b])

    print(f"Running {num_runs} measured runs over {len(signals)} eval samples"
          f" via {'C runner' if c_runner is not None else 'Python infer'}...")
    all_runs = []
    for run in range(num_runs):
        power_log     = []
        sysmon_log    = []
        sampling_flag = [True]
        thread  = threading.Thread(target=make_sampler(power_log, sysmon_log, sampling_flag))
        thread.start()
        start_time = time.time()

        if use_v2:
            s = c_runner_state
            correct = c_runner.finn_t_infer_batch_v2(
                s['signals_c'].ctypes.data,
                s['labels_c'].ctypes.data,
                len(s['signals_c']),
                s['preds_c'].ctypes.data,
                s['W_cls_int8'].ctypes.data)
            total = len(s['signals_c'])
        elif c_runner is not None:
            s = c_runner_state
            correct = c_runner.finn_t_infer_batch(
                s['signals_c'].ctypes.data,
                s['labels_c'].ctypes.data,
                len(s['signals_c']),
                s['preds_c'].ctypes.data,
                s['ibuf_virt'], s['ibuf_phys'],
                s['obuf_virt'], s['obuf_phys'],
                s['idma_mmio'], s['odma_mmio'],
                s['W_cls_int8'].ctypes.data)
            total = len(s['signals_c'])
        else:
            correct = 0
            total   = 0
            for i in range(len(signals)):
                pred = infer(signals[i])
                if pred == labels[i]:
                    correct += 1
                total += 1

        elapsed = time.time() - start_time
        end_time = time.time()
        sampling_flag[0] = False
        thread.join()
        all_runs.append(build_run_result(run + 1, correct, total, elapsed, power_log, sysmon_log,
                                         t_start=start_time, t_end=end_time))

    if use_v2:
        # Best-effort cleanup; the runner state is process-local static so this
        # is mostly cosmetic. Buffers are freed when the pynq.allocate refs
        # go out of scope.
        c_runner.finn_t_runner_destroy()

    return config, idle, all_runs


# ============================================================
# DPU benchmark (VART on PetaLinux, no XRT/PYNQ)
# ============================================================
def run_dpu_benchmark(model_path, dataset, batch_size, num_runs,
                      warmup_batches, idle_seconds, stabilize_seconds,
                      results_dir, run_name):
    import xir
    import vart

    print(f"Loading {dataset} dataset...")
    images, labels = load_mnist() if dataset == 'mnist' else load_cifar10()
    print(f"  {len(images)} images, shape {images[0].shape}")

    print(f"Loading xmodel: {model_path}")
    graph = xir.Graph.deserialize(model_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = [s for s in subgraphs
                    if s.has_attr('device') and s.get_attr('device') == 'DPU'][0]
    print(f"  DPU subgraph: {dpu_subgraph.get_name()}")

    runner = vart.Runner.create_runner(dpu_subgraph, 'run')
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_shape = tuple(input_tensors[0].dims)
    output_shape = tuple(output_tensors[0].dims)
    print(f"  Input:  {input_tensors[0].name}, shape={input_shape}, dtype={input_tensors[0].dtype}")
    print(f"  Output: {output_tensors[0].name}, shape={output_shape}, dtype={output_tensors[0].dtype}")

    config = {
        'toolchain': 'dpu',
        'model_path': model_path,
        'dataset': dataset,
        'batch_size': 1,
        'num_runs': num_runs,
        'num_images': len(images),
        'image_shape': list(images[0].shape),
        'dpu_input_shape': list(input_shape),
        'dpu_output_shape': list(output_shape),
        'timestamp': datetime.now().isoformat(),
        'board': 'AUP-ZU3',
        'dpu': 'DPUCZDX8G_ISA1_B512',
        'power_method': 'ina260' if POWER_AVAILABLE else 'none',
    }

    print(f"Thermal stabilization ({stabilize_seconds}s)...")
    time.sleep(stabilize_seconds)

    idle = measure_idle(idle_seconds)

    print(f"Warmup ({warmup_batches} images)...")
    for i in range(warmup_batches):
        inp = (images[i].astype(np.float32)).reshape(input_shape)
        out = np.empty(output_shape, dtype=np.float32)
        job_id = runner.execute_async([inp], [out])
        runner.wait(job_id)

    print(f"Running {num_runs} measured runs...")
    all_runs = []
    for run in range(num_runs):
        power_log = []
        sysmon_log = []
        sampling_flag = [True]
        thread = threading.Thread(target=make_sampler(power_log, sysmon_log, sampling_flag))
        correct = 0
        total = 0
        thread.start()
        start_time = time.time()

        for i in range(len(images)):
            inp = (images[i].astype(np.float32)).reshape(input_shape)
            out = np.empty(output_shape, dtype=np.float32)
            job_id = runner.execute_async([inp], [out])
            runner.wait(job_id)
            pred = int(np.argmax(out.flatten()))
            if pred == labels[i]:
                correct += 1
            total += 1

        elapsed = time.time() - start_time
        end_time = time.time()
        sampling_flag[0] = False
        thread.join()
        all_runs.append(build_run_result(run + 1, correct, total, elapsed, power_log, sysmon_log,
                                         t_start=start_time, t_end=end_time))

    del runner
    return config, idle, all_runs

# ============================================================
# VTA benchmark (board-side, pre-compiled modules, no RPC)
# Supports both MLP and CNN models via config.json model_type
# ============================================================
def run_vta_benchmark(model_dir, dataset, batch_size, num_runs,
                      warmup_batches, idle_seconds, stabilize_seconds,
                      results_dir, run_name):
    import tvm
    import tvm.runtime
    import vta
    import math

    print(f"Loading {dataset} dataset...")
    images, labels = load_mnist() if dataset == 'mnist' else load_cifar10()
    print(f"  {len(images)} images, shape {images[0].shape}")

    # ---- Load VTA model config ----
    import json as _json
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as f:
        vta_config = _json.load(f)

    env = vta.get_env()
    print(f"  VTA env: BATCH={env.BATCH}, BLOCK_IN={env.BLOCK_IN}, BLOCK_OUT={env.BLOCK_OUT}")

    model_type = vta_config.get('model_type', 'mlp')
    num_layers = vta_config['num_layers']
    is_cnn = model_type.startswith('cnn')
    requant_mode = vta_config.get('requant_mode', 'cpu_per_image')
    print(f"  Model type: {model_type}, layers: {num_layers}, requant: {requant_mode}")

    if not is_cnn:
        raw_dims = vta_config['architecture']
        print(f"  Architecture: {' -> '.join(str(d) for d in raw_dims)}")

    # ---- Clear stale PYNQ state ----
    stale_json = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale_json):
            os.remove(stale_json)
    except Exception:
        pass

    # ---- Load bitstream ----
    bitstream_name = vta_config.get('bitstream', '1x16_i8w8a32_15_15_18_17.bit')
    bitstream_path = None
    for candidate in [
        f'/root/.vta_cache/ultra96/0_0_2/{bitstream_name}',
        f'/home/xilinx/.vta_cache/ultra96/0_0_2/{bitstream_name}',
        os.path.join(model_dir, bitstream_name),
    ]:
        if os.path.exists(candidate):
            bitstream_path = candidate
            break

    if bitstream_path is None:
        print(f"ERROR: Bitstream {bitstream_name} not found")
        sys.exit(1)

    print(f"  Loading bitstream: {bitstream_path}")
    try:
        from pynq import Overlay
        overlay = Overlay(bitstream_path)
        print(f"  Overlay loaded, IPs: {list(overlay.ip_dict.keys())}")
    except Exception as e:
        print(f"  Overlay load failed: {e}")
        print(f"  Trying without Overlay (bitstream may already be loaded)...")

    # ---- Load VTA runtime library ----
    import ctypes
    vta_lib = None
    for candidate in [
        '/home/xilinx/tvm-src/build/libvta.so',
        os.path.join(os.environ.get('TVM_HOME', ''), 'build/libvta.so'),
    ]:
        if os.path.exists(candidate):
            vta_lib = candidate
            break
    if vta_lib is None:
        print("ERROR: libvta.so not found. Check TVM build on board.")
        sys.exit(1)
    print(f"  Loading VTA runtime: {vta_lib}")
    ctypes.CDLL(vta_lib, ctypes.RTLD_GLOBAL)

    # ---- Get VTA device context ----
    ctx = tvm.device("ext_dev", 0)

    # ---- Load pre-compiled modules and weights ----
    layer_info = vta_config['layers']
    gemm_modules = []
    W_nds = []
    D_nds = []  # bias VTA tensors (vta_native only, None for cpu_per_image)
    layer_meta = []
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT

    for lc in layer_info:
        mod_file = lc['module_file']
        # Try .so first (linked), then .o
        so_file = mod_file.replace('.o', '.so') if mod_file.endswith('.o') else mod_file
        mod_path = os.path.join(model_dir, so_file)
        if not os.path.exists(mod_path):
            mod_path = os.path.join(model_dir, mod_file)

        print(f"  Loading layer {lc['index']}: {mod_path}")
        f = tvm.runtime.load_module(mod_path)
        gemm_modules.append(f)

        W_tiled = np.load(os.path.join(model_dir, lc['weight_file']))
        if requant_mode in ('vta_native', 'vta_native_o8'):
            W_tiled = pack_int4_for_vta(W_tiled)
        W_nd = tvm.nd.array(W_tiled, ctx)
        W_nds.append(W_nd)

        has_vta_bias = lc.get('has_vta_bias', False)
        bias_data = np.load(os.path.join(model_dir, lc['bias_file']))

        if requant_mode in ('vta_native', 'vta_native_o8') and has_vta_bias:
            # Bias is int32, loaded to VTA as 4D: (o_tile, m, 1, BLOCK_OUT).
            # For CNN, broadcast (m, BLOCK_OUT) over the o_tile dimension.
            # For MLP (o=1), the reshape is trivial but still produces 4D.
            bias_i32 = bias_data.astype(np.int32)
            o_t = lc.get('o_tile', 1)
            m_t = lc.get('m_tiles', lc.get('m', 1))
            bias_i32 = np.ascontiguousarray(
                np.broadcast_to(
                    bias_i32.reshape(1, m_t, 1, BLOCK_OUT),
                    (o_t, m_t, 1, BLOCK_OUT)),
                dtype=np.int32)
            D_nd = tvm.nd.array(bias_i32, ctx)
            D_nds.append(D_nd)
            b_float_for_meta = None
        else:
            # Bias is float32, applied CPU-side (INT8 path or last layer)
            D_nds.append(None)
            b_float_for_meta = bias_data.astype(np.float32)

        meta = {
            'in_f': lc['in_f'],
            'out_f': lc['out_f'],
            'real_out': lc['real_out'],
            'n_tiles': lc['n_tiles'],
            'm_tiles': lc['m_tiles'],
            'shift': lc['shift'],
            'w_scale': lc['w_scale'],
            'b_float': b_float_for_meta,
            'has_vta_bias': has_vta_bias,
            'in_scale': lc.get('in_scale', 0),
        }
        if is_cnn:
            meta['type'] = lc['type']
            meta['o_total'] = lc['o_total']
            meta['o_tile'] = lc['o_tile']
            meta['n_chunks'] = lc['n_chunks']
            if lc['type'] == 'conv':
                meta['kernel_size'] = lc['kernel_size']
                meta['padding'] = lc['padding']
                meta['in_channels'] = lc['in_channels']
                meta['out_channels'] = lc['out_channels']
                meta['pool'] = lc.get('pool', 0)
        layer_meta.append(meta)

    # ---- Pre-allocate VTA buffers ----
    A_nds = []
    C_nds = []
    if is_cnn:
        for lm in layer_meta:
            A_nds.append(tvm.nd.array(
                np.zeros((lm['o_tile'], lm['n_tiles'], 1, BLOCK_IN), dtype=np.int8), ctx))
            C_nds.append(tvm.nd.array(
                np.zeros((lm['o_tile'], lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8), ctx))
    else:
        for lm in layer_meta:
            A_nds.append(tvm.nd.array(
                np.zeros((1, lm['n_tiles'], 1, BLOCK_IN), dtype=np.int8), ctx))
            C_nds.append(tvm.nd.array(
                np.zeros((1, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8), ctx))

    # ---- CNN helper functions ----
    def im2col(x, kH, kW, pad=0, stride=1):
        """x: (H, W, C) -> (H_out*W_out, kH*kW*C)"""
        H, W, C = x.shape
        H_out = (H + 2 * pad - kH) // stride + 1
        W_out = (W + 2 * pad - kW) // stride + 1
        if pad > 0:
            x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        patches = np.zeros((H_out * W_out, kH * kW * C), dtype=x.dtype)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patches[idx] = x[i*stride:i*stride+kH, j*stride:j*stride+kW, :].flatten()
                idx += 1
        return patches, H_out, W_out

    def maxpool2d(x, pool_size=2):
        """x: (H, W, C) -> (H//pool, W//pool, C)"""
        H, W, C = x.shape
        H_out = H // pool_size
        W_out = W // pool_size
        out = np.zeros((H_out, W_out, C), dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                out[i, j] = x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size].max(axis=(0, 1))
        return out

    # ---- Inference functions ----
    if is_cnn:
        def infer_one(img_2d):
            """Run single image through VTA CNN. img_2d: (28, 28) float32 [0,1]."""
            x_s = np.max(np.abs(img_2d)) / 127.0 if np.max(np.abs(img_2d)) > 0 else 1e-10
            current_scale = x_s
            h_float = img_2d

            for i, (lm, gm) in enumerate(zip(layer_meta, gemm_modules)):
                if lm['type'] == 'conv':
                    if h_float.ndim == 2:
                        h_spatial = h_float[:, :, np.newaxis]
                    else:
                        h_spatial = h_float

                    patches, H_out, W_out = im2col(
                        h_spatial, lm['kernel_size'], lm['kernel_size'],
                        pad=lm['padding'])

                    real_dim = patches.shape[1]
                    if real_dim < lm['in_f']:
                        patches = np.pad(patches, ((0, 0), (0, lm['in_f'] - real_dim)),
                                         mode='constant')

                    p_int8 = np.clip(np.round(patches / current_scale), -128, 127).astype(np.int8)

                    o_total = lm['o_total']
                    o_tile = lm['o_tile']
                    n_chunks = lm['n_chunks']
                    vta_out_full = np.zeros((o_total, lm['out_f']), dtype=np.int8)

                    for chunk in range(n_chunks):
                        start = chunk * o_tile
                        end = start + o_tile
                        p_tiled = p_int8[start:end].reshape(o_tile, lm['n_tiles'], 1, BLOCK_IN)
                        A_nds[i].copyfrom(p_tiled)
                        C_nds[i].copyfrom(np.zeros((o_tile, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                        gm(A_nds[i], W_nds[i], C_nds[i])
                        vta_out_full[start:end] = C_nds[i].numpy().reshape(o_tile, lm['out_f'])

                    combined = current_scale * lm['w_scale'] * (2 ** lm['shift'])
                    y_float = vta_out_full.astype(np.float32) * combined + lm['b_float'][:lm['out_f']]
                    y_float = np.maximum(y_float, 0)
                    y_spatial = y_float[:, :lm['real_out']].reshape(H_out, W_out, lm['real_out'])

                    if lm.get('pool', 0) > 0:
                        y_spatial = maxpool2d(y_spatial, lm['pool'])

                    h_float = y_spatial
                    next_scale = np.max(np.abs(h_float)) / 127.0
                    current_scale = max(next_scale, 1e-10)

                elif lm['type'] == 'dense':
                    h_vec = h_float.mean(axis=(0, 1))
                    if len(h_vec) < lm['in_f']:
                        h_vec_padded = np.zeros(lm['in_f'], dtype=np.float32)
                        h_vec_padded[:len(h_vec)] = h_vec
                        h_vec = h_vec_padded

                    h_int8 = np.clip(np.round(h_vec / current_scale), -128, 127).astype(np.int8)
                    h_tiled = h_int8.reshape(1, lm['n_tiles'], 1, BLOCK_IN)
                    A_nds[i].copyfrom(h_tiled)
                    C_nds[i].copyfrom(np.zeros((1, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                    gm(A_nds[i], W_nds[i], C_nds[i])

                    vta_out = C_nds[i].numpy().reshape(lm['out_f'])
                    combined = current_scale * lm['w_scale'] * (2 ** lm['shift'])
                    y_float = vta_out.astype(np.float32) * combined + lm['b_float'][:lm['out_f']]
                    return int(np.argmax(y_float[:lm['real_out']]))

            raise RuntimeError("No dense layer found at end of CNN")

        # Images: (N, 1, 28, 28) -> (N, 28, 28) for CNN
        images_2d = images.reshape(len(images), images.shape[2], images.shape[3])

        # ---- INT4-o8 CNN closure (Mode G, per-channel) ----
        if requant_mode == 'vta_native_o8':
            _ZP = vta_config.get('zero_point', 8)
            _act_scale = vta_config['act_scales_brevitas']

            def _im2col_chw(x_chw, kH, kW, pad, pad_value=0):
                """CHW im2col with configurable pad value for Mode G."""
                C, H, W = x_chw.shape
                if pad > 0:
                    x_p = np.full((C, H+2*pad, W+2*pad), pad_value, dtype=x_chw.dtype)
                    x_p[:, pad:pad+H, pad:pad+W] = x_chw
                else:
                    x_p = x_chw
                oH = H + 2*pad - kH + 1
                oW = W + 2*pad - kW + 1
                cols = np.empty((oH*oW, kH*kW*C), dtype=x_chw.dtype)
                idx = 0
                for ii in range(oH):
                    for jj in range(oW):
                        cols[idx] = x_p[:, ii:ii+kH, jj:jj+kW].transpose(1,2,0).reshape(-1)
                        idx += 1
                return cols

            def _maxpool_chw(x, k, s):
                C, H, W = x.shape
                oH, oW = (H-k)//s+1, (W-k)//s+1
                sh = (C, oH, oW, k, k)
                st = (x.strides[0], x.strides[1]*s, x.strides[2]*s,
                      x.strides[1], x.strides[2])
                return np.lib.stride_tricks.as_strided(
                    x, shape=sh, strides=st, writeable=False).max(axis=(3,4))

            def infer_one_vta_native_o8_cnn(img_2d):
                """INT4-o8 CNN via Mode G: offset-encoded int4 input, int8 output,
                per-channel dequant, im2col + o-tiling, all 4-arg VTA calls."""
                x_bre = np.clip(np.round(img_2d[None, :, :] / _act_scale[0]),
                               0, 15).astype(np.int32)
                x_vta = (x_bre - _ZP).astype(np.int8)

                for i, (lm, gm) in enumerate(zip(layer_meta, gemm_modules)):
                    if lm['type'] == 'conv':
                        o_tile = lm['o_tile']
                        n_chunks = lm['n_chunks']
                        C_out = lm['real_out']

                        patches = _im2col_chw(x_vta, 3, 3, pad=1, pad_value=-_ZP)
                        n_t = lm['n_tiles']
                        if patches.shape[1] < n_t * BLOCK_IN:
                            patches = np.pad(patches,
                                ((0, 0), (0, n_t * BLOCK_IN - patches.shape[1])))
                        spatial = patches.shape[0]

                        full_int8 = np.zeros((spatial, BLOCK_OUT), dtype=np.int8)
                        for ch in range(n_chunks):
                            st = ch * o_tile
                            en = min(st + o_tile, spatial)
                            ao = en - st
                            if ao <= 0: break
                            a = patches[st:en].reshape(
                                ao, n_t, 1, BLOCK_IN).astype(np.int8)
                            if ao < o_tile:
                                a = np.pad(a, ((0, o_tile-ao),(0,0),(0,0),(0,0)))
                            A_nds[i].copyfrom(pack_int4_for_vta(a))
                            C_nds[i].copyfrom(np.zeros(
                                (o_tile, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                            gm(A_nds[i], W_nds[i], D_nds[i], C_nds[i])
                            full_int8[st:en] = C_nds[i].numpy()[:ao, 0, 0, :]

                        H = int(math.sqrt(spatial))
                        out_chw = full_int8[:, :C_out].reshape(
                            H, H, C_out).transpose(2, 0, 1)
                        w_sc = np.array(lm['w_scale'], dtype=np.float64)
                        cs = w_sc * _act_scale[i] * (2.0 ** lm['shift'])
                        fl = out_chw.astype(np.float64) * cs[:, None, None]
                        pooled = _maxpool_chw(np.maximum(fl, 0.0), 2, 2)
                        x_bre = np.clip(np.round(pooled / _act_scale[i + 1]),
                                       0, 15).astype(np.int32)
                        x_vta = (x_bre - _ZP).astype(np.int8)

                    elif lm['type'] == 'dense':
                        last_s = _act_scale[len([
                            l for l in layer_meta if l.get('type') == 'conv'])]
                        x_fl = (x_vta.astype(np.int32) + _ZP).astype(
                            np.float64) * last_s
                        x_avg = x_fl.mean(axis=(1, 2))
                        x_d = np.clip(np.round(x_avg / last_s),
                                     0, 15).astype(np.int32)
                        x_d_vta = (x_d - _ZP).astype(np.int8)
                        x_pad = np.zeros(BLOCK_IN, dtype=np.int8)
                        x_pad[:len(x_d_vta)] = x_d_vta
                        a_d = pack_int4_for_vta(
                            x_pad.reshape(1, lm['n_tiles'], 1, BLOCK_IN))
                        A_nds[i].copyfrom(a_d)
                        C_nds[i].copyfrom(np.zeros(
                            (1, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                        gm(A_nds[i], W_nds[i], D_nds[i], C_nds[i])
                        d_int8 = C_nds[i].numpy()[0, 0, 0, :]
                        w_d = np.array(lm['w_scale'], dtype=np.float64)
                        cs_d = w_d * last_s * (2.0 ** lm['shift'])
                        d_fl = d_int8[:lm['real_out']].astype(np.float64) * cs_d
                        return int(np.argmax(d_fl))

                raise RuntimeError("No dense layer found")

            infer_one = infer_one_vta_native_o8_cnn

    else:
        def infer_one_cpu_per_image(img_flat):
            """Run single image through VTA MLP. Returns predicted class.
            INT8 path: per-image calibration, CPU-side dequant/requant."""
            x_abs_max = np.max(np.abs(img_flat))
            x_s = x_abs_max / 127.0 if x_abs_max > 0 else 1e-10
            h_int8 = np.clip(np.round(img_flat / x_s), -128, 127).astype(np.int8)
            current_scale = x_s

            for i, (lm, f) in enumerate(zip(layer_meta, gemm_modules)):
                x_tiled = h_int8.reshape(1, lm['n_tiles'], 1, BLOCK_IN)
                A_nds[i].copyfrom(x_tiled)
                C_nds[i].copyfrom(np.zeros((1, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                f(A_nds[i], W_nds[i], C_nds[i])
                vta_out = C_nds[i].numpy().reshape(-1)[:lm['out_f']]

                combined = current_scale * lm['w_scale'] * (2 ** lm['shift'])
                y_float = vta_out.astype(np.float32) * combined + lm['b_float']

                if i < num_layers - 1:
                    y_float = np.maximum(y_float, 0)
                    y_abs_max = np.max(np.abs(y_float))
                    current_scale = y_abs_max / 127.0 if y_abs_max > 0 else 1e-10
                    h_int8 = np.clip(np.round(y_float / current_scale), -128, 127).astype(np.int8)
                else:
                    return int(np.argmax(y_float[:lm['real_out']]))

        def infer_one_vta_native(img_flat):
            """Run single image through VTA MLP. Returns predicted class.
            INT4 vta_native path: fixed learned scales, bias inside VTA for
            hidden layers, CPU-side float bias for the last layer only.
            VTA output is passed directly to the next layer (no CPU requant).
            Input and inter-layer activations are nibble-packed for VTA int4."""
            input_scale = vta_config['input_scale']
            input_clip_max = vta_config['input_clip_max']
            h_int8 = np.clip(np.round(img_flat / input_scale),
                             0, input_clip_max).astype(np.int8)

            for i, (lm, f) in enumerate(zip(layer_meta, gemm_modules)):
                # Pack int4 values into nibble pairs for VTA DMA
                x_packed = pack_int4_for_vta(h_int8)
                x_tiled = x_packed.reshape(1, lm['n_tiles'], 1, BLOCK_IN)
                A_nds[i].copyfrom(x_tiled)
                C_nds[i].copyfrom(np.zeros((1, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))

                if lm['has_vta_bias']:
                    # Hidden layer: 4-arg call (A, W, D, C).
                    # VTA does GEMM + bias_int + SHR + CLIP.
                    # Output is packed int4 — unpack for next layer.
                    f(A_nds[i], W_nds[i], D_nds[i], C_nds[i])
                    vta_out_packed = C_nds[i].numpy().reshape(-1)
                    vta_out = unpack_int4_from_vta(
                        vta_out_packed)[:lm['real_out']]
                    h_int8 = vta_out.copy()
                else:
                    # Last layer: 3-arg call (A, W, C).
                    # VTA does GEMM + SHR + CLIP (no bias).
                    # Unpack, then CPU adds float bias and takes argmax.
                    f(A_nds[i], W_nds[i], C_nds[i])
                    vta_out_packed = C_nds[i].numpy().reshape(-1)
                    vta_out = unpack_int4_from_vta(
                        vta_out_packed)[:lm['out_f']]
                    combined = lm['in_scale'] * lm['w_scale'] * (2 ** lm['shift'])
                    y_float = (vta_out[:lm['real_out']].astype(np.float32)
                               * combined + lm['b_float'][:lm['real_out']])
                    return int(np.argmax(y_float))

        if requant_mode == 'vta_native':
            infer_one = infer_one_vta_native
        else:
            infer_one = infer_one_cpu_per_image

        # Images: (N, 1, 28, 28) -> (N, 784) for MLP
        images_2d = None  # not used
        images_flat = images.reshape(len(images), -1)

    # ---- Config dict ----
    config = {
        'toolchain': 'vta',
        'model_type': model_type,
        'model_dir': model_dir,
        'dataset': dataset,
        'batch_size': 1,
        'num_runs': num_runs,
        'num_images': len(images),
        'image_shape': list(images[0].shape),
        'shift_amounts': [lm['shift'] for lm in layer_meta],
        'vta_clock_mhz': vta_config.get('clock_mhz', 250),
        'timestamp': datetime.now().isoformat(),
        'board': 'AUP-ZU3',
        'power_method': 'ina260' if POWER_AVAILABLE else 'none',
    }
    if not is_cnn:
        config['architecture'] = raw_dims

    print(f"Thermal stabilization ({stabilize_seconds}s)...")
    time.sleep(stabilize_seconds)

    idle = measure_idle(idle_seconds)

    # ---- Get the right image array ----
    if is_cnn:
        img_array = images_2d
    else:
        img_array = images_flat

    print(f"Warmup ({warmup_batches} images)...")
    for i in range(min(warmup_batches, len(img_array))):
        infer_one(img_array[i])

    # ---- Verification (first 100 images) ----
    print("Quick verification (100 images)...")
    verify_correct = 0
    for i in range(min(100, len(img_array))):
        pred = infer_one(img_array[i])
        if pred == labels[i]:
            verify_correct += 1
    print(f"  Verification accuracy: {verify_correct}/100 = {verify_correct}%")
    if not is_cnn and verify_correct < 90:
        print("  WARNING: Accuracy suspiciously low. Check bitstream and modules.")
    if is_cnn and verify_correct < 80:
        print("  WARNING: Accuracy suspiciously low. Check bitstream and modules.")

    print(f"Running {num_runs} measured runs...")
    all_runs = []
    for run in range(num_runs):
        power_log = []
        sysmon_log = []
        sampling_flag = [True]
        thread = threading.Thread(target=make_sampler(power_log, sysmon_log, sampling_flag))
        correct = 0
        total = 0
        thread.start()
        start_time = time.time()

        for i in range(len(img_array)):
            pred = infer_one(img_array[i])
            if pred == labels[i]:
                correct += 1
            total += 1

        elapsed = time.time() - start_time
        end_time = time.time()
        sampling_flag[0] = False
        thread.join()
        all_runs.append(build_run_result(run + 1, correct, total, elapsed, power_log, sysmon_log,
                                         t_start=start_time, t_end=end_time))

    return config, idle, all_runs

# ============================================================
# Shared result saving and summary
# ============================================================
def save_results(config, idle, all_runs, run_name, dataset, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    actual_batch = config.get('batch_size', 1)
    has_power    = any(r['avg_power_w'] is not None for r in all_runs)

    summary = {
        'accuracy':            float(np.mean([r['accuracy']       for r in all_runs])),
        'throughput_fps_mean': float(np.mean([r['throughput_fps'] for r in all_runs])),
        'throughput_fps_std':  float(np.std( [r['throughput_fps'] for r in all_runs])),
        'latency_ms_mean':     float(np.mean([r['latency_ms']     for r in all_runs])),
        'latency_ms_std':      float(np.std( [r['latency_ms']     for r in all_runs])),
        'idle_power_w':        idle['power']['mean'],
        'idle_power_std':      idle['power']['std'],
        'idle_temp_pl_c':      idle['sysmon']['temp_pl_c'],
        'avg_power_w_mean':    float(np.mean([r['avg_power_w']          for r in all_runs])) if has_power else None,
        'avg_power_w_std':     float(np.std( [r['avg_power_w']          for r in all_runs])) if has_power else None,
        'dynamic_power_w':     (float(np.mean([r['avg_power_w'] for r in all_runs])) - idle['power']['mean'])
                               if has_power and idle['power']['mean'] is not None else None,
        'energy_per_image_mj_mean': float(np.mean([r['energy_per_image_mj'] for r in all_runs])) if has_power else None,
        'energy_per_image_mj_std':  float(np.std( [r['energy_per_image_mj'] for r in all_runs])) if has_power else None,
    }

    sysmon_runs = [r['sysmon'] for r in all_runs if r['sysmon'] is not None]
    if sysmon_runs:
        summary['sysmon'] = {
            'temp_ps_c_mean':  float(np.mean([s['temp_ps_c_mean'] for s in sysmon_runs])),
            'temp_pl_c_mean':  float(np.mean([s['temp_pl_c_mean'] for s in sysmon_runs])),
            'temp_pl_c_max':   float(np.max( [s['temp_pl_c_max']  for s in sysmon_runs])),
            'vccint_v_mean':   float(np.mean([s['vccint_v_mean']  for s in sysmon_runs])),
        }

    output = {'config': config, 'idle': idle, 'runs': all_runs, 'summary': summary}

    filename = f"{run_name}_{dataset}_b{actual_batch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Toolchain:   {config['toolchain']}")
    print(f"  Model:       {run_name}")
    print(f"  Dataset:     {dataset}")
    print(f"  Batch size:  {actual_batch}")
    print(f"  Accuracy:    {summary['accuracy']:.2f}%")
    print(f"  Throughput:  {summary['throughput_fps_mean']:.1f} +/- {summary['throughput_fps_std']:.1f} FPS")
    print(f"  Latency:     {summary['latency_ms_mean']:.3f} +/- {summary['latency_ms_std']:.3f} ms")
    if has_power:
        print(f"  Idle power:  {summary['idle_power_w']:.3f} +/- {summary['idle_power_std']:.3f} W")
        print(f"  Avg power:   {summary['avg_power_w_mean']:.3f} +/- {summary['avg_power_w_std']:.3f} W")
        print(f"  Dynamic pwr: {summary['dynamic_power_w']:.3f} W")
        print(f"  Energy/img:  {summary['energy_per_image_mj_mean']:.4f} +/- {summary['energy_per_image_mj_std']:.4f} mJ")
    else:
        print(f"  Power:       collected host-side (FNB58 via fnb58_logger.py)")
    if 'sysmon' in summary:
        print(f"  PS temp:     {summary['sysmon']['temp_ps_c_mean']:.1f} C")
        print(f"  PL temp:     {summary['sysmon']['temp_pl_c_mean']:.1f} C  (max: {summary['sysmon']['temp_pl_c_max']:.1f} C)")
        print(f"  VCCINT:      {summary['sysmon']['vccint_v_mean']:.3f} V")
    print(f"  Saved to:    {filepath}")
    print(f"{'='*60}")
    return output

# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--toolchain',   required=True, choices=['vitis_ai', 'finn', 'finn-t', 'dpu', 'vta'])
    parser.add_argument('--model',       required=True,
                        help='Path to xmodel (vitis_ai/dpu), deploy/ dir (finn/finn-t), or model dir (vta)')
    parser.add_argument('--name',        default=None)
    parser.add_argument('--dataset',     required=True, choices=['mnist', 'cifar10', 'radioml2018'])
    parser.add_argument('--weights-dir', default=None,
                        help='CPU tail weights dir for finn-t (default: <model>/weights)')
    parser.add_argument('--hdf5',        default=None,
                        help='RadioML 2018 HDF5 path (required for --dataset radioml2018)')
    parser.add_argument('--batch',       type=int, default=1)
    parser.add_argument('--runs',        type=int, default=5)
    parser.add_argument('--stabilize',   type=int, default=10)
    parser.add_argument('--idle',        type=int, default=10)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--finn-runtime', choices=['python', 'c'], default='python',
                        help='FINN MLP inference runtime: "c" uses libfinn_mlp_infer.so, '
                             '"python" uses the numpy infer loop (default). Only applies '
                             'to --toolchain finn with MLP (has_mlp_pre) deploys.')
    args = parser.parse_args()

    if args.results_dir is None:
        for p in ['/home/petalinux/results', '/home/xilinx/results']:
            if os.path.exists(os.path.dirname(p)):
                args.results_dir = p
                break
        else:
            args.results_dir = './results'

    run_name = args.name or os.path.basename(args.model.rstrip('/'))

    # ---- Clock sanity check ----
    if time.gmtime().tm_year > 2030:
        print("ERROR: Board clock not synced (year > 2030).")
        print("  From host: ssh xilinx@192.168.3.1 \"sudo date -s '$(date -u +%%Y-%%m-%%d\\ %%H:%%M:%%S)'\"")
        print("  Or manually: sudo date -s \"YYYY-MM-DD HH:MM:SS\"  (UTC from host: date -u)")
        sys.exit(1)

    if args.toolchain == 'vitis_ai':
        config, idle, all_runs = run_vitisai_benchmark(
            model_path=args.model, dataset=args.dataset, batch_size=args.batch,
            num_runs=args.runs, warmup_batches=10, idle_seconds=args.idle,
            stabilize_seconds=args.stabilize, results_dir=args.results_dir,
            run_name=run_name)
    elif args.toolchain == 'finn':
        config, idle, all_runs = run_finn_benchmark(
            deploy_dir=args.model, dataset=args.dataset, batch_size=args.batch,
            num_runs=args.runs, warmup_batches=10, idle_seconds=args.idle,
            stabilize_seconds=args.stabilize, results_dir=args.results_dir,
            run_name=run_name, finn_runtime=args.finn_runtime)
    elif args.toolchain == 'finn-t':
        if args.dataset != 'radioml2018':
            print("ERROR: finn-t requires --dataset radioml2018")
            sys.exit(1)
        if args.hdf5 is None:
            for p in ['/home/xilinx/data/radioml2018_eval.npz',
                       '/home/xilinx/data/RML2018.hdf5',
                       '/home/petalinux/data/RML2018.hdf5']:
                if os.path.exists(p):
                    args.hdf5 = p
                    break
            else:
                print("ERROR: RadioML HDF5 not found. Specify --hdf5 <path>")
                sys.exit(1)
        weights_dir = args.weights_dir or os.path.join(args.model, 'weights')
        config, idle, all_runs = run_finn_t_benchmark(
            deploy_dir=args.model, weights_dir=weights_dir, hdf5_path=args.hdf5,
            num_runs=args.runs, warmup_batches=10, idle_seconds=args.idle,
            stabilize_seconds=args.stabilize, results_dir=args.results_dir,
            run_name=run_name)
    elif args.toolchain == 'dpu':
        config, idle, all_runs = run_dpu_benchmark(
            model_path=args.model, dataset=args.dataset, batch_size=args.batch,
            num_runs=args.runs, warmup_batches=10, idle_seconds=args.idle,
            stabilize_seconds=args.stabilize, results_dir=args.results_dir,
            run_name=run_name)
    elif args.toolchain == 'vta':
        config, idle, all_runs = run_vta_benchmark(
            model_dir=args.model, dataset=args.dataset, batch_size=args.batch,
            num_runs=args.runs, warmup_batches=10, idle_seconds=args.idle,
            stabilize_seconds=args.stabilize, results_dir=args.results_dir,
            run_name=run_name)

    save_results(config, idle, all_runs, run_name, args.dataset, args.results_dir)
