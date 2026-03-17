import numpy as np
import time
import json
import os
import struct
import gzip
import pickle
import threading
import glob
from datetime import datetime

# ============================================================
# Power measurement
# ============================================================
def find_power_file():
    for f in glob.glob('/sys/class/hwmon/hwmon*/name'):
        with open(f) as fh:
            if 'ina260' in fh.read():
                return f.rsplit('/', 1)[0] + '/power1_input'
    return None

POWER_FILE = find_power_file()

def read_power():
    with open(POWER_FILE) as f:
        return int(f.read().strip()) / 1e6

# ============================================================
# Data loaders
# ============================================================
def load_mnist(path='/home/ubuntu/MNIST/raw'):
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

def load_cifar10(path='/home/ubuntu/cifar-10-batches-py/test_batch'):
    with open(path, 'rb') as f:
        test = pickle.load(f, encoding='bytes')
    images = test[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(test[b'labels'])
    return images, labels

# ============================================================
# Benchmark runner
# ============================================================
def run_benchmark(model_path, dataset, batch_size=1, num_runs=5, 
                  warmup_batches=10, idle_seconds=10, stabilize_seconds=10,
                  results_dir='/home/ubuntu/results'):
    
    from pynq_dpu import DpuOverlay
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {dataset} dataset...")
    if dataset == 'mnist':
        images, labels = load_mnist()
    elif dataset == 'cifar10':
        images, labels = load_cifar10()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"  {len(images)} images, shape {images[0].shape}")
    
    # Load model
    print(f"Loading model: {model_path}")
    overlay = DpuOverlay('dpu.bit')
    overlay.load_model(model_path)
    dpu = overlay.runner
    
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_shape = tuple(inputTensors[0].dims)
    output_shape = tuple(outputTensors[0].dims)
    print(f"  DPU input: {input_shape}, output: {output_shape}")
    
    # Determine if batched (input doesn't match image shape)
    single_image_size = int(np.prod(images[0].shape))
    dpu_input_size = int(np.prod(input_shape[1:]))
    is_batched = dpu_input_size > single_image_size
    
    if is_batched:
        actual_batch = batch_size
        num_batches = len(images) // actual_batch
        print(f"  Batched mode: {actual_batch} images per DPU call, {num_batches} batches")
    else:
        actual_batch = 1
        num_batches = len(images)
        print(f"  Single image mode: {num_batches} images")
    
    # Config dict
    config = {
        'model_path': model_path,
        'dataset': dataset,
        'batch_size': actual_batch,
        'num_runs': num_runs,
        'num_images': len(images),
        'image_shape': list(images[0].shape),
        'dpu_input_shape': list(input_shape),
        'dpu_output_shape': list(output_shape),
        'timestamp': datetime.now().isoformat(),
        'board': 'KV260',
        'dpu': 'DPUCZDX8G_ISA1_B4096'
    }
    
    # Thermal stabilization
    print(f"Thermal stabilization ({stabilize_seconds}s)...")
    time.sleep(stabilize_seconds)
    
    # Idle power
    print(f"Measuring idle power ({idle_seconds}s)...")
    idle_samples = []
    for i in range(int(idle_seconds / 0.02)):
        idle_samples.append(read_power())
        time.sleep(0.02)
    idle_power = float(np.mean(idle_samples))
    idle_std = float(np.std(idle_samples))
    print(f"  Idle: {idle_power:.3f} +/- {idle_std:.3f} W")
    
    # Warmup
    print(f"Warmup ({warmup_batches} batches)...")
    for b in range(warmup_batches):
        if is_batched:
            batch_imgs = images[b*actual_batch:(b+1)*actual_batch]
            flat = batch_imgs.flatten()
            dpu_input = flat.reshape(input_shape)
            input_data = [np.ascontiguousarray(dpu_input, dtype=np.float32)]
        else:
            input_data = [np.empty(input_shape, dtype=np.float32, order='C')]
            input_data[0][0] = images[b]
        output_data = [np.empty(output_shape, dtype=np.float32, order='C')]
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
    
    # Measured runs
    print(f"Running {num_runs} measured runs...")
    all_runs = []
    
    for run in range(num_runs):
        power_log = []
        sampling = True
        
        def sampler():
            while sampling:
                power_log.append((time.time(), read_power()))
                time.sleep(0.01)
        
        thread = threading.Thread(target=sampler)
        correct = 0
        total = 0
        
        thread.start()
        start_time = time.time()
        
        if is_batched:
            for b in range(num_batches):
                batch_imgs = images[b*actual_batch:(b+1)*actual_batch]
                batch_labels = labels[b*actual_batch:(b+1)*actual_batch]
                flat = batch_imgs.flatten()
                dpu_input = flat.reshape(input_shape)
                input_data = [np.ascontiguousarray(dpu_input, dtype=np.float32)]
                output_data = [np.empty(output_shape, dtype=np.float32, order='C')]
                job_id = dpu.execute_async(input_data, output_data)
                dpu.wait(job_id)
                predictions = output_data[0].reshape(actual_batch, -1)
                preds = np.argmax(predictions, axis=1)
                correct += int(np.sum(preds == batch_labels))
                total += actual_batch
        else:
            for i in range(len(images)):
                input_data = [np.empty(input_shape, dtype=np.float32, order='C')]
                output_data = [np.empty(output_shape, dtype=np.float32, order='C')]
                input_data[0][0] = images[i]
                job_id = dpu.execute_async(input_data, output_data)
                dpu.wait(job_id)
                pred = int(np.argmax(output_data[0][0]))
                if pred == labels[i]:
                    correct += 1
                total += 1
        
        elapsed = time.time() - start_time
        sampling = False
        thread.join()
        
        avg_power = float(np.mean([s[1] for s in power_log]))
        
        run_result = {
            'run': run + 1,
            'accuracy': 100*correct/total,
            'time_s': elapsed,
            'throughput_fps': total/elapsed,
            'latency_ms': 1000*elapsed/total,
            'avg_power_w': avg_power,
            'energy_total_j': avg_power * elapsed,
            'energy_per_image_mj': 1000 * (avg_power * elapsed) / total,
            'power_samples': len(power_log)
        }
        all_runs.append(run_result)
        print(f"  Run {run+1}: {run_result['throughput_fps']:.1f} FPS, "
              f"{run_result['avg_power_w']:.3f} W, "
              f"{run_result['energy_per_image_mj']:.4f} mJ/img")
    
    # Summary
    summary = {
        'accuracy': float(np.mean([r['accuracy'] for r in all_runs])),
        'throughput_fps_mean': float(np.mean([r['throughput_fps'] for r in all_runs])),
        'throughput_fps_std': float(np.std([r['throughput_fps'] for r in all_runs])),
        'latency_ms_mean': float(np.mean([r['latency_ms'] for r in all_runs])),
        'latency_ms_std': float(np.std([r['latency_ms'] for r in all_runs])),
        'idle_power_w': idle_power,
        'idle_power_std': idle_std,
        'avg_power_w_mean': float(np.mean([r['avg_power_w'] for r in all_runs])),
        'avg_power_w_std': float(np.std([r['avg_power_w'] for r in all_runs])),
        'dynamic_power_w': float(np.mean([r['avg_power_w'] for r in all_runs])) - idle_power,
        'energy_per_image_mj_mean': float(np.mean([r['energy_per_image_mj'] for r in all_runs])),
        'energy_per_image_mj_std': float(np.std([r['energy_per_image_mj'] for r in all_runs])),
    }
    
    # Save everything
    output = {
        'config': config,
        'idle_power': {'mean': idle_power, 'std': idle_std, 'n_samples': len(idle_samples)},
        'runs': all_runs,
        'summary': summary
    }
    
    run_name = args.name or os.path.basename(model_path).replace('.xmodel', '')
    filename = f"{run_name}_{dataset}_b{actual_batch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Model:       {run_name}")
    print(f"  Dataset:     {dataset}")
    print(f"  Batch size:  {actual_batch}")
    print(f"  Accuracy:    {summary['accuracy']:.2f}%")
    print(f"  Throughput:  {summary['throughput_fps_mean']:.1f} +/- {summary['throughput_fps_std']:.1f} FPS")
    print(f"  Latency:     {summary['latency_ms_mean']:.3f} +/- {summary['latency_ms_std']:.3f} ms")
    print(f"  Idle power:  {idle_power:.3f} +/- {idle_std:.3f} W")
    print(f"  Avg power:   {summary['avg_power_w_mean']:.3f} +/- {summary['avg_power_w_std']:.3f} W")
    print(f"  Dynamic pwr: {summary['dynamic_power_w']:.3f} W")
    print(f"  Energy/img:  {summary['energy_per_image_mj_mean']:.4f} +/- {summary['energy_per_image_mj_std']:.4f} mJ")
    print(f"  Saved to:    {filepath}")
    print(f"{'='*60}")
    
    return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='Run name (e.g. mlp-tiny, cnn-tiny')
    parser.add_argument('--model', required=True, help='Path to xmodel')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--stabilize', type=int, default=10)
    parser.add_argument('--idle', type=int, default=10)
    args = parser.parse_args()
    
    run_benchmark(
        model_path=args.model,
        dataset=args.dataset,
        batch_size=args.batch,
        num_runs=args.runs,
        stabilize_seconds=args.stabilize,
        idle_seconds=args.idle
    )
