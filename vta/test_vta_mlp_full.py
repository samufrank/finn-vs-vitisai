"""Full MLP inference on VTA with proper ALU shift+clip.

VTA GEMM accumulates int8*int8 into int32. Before truncating to int8,
the accumulator must be right-shifted to bring values into [-128, 127].
Then clip ensures the range. This is what Relay does automatically via
ALU instructions; here we do it manually in the TE schedule.

Pipeline per GEMM: LOAD -> GEMM -> ALU(shift) -> ALU(clip_max) -> ALU(clip_min) -> STORE

Usage:
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 test_vta_mlp_full.py
"""
import numpy as np
import tvm
from tvm import te, rpc
from tvm.contrib import utils
import vta
import time
import sys
import math

BOARD_IP = "192.168.3.1"
RPC_PORT = 9091


def build_gemm_with_shift(remote, env, o, n, m, shift_bits):
    """Build VTA GEMM + ALU shift + clip module.

    Compute graph:
        A_buf, B_buf (LOAD from DRAM to SRAM)
        -> C_buf (GEMM: int8*int8 -> int32 in acc_scope)
        -> C_shr (ALU: right-shift int32 by shift_bits)
        -> C_clip_hi (ALU: min(x, 127))
        -> C_clip_lo (ALU: max(x, -128))
        -> C (cast int32 -> int8, STORE to DRAM)
    """
    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)

    # Load buffers (DRAM -> SRAM)
    A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
    B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf")

    # GEMM (int8*int8 -> int32 accumulator)
    ko = te.reduce_axis((0, n), name="ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
    C_buf = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda bo, co, bi, ci: te.sum(
            A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
            B_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki]),
        name="C_buf")

    # ALU: right-shift (brings int32 accumulator into int8 range)
    shr_const = tvm.tir.const(shift_bits, env.acc_dtype)
    C_shr = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_buf(*i) >> shr_const,
        name="C_shr")

    # ALU: clip max -- min(x, 127)
    clip_max = tvm.tir.const(127, env.acc_dtype)
    C_clip_hi = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.min(C_shr(*i), clip_max),
        name="C_clip_hi")

    # ALU: clip min -- max(x, -128)
    clip_min_val = tvm.tir.const(-128, env.acc_dtype)
    C_clip_lo = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.max(C_clip_hi(*i), clip_min_val),
        name="C_clip_lo")

    # Cast int32 -> int8 and store to DRAM
    C = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_clip_lo(*i).astype(env.inp_dtype),
        name="C")

    # ---- Schedule ----
    s = te.create_schedule(C.op)

    # Memory scopes
    s[A_buf].set_scope(env.inp_scope)
    s[B_buf].set_scope(env.wgt_scope)
    s[C_buf].set_scope(env.acc_scope)
    s[C_shr].set_scope(env.acc_scope)
    s[C_clip_hi].set_scope(env.acc_scope)
    s[C_clip_lo].set_scope(env.acc_scope)

    # GEMM schedule (same proven pattern as test_vta_verify.py)
    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
        s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    # ALU schedule -- each op runs on VTA's vector ALU
    s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
    s[C_clip_hi].pragma(s[C_clip_hi].op.axis[0], env.alu)
    s[C_clip_lo].pragma(s[C_clip_lo].op.axis[0], env.alu)

    # Store schedule
    s[C].pragma(s[C].op.axis[0], env.dma_copy)

    # Build
    mod = vta.build(s, [A, B, C],
                    tvm.target.vta(),
                    tvm.target.arm_cpu("ultra96"),
                    name="my_gemm")

    fname = f"gemm_shr_o{o}_n{n}_m{m}_s{shift_bits}.o"
    temp = utils.tempdir()
    mod.save(temp.relpath(fname))
    remote.upload(temp.relpath(fname))
    return remote.load_module(fname)


def tile_input(x_int8, env):
    """(in_features,) -> (1, n, 1, BLOCK_IN)"""
    n = len(x_int8) // env.BLOCK_IN
    return x_int8.reshape(1, n, 1, env.BLOCK_IN)


def tile_weights(W_int8, env):
    """(out, in) -> (m, n, BLOCK_OUT, BLOCK_IN)"""
    out_f, in_f = W_int8.shape
    m = out_f // env.BLOCK_OUT
    n = in_f // env.BLOCK_IN
    return W_int8.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)


def untile_output(C_tiled, out_features):
    """(1, m, 1, BLOCK_OUT) -> (out_features,)"""
    return C_tiled.reshape(-1)[:out_features]


def compute_shift_bits(x_int8, W_int8):
    """Determine right-shift amount so max(|accumulator >> shift|) <= 127."""
    acc = x_int8.astype(np.int32) @ W_int8.T.astype(np.int32)
    max_abs = np.max(np.abs(acc))
    if max_abs <= 127:
        return 0
    return int(math.ceil(math.log2(max_abs / 127.0)))


def reference_fc_shifted(x_int8, W_int8, shift_bits):
    """CPU reference matching VTA's GEMM + shift + clip."""
    acc = x_int8.astype(np.int32) @ W_int8.T.astype(np.int32)
    shifted = acc >> shift_bits  # arithmetic right-shift
    clipped = np.clip(shifted, -128, 127)
    return clipped.astype(np.int8), acc


def main():
    env = vta.get_env()
    print(f"VTA env: TARGET={env.TARGET}")
    print(f"  BATCH={env.BATCH}, BLOCK_IN={env.BLOCK_IN}, BLOCK_OUT={env.BLOCK_OUT}")

    # ---- MLP Architecture ----
    layer_dims = [784, 64, 32, 16]
    print(f"\nMLP architecture: {' -> '.join(str(d) for d in layer_dims)}")

    # ---- Create random weights ----
    np.random.seed(42)
    layers = []
    for i in range(len(layer_dims) - 1):
        in_f, out_f = layer_dims[i], layer_dims[i + 1]
        std = np.sqrt(2.0 / (in_f + out_f))
        W_float = np.random.randn(out_f, in_f).astype(np.float32) * std
        b_float = np.zeros(out_f, dtype=np.float32)
        w_scale = np.max(np.abs(W_float)) / 127.0
        W_int8 = np.clip(np.round(W_float / w_scale), -128, 127).astype(np.int8)
        layers.append({
            'W_float': W_float, 'b_float': b_float,
            'W_int8': W_int8, 'w_scale': w_scale,
            'in_f': in_f, 'out_f': out_f,
        })
        print(f"  Layer {i+1}: {in_f}->{out_f}, w_scale={w_scale:.6f}")

    # ---- CPU float32 reference ----
    x_float = np.random.randn(784).astype(np.float32) * 0.5
    h_float = x_float.copy()
    for i, layer in enumerate(layers):
        h_float = h_float @ layer['W_float'].T + layer['b_float']
        if i < len(layers) - 1:
            h_float = np.maximum(h_float, 0)
    y_cpu_ref = h_float
    print(f"\nCPU float32 reference: {y_cpu_ref}")

    # ---- Quantize input ----
    x_scale = np.max(np.abs(x_float)) / 127.0
    x_int8 = np.clip(np.round(x_float / x_scale), -128, 127).astype(np.int8)

    # ---- Calibrate shift amounts ----
    print(f"\nCalibrating shift amounts...")
    h_cal = x_int8.copy()
    shift_amounts = []
    cal_scales = [x_scale]
    for i, layer in enumerate(layers):
        shift = compute_shift_bits(h_cal, layer['W_int8'])
        shift_amounts.append(shift)
        ref_out, acc = reference_fc_shifted(h_cal, layer['W_int8'], shift)
        print(f"  Layer {i+1}: acc range [{acc.min()}, {acc.max()}], "
              f"shift={shift}, shifted range [{(acc >> shift).min()}, {(acc >> shift).max()}]")

        # Dequantize for next layer
        combined = cal_scales[-1] * layer['w_scale'] * (2 ** shift)
        y_f = ref_out.astype(np.float32) * combined + layer['b_float']
        if i < len(layers) - 1:
            y_f = np.maximum(y_f, 0)
            next_scale = np.max(np.abs(y_f)) / 127.0 if np.max(np.abs(y_f)) > 0 else 1e-10
            h_cal = np.clip(np.round(y_f / next_scale), -128, 127).astype(np.int8)
            cal_scales.append(next_scale)

    # ---- Connect to board ----
    print(f"\nConnecting to {BOARD_IP}:{RPC_PORT}...")
    remote = rpc.connect(BOARD_IP, RPC_PORT)
    vta.reconfig_runtime(remote)
    try:
        vta.program_fpga(remote, bitstream=None)
    except Exception:
        pass
    ctx = remote.ext_dev(0)

    # ---- Build GEMM+shift modules ----
    gemm_modules = []
    for i, layer in enumerate(layers):
        n_tiles = layer['in_f'] // env.BLOCK_IN
        m_tiles = layer['out_f'] // env.BLOCK_OUT
        shift = shift_amounts[i]
        print(f"  Building layer {i+1} (o=1, n={n_tiles}, m={m_tiles}, shift={shift})...",
              end=" ", flush=True)
        f = build_gemm_with_shift(remote, env, 1, n_tiles, m_tiles, shift)
        gemm_modules.append(f)
        print("OK")

    # ---- Per-layer verification ----
    print(f"\n{'='*60}")
    print("Per-layer GEMM+shift+clip verification")
    print(f"{'='*60}")

    h_verify = x_int8.copy()
    verify_scales = [x_scale]
    all_correct = True
    y_vta_final = None

    for i, (layer, f, shift) in enumerate(zip(layers, gemm_modules, shift_amounts)):
        ref_int8, acc = reference_fc_shifted(h_verify, layer['W_int8'], shift)

        # VTA execution
        x_tiled = tile_input(h_verify, env)
        W_tiled = tile_weights(layer['W_int8'], env)
        m_tiles = layer['out_f'] // env.BLOCK_OUT

        A_nd = tvm.nd.array(x_tiled, ctx)
        B_nd = tvm.nd.array(W_tiled, ctx)
        C_nd = tvm.nd.array(
            np.zeros((1, m_tiles, env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)
        f(A_nd, B_nd, C_nd)
        vta_int8 = untile_output(C_nd.numpy(), layer['out_f'])

        match = np.array_equal(ref_int8, vta_int8)
        all_correct &= match
        print(f"  Layer {i+1} ({layer['in_f']}->{layer['out_f']}, shift={shift}): "
              f"{'PASS' if match else 'FAIL'}")
        if not match:
            mismatches = np.sum(ref_int8 != vta_int8)
            print(f"    {mismatches}/{layer['out_f']} mismatches")
            print(f"    ref[:16]: {ref_int8[:16]}")
            print(f"    vta[:16]: {vta_int8[:16]}")
        else:
            print(f"    output[:8]: {vta_int8[:8]}")

        # Prepare next layer
        combined = verify_scales[-1] * layer['w_scale'] * (2 ** shift)
        y_f = vta_int8.astype(np.float32) * combined + layer['b_float']
        if i < len(layers) - 1:
            y_f = np.maximum(y_f, 0)
            next_scale = np.max(np.abs(y_f)) / 127.0 if np.max(np.abs(y_f)) > 0 else 1e-10
            h_verify = np.clip(np.round(y_f / next_scale), -128, 127).astype(np.int8)
            verify_scales.append(next_scale)
        else:
            y_vta_final = y_f

    # ---- End-to-end comparison ----
    print(f"\n{'='*60}")
    print("End-to-end inference comparison")
    print(f"{'='*60}")
    print(f"  VTA output:      {y_vta_final}")
    print(f"  CPU float32 ref: {y_cpu_ref}")

    cos_sim = np.dot(y_vta_final, y_cpu_ref) / (
        np.linalg.norm(y_vta_final) * np.linalg.norm(y_cpu_ref) + 1e-8)
    argmax_vta = np.argmax(y_vta_final)
    argmax_cpu = np.argmax(y_cpu_ref)
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Argmax: VTA={argmax_vta}, CPU={argmax_cpu}, match={argmax_vta == argmax_cpu}")

    # ---- Timing ----
    print(f"\n{'='*60}")
    print("End-to-end MLP timing")
    print(f"{'='*60}")

    W_tileds = [tile_weights(l['W_int8'], env) for l in layers]
    num_runs = 20
    times = []
    for _ in range(num_runs):
        h = x_int8.copy()
        sc = x_scale
        t0 = time.time()
        for i, (layer, f, shift) in enumerate(zip(layers, gemm_modules, shift_amounts)):
            x_t = tile_input(h, env)
            m_t = layer['out_f'] // env.BLOCK_OUT
            A_nd = tvm.nd.array(x_t, ctx)
            B_nd = tvm.nd.array(W_tileds[i], ctx)
            C_nd = tvm.nd.array(
                np.zeros((1, m_t, env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)
            f(A_nd, B_nd, C_nd)
            out = untile_output(C_nd.numpy(), layer['out_f'])
            if i < len(layers) - 1:
                combined = sc * layer['w_scale'] * (2 ** shift)
                y_f = out.astype(np.float32) * combined + layer['b_float']
                y_f = np.maximum(y_f, 0)
                sc = np.max(np.abs(y_f)) / 127.0 if np.max(np.abs(y_f)) > 0 else 1e-10
                h = np.clip(np.round(y_f / sc), -128, 127).astype(np.int8)
        t1 = time.time()
        times.append(t1 - t0)

    avg_ms = np.mean(times) * 1000
    n_layers = len(layers)
    sleep_ms = n_layers * 200
    print(f"  Architecture: {' -> '.join(str(d) for d in layer_dims)}")
    print(f"  {num_runs} runs, avg = {avg_ms:.1f}ms/inference")
    print(f"  Sleep overhead: {sleep_ms}ms ({n_layers} layers x 200ms)")
    print(f"  Estimated actual: {avg_ms - sleep_ms:.1f}ms/inference")

    # ---- Summary ----
    print(f"\n{'='*60}")
    status = "ALL PASS" if all_correct else "SOME FAILED"
    print(f"GEMM+shift+clip correctness: {status}")
    print(f"Cosine similarity: {cos_sim:.4f}")
    print(f"{'='*60}")

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
