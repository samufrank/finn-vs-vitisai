#!/usr/bin/env python3
"""VTA INT4-o8 Transformer board-side inference.

Runs the RadioML 2018 INT4 transformer (Phase-3 Mode-E-tuned config)
on an AUP-ZU3 (ZU3EG) board. Loads pre-compiled VTA modules from a
deployment directory (e.g. /home/xilinx/transformer_export/) produced
by finn-vs-vitisai/vta/export_vta_transformer.py.

Pipeline per inference (matches vta_transformer_sim_o8.py mode='E' bit-exactly).
All VTA modules compile with m_tiles=1 to work around a TVM multi-tile INT4-o8
codegen bug; the runtime concatenates int8 output stripes from N_full/16 calls.

   CPU (INT8): patch_embedding -> positional_encoding
   CPU:        BN_attn + INT4 quant
   VTA:        Q proj (proj_k96_s3_m1, 6 calls), K proj (same, 6 calls),
               V proj (proj_k96_s2_m1, 2 calls)
   CPU:        int8 -> INT4 requant for Q/K/V; reshape to 3 heads
   for h in 0..2:
       VTA:    Q_h @ K_h.T  (qkt_s3_m1, 4 calls)
       CPU:    int8 -> INT4 requant (folds 1/sqrt(96)); float softmax;
               INT4 quant of attn weights
       VTA:    attn_h @ V_h (av_s3_m1, 2 calls)
       CPU:    int8 -> INT4 requant
   CPU:        concat heads (64, 96)
   VTA:        O proj (proj_k96_s3_m1, 6 calls)
   CPU:        INT4 requant; residual add with positional skip
   CPU:        BN_mlp + INT4 quant
   VTA:        fc1 (fc1_s3_m1, 4-arg with bias, 24 calls)
   CPU:        unsigned-INT4 requant ([0,15]); zero-point shift -8 to signed
   VTA:        fc2 (fc2_s4_m1, 4-arg with corrected bias, 6 calls)
   CPU:        INT4 requant; residual add with attention skip
   CPU:        GAP -> classifier matmul -> argmax

Modes:
    --validate                    : full eval set, accuracy + match-vs-ref
    --benchmark N [--warmup K]    : N inferences with per-section timing

Usage on the board:
    sudo LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH \\
         PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH \\
         python3 benchmark_vta_transformer.py \\
             --export-dir /home/xilinx/transformer_export \\
             --validate \\
             --reference-preds /home/xilinx/transformer_export/phase3_mode_e_optimal_preds.npy
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time

import numpy as np
import tvm
import tvm.runtime


SQRT_96 = 96 ** -0.5
BLOCK = 16
BITSTREAM = "1x16_i4w4o8a32_15_14_17_17.bit"


# ============================================================
# INT4 nibble packing — same shape as input, packed data in first half of
# the FLAT byte buffer; second half zero-padded. Matches benchmark.py's
# `pack_int4_for_vta` (the working CNN convention). This shape MUST match
# the compiled module's int4 placeholder (64, 6, 1, 16) etc., or TVM
# silently rejects the call and the module produces zeros.
# ============================================================
def pack_int4_for_vta(vals_int8):
    """Pack int8 array of int4 values into VTA nibble format.

    First half of flat buffer holds packed nibble pairs (lo=even, hi=odd);
    second half is zero-padded. Same shape preserved.
    """
    vals = np.asarray(vals_int8, dtype=np.int8)
    flat = vals.flatten()
    n = len(flat)
    lo = flat[0::2].view(np.uint8) & 0xF
    hi = flat[1::2].view(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[: n // 2] = packed
    return out.reshape(vals.shape)


# NOTE: tried `tvm.nd.empty(shape, dtype="int4", device=ext_dev)` + ctypes.memmove
# as a workaround. On VTA ext_dev(0) this produced corrupted NDArray metadata
# (garbage shape values), so it's unsafe — never use that path. The CNN-style
# `tvm.nd.array(packed_int8, ctx)` is the only Python path that gives a sane
# NDArray on VTA, even though it allocates 2x the int4 byte size.


# ============================================================
# Board setup: clear stale PYNQ state, load bitstream, load libvta.
# ============================================================
def setup_board(bitstream_paths_extra=None, require_int4_o8: bool = True):
    """Load INT4-o8 bitstream + libvta. If require_int4_o8 is True, hard-error
    when the live VTA env doesn't match (INP=4, WGT=4, OUT=8) since wrong-build
    bitstreams produce silent all-zero VTA output.
    """
    stale = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale):
            os.remove(stale)
    except Exception:
        pass

    candidates = [
        f'/root/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
        f'/home/xilinx/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
        f'/home/xilinx/{BITSTREAM}',
    ]
    if bitstream_paths_extra:
        candidates.extend(bitstream_paths_extra)
    bit = None
    for p in candidates:
        if os.path.exists(p):
            bit = p
            break
    if bit:
        print(f"[bitstream] {bit}")
        try:
            from pynq import Overlay
            Overlay(bit)
        except Exception as e:
            print(f"  Overlay: {e} (continuing — bitstream may already be loaded)")
    else:
        print(f"WARNING: bitstream {BITSTREAM} not found at standard cache paths.")
        print(f"  Searched: {candidates}")
        print(f"  If vta.bit is already loaded, ensure it IS the {BITSTREAM} build —")
        print(f"  a wrong-build bitstream will silently give all-zero VTA output.")

    for p in [
        '/home/xilinx/tvm-src/build/libvta.so',
        os.path.join(os.environ.get('TVM_HOME', ''), 'build/libvta.so'),
    ]:
        if p and os.path.exists(p):
            print(f"[libvta] {p}")
            ctypes.CDLL(p, ctypes.RTLD_GLOBAL)
            break
    else:
        print("WARNING: libvta.so not found at standard paths")

    # ---- Hard env check ----
    import vta as _vta
    env = _vta.get_env()
    print(f"[vta env] TARGET={env.TARGET}  INP={env.INP_WIDTH}  WGT={env.WGT_WIDTH}  "
          f"OUT={env.OUT_WIDTH}  ACC={env.ACC_WIDTH}  BATCH={env.BATCH}  "
          f"BLOCK={env.BLOCK_IN}/{env.BLOCK_OUT}")
    if require_int4_o8:
        bad = []
        if env.INP_WIDTH  != 4:  bad.append(f"INP_WIDTH={env.INP_WIDTH} (need 4)")
        if env.WGT_WIDTH  != 4:  bad.append(f"WGT_WIDTH={env.WGT_WIDTH} (need 4)")
        if env.OUT_WIDTH  != 8:  bad.append(f"OUT_WIDTH={env.OUT_WIDTH} (need 8)")
        if env.BATCH      != 1:  bad.append(f"BATCH={env.BATCH} (need 1)")
        if env.BLOCK_IN   != 16: bad.append(f"BLOCK_IN={env.BLOCK_IN} (need 16)")
        if env.BLOCK_OUT  != 16: bad.append(f"BLOCK_OUT={env.BLOCK_OUT} (need 16)")
        if bad:
            raise RuntimeError(
                "VTA env does not match the INT4-o8 modules compiled by "
                "export_vta_transformer.py:\n  " + "\n  ".join(bad) +
                "\n\nFix: load the 1x16_i4w4o8a32_15_14_17_17.bit bitstream and "
                "ensure the board's vta_config.json matches "
                "(LOG_INP_WIDTH=2, LOG_WGT_WIDTH=2, LOG_OUT_WIDTH=3, "
                "LOG_ACC_WIDTH=5, LOG_BLOCK=4, LOG_BATCH=0)."
            )
    return env


# ============================================================
# RadioML 2018 SNR-filtered eval loader.
# Data layout in npz: signals (N, 1, 1024, 2) float32, labels (N,) int.
# ============================================================
def load_radioml(data_path):
    d = np.load(data_path)
    return d['signals'], d['labels']


# ============================================================
# Tiling helpers for runtime activations used as VTA "weights".
# Static weights are tiled at export time; these are for K/V in attention.
# ============================================================
def tile_for_wgt(W, m_tiles, n_tiles):
    """Tile a numpy array for the VTA weight buffer.
    Input W shape: (m_tiles*BLOCK_OUT, n_tiles*BLOCK_IN).
    Output shape: (m_tiles, n_tiles, BLOCK_OUT, BLOCK_IN).
    """
    return W.reshape(m_tiles, BLOCK, n_tiles, BLOCK).transpose(0, 2, 1, 3)


# ============================================================
# CPU operations between VTA calls — copied bit-for-bit from
# vta_transformer_sim_o8.py to guarantee numerical match.
# ============================================================
def quant_signed(x_float, scale, lo, hi):
    return np.clip(np.round(x_float / scale), lo, hi).astype(np.int32)


def quant_unsigned(x_float, scale, lo, hi):
    return np.clip(np.round(x_float / scale), lo, hi).astype(np.int32)


def cpu_requant_after_vta(x_int8, coarse_shift, in_scale, w_scale, out_scale,
                           clip_lo, clip_hi, extra_factor=1.0):
    """Mirror of TransformerSimO8._requant_E's CPU portion.

    VTA already shifted the int32 acc right by `coarse_shift` and clipped to
    int8 [-128, 127] before DMA. CPU recovers the float-domain value:
      real = x_int8 * 2^coarse_shift * w_scale * in_scale * extra_factor
    Then requantizes to int4 with `out_scale`.
    """
    recovered = x_int8.astype(np.float64) * (2.0 ** coarse_shift)
    real = recovered * (w_scale * in_scale * extra_factor)
    return np.clip(np.round(real / out_scale), clip_lo, clip_hi).astype(np.int8)


# ============================================================
# Main inference class.
# ============================================================
class TransformerVTA:
    def __init__(self, export_dir):
        import tvm
        import tvm.runtime
        self.tvm = tvm
        self.export_dir = export_dir
        self.ctx = tvm.device('ext_dev', 0)

        with open(os.path.join(export_dir, 'config.json')) as f:
            self.cfg = json.load(f)
        with open(os.path.join(export_dir, 'scales.json')) as f:
            self.S = json.load(f)
        self.tuned = self.cfg['tuned_shifts']
        self.modules_meta = self.cfg['modules']

        # ----- Load 6 compiled modules (.so or .o) -----
        self.mods = {}
        for mid in self.modules_meta:
            mod_rel = self.modules_meta[mid]['file']     # e.g. modules/proj_k96_s3_m1.o
            o_path = os.path.join(export_dir, mod_rel)
            so_path = o_path.replace('.o', '.so')
            if not os.path.exists(so_path):
                import subprocess
                print(f"  linking {os.path.basename(o_path)} -> .so")
                subprocess.check_call([
                    'gcc', '-shared', '-o', so_path, o_path,
                    '-L/home/xilinx/tvm-src/build', '-ltvm_runtime'])
            print(f"[load module] {mid} <- {os.path.basename(so_path)}")
            self.mods[mid] = tvm.runtime.load_module(so_path)

        # ----- Load + pre-pack static weights as m=1 SLICES -----
        # Disk weights are int8-tiled full-shape (m_full, n, BLOCK, BLOCK).
        # Each m-slice (1, n, BLOCK, BLOCK) is packed once at init via
        # pack_int4_for_vta. The runtime feeds slices to the m=1 VTA module
        # one per call, then concatenates the int8 outputs.
        self.W_slices_packed = {}   # gemm name -> list of (1, n, BLOCK, BLOCK) packed int8
        for gemm_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'fc1', 'fc2']:
            W_full = np.load(os.path.join(export_dir,
                                           f'weights/{gemm_name}_W_tiled.npy')).astype(np.int8)
            slices = []
            for m_call in range(W_full.shape[0]):
                slc = np.ascontiguousarray(W_full[m_call:m_call+1])
                slices.append(pack_int4_for_vta(slc))
            self.W_slices_packed[gemm_name] = slices

        # ----- Load full real biases for fc1/fc2 (sliced at runtime) -----
        # Stored shape (m_full, BLOCK) int32. Per-call slice [m_call:m_call+1]
        # is broadcast to (o_tile, 1, 1, BLOCK) before each VTA call.
        self.bias_full = {}
        for fc_name, fc_file in [('fc1', 'fc1_bias_int32.npy'),
                                 ('fc2', 'fc2_bias_int32_corrected.npy')]:
            self.bias_full[fc_name] = np.load(
                os.path.join(export_dir, f'weights/{fc_file}')).astype(np.int32)

        # ----- Load CPU-side parameters -----
        cpu = lambda name: np.load(os.path.join(export_dir, 'cpu_params', name))
        self.W_emb       = cpu('emb_conv_W_int.npy').astype(np.int8)        # (96, 2, 1, 16)
        self.b_emb       = cpu('emb_conv_bias.npy').astype(np.float32)
        self.bn_emb_mean = cpu('bn_emb_mean.npy').astype(np.float32)
        self.bn_emb_var  = cpu('bn_emb_var.npy').astype(np.float32)
        self.pos_enc     = cpu('pos_enc.npy').astype(np.float32)            # (1, 64, 96)
        self.bn_attn_mean = cpu('bn_attn_mean.npy').astype(np.float32)
        self.bn_attn_var  = cpu('bn_attn_var.npy').astype(np.float32)
        self.bn_mlp_mean  = cpu('bn_mlp_mean.npy').astype(np.float32)
        self.bn_mlp_var   = cpu('bn_mlp_var.npy').astype(np.float32)
        self.W_cls        = cpu('cls_W_int.npy').astype(np.int8)            # (24, 96)
        self.b_cls        = cpu('cls_bias.npy').astype(np.float32)

        # Pre-compute classifier dequantized weights and patch-embed flat weights+biases
        self.W_emb_flat = self.W_emb.reshape(96, 32).astype(np.int32)
        S = self.S
        self.bias_emb_int32 = np.round(
            self.b_emb / (S['emb_w'] * S['emb_in'])).astype(np.int32)
        self.W_cls_float = self.W_cls.astype(np.float64) * S['cls_w']
        self.b_cls_f64   = self.b_cls.astype(np.float64)
        # Squeeze pos_enc to (64, 96) for direct adds
        self.pos2d = self.pos_enc[0]  # (64, 96)

    # ------------------------------------------------------------
    # m=1 chunked VTA call. All modules compile with m=1; runtime issues
    # one VTA call per m-slice and concatenates int8 output stripes.
    #
    # Per-call protocol on this PL (verified empirically):
    #   - Fresh tvm.nd.array A/W/D/C with .copy() on every numpy input
    #   - ctx.sync() BEFORE and AFTER mod() to keep the queue clean
    #   - Detect all-zero output and retry up to 3× with fresh buffers
    #     (intermittent VTA glitch); log each retry to stderr
    #   - DO NOT del + gc.collect after the call — that forces premature
    #     CMA release and destabilises the in-flight DMA state
    # ------------------------------------------------------------
    def _vta_gemm_m1(self, mod_id, A_packed, *,
                     gemm_name=None, runtime_W_slices=None, bias_gemm_name=None):
        """Issue n_calls m=1 VTA calls and concat outputs.

        A_packed:           pre-packed int4 input, shape (o_tile, n, 1, BLOCK).
        gemm_name:          static-weight gemm; uses self.W_slices_packed[gemm_name].
        runtime_W_slices:   list of pre-packed (1, n, BLOCK, BLOCK) numpy int8 slices
                            for runtime-weight GEMMs (qk/av per head).
        bias_gemm_name:     for fc1/fc2 — slice/broadcast bias_full[bias_gemm_name]
                            per call. Otherwise a zero D buffer is used.
        Returns: int8 (o_tile, n_calls * BLOCK).
        """
        info = self.modules_meta[mod_id]
        o_t  = info['o_tile']
        slices = self.W_slices_packed[gemm_name] if gemm_name else runtime_W_slices
        n_calls = len(slices)
        bias_full = self.bias_full[bias_gemm_name] if bias_gemm_name else None
        full_out = np.empty((o_t, n_calls * BLOCK), dtype=np.int8)
        zero_D = np.zeros((o_t, 1, 1, BLOCK), dtype=np.int32)
        out_shape = (o_t, 1, 1, BLOCK)
        for m_call in range(n_calls):
            if bias_full is not None:
                bias_slice = bias_full[m_call:m_call+1]   # (1, BLOCK)
                D_arr = np.ascontiguousarray(
                    np.broadcast_to(bias_slice.reshape(1, 1, 1, BLOCK), (o_t, 1, 1, BLOCK)),
                    dtype=np.int32)
            else:
                D_arr = zero_D
            out_slice = None
            for attempt in range(4):    # 1 initial + up to 3 retries
                A_nd = tvm.nd.array(A_packed.copy(),       self.ctx)
                W_nd = tvm.nd.array(slices[m_call].copy(), self.ctx)
                D_nd = tvm.nd.array(D_arr.copy(),          self.ctx)
                C_nd = tvm.nd.array(np.zeros(out_shape, dtype=np.int8), self.ctx)
                self.ctx.sync()
                self.mods[mod_id](A_nd, W_nd, D_nd, C_nd)
                self.ctx.sync()
                out_slice = C_nd.numpy()[:, 0, 0, :].copy()
                if (out_slice != 0).any():
                    break
                if attempt < 3:
                    print(f"[retry] {mod_id} m_call={m_call}: zero output, "
                          f"retry {attempt+1}/3", file=sys.stderr)
            full_out[:, m_call*BLOCK:(m_call+1)*BLOCK] = out_slice
        return full_out

    # ------------------------------------------------------------
    # Static-weight projection (Q/K/V/O): m=1 chunked VTA calls with
    # M-chunking when module o_tile < M_full. Concatenated to (M_full, N_full).
    # ------------------------------------------------------------
    def _proj(self, mod_id, gemm_name, in_int4):
        """in_int4: (M_full, K) int8 (one int4 value per byte).
        Returns int8 (M_full, N_full)."""
        info   = self.modules_meta[mod_id]
        o_tile = info['o_tile']
        n_t    = info['n']
        M_full = in_int4.shape[0]
        n_m_chunks = M_full // o_tile
        if n_m_chunks == 1:
            A_tiled = in_int4.reshape(o_tile, n_t, 1, BLOCK)
            A_packed = pack_int4_for_vta(A_tiled.astype(np.int8))
            return self._vta_gemm_m1(mod_id, A_packed, gemm_name=gemm_name)
        chunks = []
        for m_chunk in range(n_m_chunks):
            chunk_in = in_int4[m_chunk*o_tile:(m_chunk+1)*o_tile]
            A_tiled  = chunk_in.reshape(o_tile, n_t, 1, BLOCK)
            A_packed = pack_int4_for_vta(np.ascontiguousarray(A_tiled.astype(np.int8)))
            chunks.append(self._vta_gemm_m1(mod_id, A_packed, gemm_name=gemm_name))
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------
    # Runtime-weight GEMM (qk / av): A is the activation, W slices come
    # from K_h or V_h tiled at runtime. Same M-chunking pattern as _proj.
    # ------------------------------------------------------------
    def _runtime_w_gemm(self, mod_id, A_unpacked, W_slices):
        """A_unpacked: (M_full, K) int8 activation. W_slices: list of pre-packed
        (1, n, BLOCK, BLOCK) int4 slices for the runtime weight.
        Returns int8 (M_full, len(W_slices) * BLOCK)."""
        info   = self.modules_meta[mod_id]
        o_tile = info['o_tile']
        n_t    = info['n']
        M_full = A_unpacked.shape[0]
        n_m_chunks = M_full // o_tile
        if n_m_chunks == 1:
            A_tiled = A_unpacked.reshape(o_tile, n_t, 1, BLOCK)
            A_packed = pack_int4_for_vta(A_tiled.astype(np.int8))
            return self._vta_gemm_m1(mod_id, A_packed, runtime_W_slices=W_slices)
        chunks = []
        for m_chunk in range(n_m_chunks):
            chunk_in = A_unpacked[m_chunk*o_tile:(m_chunk+1)*o_tile]
            A_tiled  = chunk_in.reshape(o_tile, n_t, 1, BLOCK)
            A_packed = pack_int4_for_vta(np.ascontiguousarray(A_tiled.astype(np.int8)))
            chunks.append(self._vta_gemm_m1(mod_id, A_packed, runtime_W_slices=W_slices))
        return np.concatenate(chunks, axis=0)

    def _mlp(self, mod_id, gemm_name, in_int4):
        """fc1 / fc2: 4-arg path with M-chunking when module o_tile < M_full.

        in_int4 has shape (M_full, K). With o_tile<M_full (fc1/fc2 use o_tile=8
        because the larger o×n product triggered a hardware all-zero condition),
        we issue n_m_chunks = M_full / o_tile invocations of the m=1 chunked
        path and concatenate outputs on the M (row) axis.
        """
        info = self.modules_meta[mod_id]
        o_tile = info['o_tile']
        n_t    = info['n']
        M_full = in_int4.shape[0]
        n_m_chunks = M_full // o_tile
        if n_m_chunks == 1:
            A_tiled = in_int4.reshape(o_tile, n_t, 1, BLOCK)
            A_packed = pack_int4_for_vta(A_tiled.astype(np.int8))
            return self._vta_gemm_m1(mod_id, A_packed,
                                      gemm_name=gemm_name, bias_gemm_name=gemm_name)
        chunks = []
        for m_chunk in range(n_m_chunks):
            chunk_in = in_int4[m_chunk*o_tile:(m_chunk+1)*o_tile]
            A_tiled  = chunk_in.reshape(o_tile, n_t, 1, BLOCK)
            A_packed = pack_int4_for_vta(np.ascontiguousarray(A_tiled.astype(np.int8)))
            chunks.append(self._vta_gemm_m1(mod_id, A_packed,
                                             gemm_name=gemm_name,
                                             bias_gemm_name=gemm_name))
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------
    # Full inference
    # ------------------------------------------------------------
    def infer(self, signal, timings=None):
        """signal: float32 array shape (1, 1024, 2). Returns int prediction.
        If timings is a dict, append per-section elapsed seconds.
        """
        S = self.S
        t0_total = time.perf_counter()
        rec = (timings is not None)

        # ============ 1. Patch embedding (CPU INT8) ============
        if rec: t0 = time.perf_counter()
        # signal (1, 1024, 2) -> (2, 1, 1024)
        x = signal.transpose(2, 0, 1).astype(np.float32)
        x_int8 = quant_signed(x, S['emb_in'], -128, 127).astype(np.int8)
        # im2col: (2, 1, 1024) -> 64 non-overlapping (2,16) patches -> (64, 32)
        x_patches = x_int8.reshape(2, 1, 64, 16).transpose(2, 0, 1, 3).reshape(64, 32)
        acc = x_patches.astype(np.int32) @ self.W_emb_flat.T + self.bias_emb_int32[None, :]
        emb_float = acc.astype(np.float64) * (S['emb_w'] * S['emb_in'])
        emb_bn = (emb_float - self.bn_emb_mean.astype(np.float64)[None, :]) / \
                  np.sqrt(self.bn_emb_var.astype(np.float64)[None, :] + 1e-5)
        emb_relu = np.maximum(emb_bn, 0.0)
        emb_q = quant_unsigned(emb_relu, S['emb_out'], 0, 255)  # (64, 96), values [0, 255]
        if rec: timings.setdefault('patch_embed', []).append(time.perf_counter() - t0)

        # ============ 2. Positional encoding (CPU INT8) ============
        if rec: t0 = time.perf_counter()
        emb_for_pos = emb_q.astype(np.float64) * S['emb_out']
        emb_q2 = quant_signed(emb_for_pos, S['pos_in'], -128, 127)
        pos_q  = quant_signed(self.pos2d, S['pos_in'], -128, 127)
        sum_float = (emb_q2.astype(np.float64) + pos_q.astype(np.float64)) * S['pos_in']
        after_pos = quant_signed(sum_float, S['pos_out'], -128, 127).astype(np.int8)  # (64, 96)
        if rec: timings.setdefault('positional', []).append(time.perf_counter() - t0)

        # ============ 3. Pre-attn BN + INT4 quant ============
        if rec: t0 = time.perf_counter()
        x_pos_float = after_pos.astype(np.float64) * S['pos_out']
        bn_attn = (x_pos_float - self.bn_attn_mean.astype(np.float64)[None, :]) / \
                   np.sqrt(self.bn_attn_var.astype(np.float64)[None, :] + 1e-5)
        pre_int4 = quant_signed(bn_attn, S['attn_pre_out'], -8, 7).astype(np.int8)  # (64, 96)
        if rec: timings.setdefault('bn_attn', []).append(time.perf_counter() - t0)

        # ============ 4. Q / K / V projections (3 VTA calls) ============
        if rec: t0 = time.perf_counter()
        Q_int8 = self._proj('proj_k96_s3_m1', 'q_proj', pre_int4)
        K_int8 = self._proj('proj_k96_s3_m1', 'k_proj', pre_int4)
        V_int8 = self._proj('proj_k96_s2_m1', 'v_proj', pre_int4)
        if rec: timings.setdefault('vta_qkv', []).append(time.perf_counter() - t0)

        # ============ 5. CPU requant Q/K/V to INT4 + head split ============
        if rec: t0 = time.perf_counter()
        cs_q = self.tuned['q']; cs_k = self.tuned['k']; cs_v = self.tuned['v']
        Q_int4 = cpu_requant_after_vta(Q_int8, cs_q, S['attn_pre_out'], S['q_w'], S['q_out'], -8, 7)
        K_int4 = cpu_requant_after_vta(K_int8, cs_k, S['attn_pre_out'], S['k_w'], S['k_out'], -8, 7)
        V_int4 = cpu_requant_after_vta(V_int8, cs_v, S['attn_pre_out'], S['v_w'], S['v_out'], -8, 7)
        # Reshape into 3 heads. einops: 'b s (h d) -> h s d' becomes
        #   x.reshape(64, 3, 32).transpose(1, 0, 2)   -> (3, 64, 32)
        Qh = Q_int4.reshape(64, 3, 32).transpose(1, 0, 2)
        Kh = K_int4.reshape(64, 3, 32).transpose(1, 0, 2)
        Vh = V_int4.reshape(64, 3, 32).transpose(1, 0, 2)
        if rec: timings.setdefault('cpu_head_split', []).append(time.perf_counter() - t0)

        # ============ 6. Per-head attention ============
        # For Q@K^T: A = Q_h (64, 32) is input, B = K_h (64, 32) treated as weight
        #   (B's "out" axis = sequence position N=64, "in" axis = d_head=32). No transpose
        #   needed because VTA's GEMM is already C[i, j] = sum_k A[i, k] * B[j, k].
        # For attn@V:  A = attn (64, 64), B = V_h.T (32, 64) — V needs transposing
        #   so its rows index N=32 (out features) and cols index K=64 (reduction).
        if rec: t0 = time.perf_counter()
        ctx_heads = np.empty((3, 64, 32), dtype=np.int8)
        cs_qk = self.tuned['qk']; cs_av = self.tuned['av']
        for h in range(3):
            Q_h = Qh[h]
            K_h = Kh[h]
            V_h = Vh[h]
            # ----- Q @ K^T -----
            # Tile K as VTA wgt (m=N/BLOCK_OUT=4 tiles, n=K/BLOCK_IN=2 tiles),
            # pre-pack each (1, 2, 16, 16) slice for the m=1 module path.
            # _runtime_w_gemm handles M-chunking on Q_h when qkt o_tile < 64.
            K_tiled  = tile_for_wgt(K_h, m_tiles=4, n_tiles=2)
            K_slices = [pack_int4_for_vta(np.ascontiguousarray(K_tiled[m:m+1].astype(np.int8)))
                        for m in range(4)]
            scores_int8 = self._runtime_w_gemm('qkt_s3_m1', Q_h, K_slices)   # (64, 64)
            # CPU requant with attn_scale = 1/sqrt(96) folded in
            presoftmax = cpu_requant_after_vta(
                scores_int8, cs_qk, S['q_out'], S['k_out'], S['softmax_in'],
                -8, 7, extra_factor=SQRT_96)
            # ----- Softmax (float, CPU) -----
            scores_float = presoftmax.astype(np.float64) * S['softmax_in']
            scores_float = scores_float - scores_float.max(axis=-1, keepdims=True)
            exps = np.exp(scores_float)
            attn_float = exps / exps.sum(axis=-1, keepdims=True)
            attn_int4 = quant_signed(attn_float, S['softmax_out'], -8, 7).astype(np.int8)  # (64, 64)
            # ----- attn @ V -----
            V_T = V_h.T                                          # (32, 64)
            V_tiled  = tile_for_wgt(V_T, m_tiles=2, n_tiles=4)
            V_slices = [pack_int4_for_vta(np.ascontiguousarray(V_tiled[m:m+1].astype(np.int8)))
                        for m in range(2)]
            ctx_int8 = self._runtime_w_gemm('av_s3_m1', attn_int4, V_slices)   # (64, 32)
            ctx_int4 = cpu_requant_after_vta(
                ctx_int8, cs_av, S['softmax_out'], S['v_out'], S['o_in'], -8, 7)
            ctx_heads[h] = ctx_int4
        if rec: timings.setdefault('attention_heads', []).append(time.perf_counter() - t0)

        # ============ 7. Concat heads, O projection ============
        if rec: t0 = time.perf_counter()
        # einops 'h s d -> s (h d)' becomes transpose(1, 0, 2).reshape(64, 96)
        ctx_cat = ctx_heads.transpose(1, 0, 2).reshape(64, 96)   # (64, 96) int8 in [-8,7]
        o_int8 = self._proj('proj_k96_s3_m1', 'o_proj', ctx_cat)
        cs_o = self.tuned['o']
        o_int4 = cpu_requant_after_vta(
            o_int8, cs_o, S['o_in'], S['o_w'], S['attn_residual'], -8, 7)
        if rec: timings.setdefault('vta_o_proj', []).append(time.perf_counter() - t0)

        # ============ 8. Attention residual add ============
        if rec: t0 = time.perf_counter()
        # Skip = positional output; re-quant from pos_out scale to attn_residual scale
        skip_float = after_pos.astype(np.float64) * S['pos_out']
        skip_int4  = quant_signed(skip_float, S['attn_residual'], -8, 7).astype(np.int32)
        attn_block_out = o_int4.astype(np.int32) + skip_int4    # values in [-16, 14]
        if rec: timings.setdefault('residual_attn', []).append(time.perf_counter() - t0)

        # ============ 9. Pre-MLP BN + INT4 quant ============
        if rec: t0 = time.perf_counter()
        x_mlp_in_float = attn_block_out.astype(np.float64) * S['attn_residual']
        bn_mlp = (x_mlp_in_float - self.bn_mlp_mean.astype(np.float64)[None, :]) / \
                  np.sqrt(self.bn_mlp_var.astype(np.float64)[None, :] + 1e-5)
        mlp_pre = quant_signed(bn_mlp, S['mlp_bn_out'], -8, 7).astype(np.int8)
        if rec: timings.setdefault('bn_mlp', []).append(time.perf_counter() - t0)

        # ============ 10. fc1 (4-arg, m=1 chunked) -> ReLU + unsigned INT4 + zp shift ============
        if rec: t0 = time.perf_counter()
        fc1_out8 = self._mlp('fc1_s3_m1', 'fc1', mlp_pre)        # (64, 384) int8
        cs_fc1 = self.tuned['fc1']
        # Mode E for fc1 clips to [0, 15] (unsigned post-ReLU), then converts to signed [-8, 7]
        # for fc2 input via zero-point offset 8.
        recovered = fc1_out8.astype(np.float64) * (2.0 ** cs_fc1)
        real = recovered * (S['mlp_bn_out'] * S['fc1_w'])
        fc1_unsigned = np.clip(np.round(real / S['fc1_out']), 0, 15).astype(np.int8)
        fc1_signed = (fc1_unsigned.astype(np.int32) - 8).astype(np.int8)
        if rec: timings.setdefault('vta_fc1', []).append(time.perf_counter() - t0)

        # ============ 11. fc2 (4-arg with corrected bias, m=1 chunked) -> INT4 ============
        if rec: t0 = time.perf_counter()
        fc2_out8 = self._mlp('fc2_s4_m1', 'fc2', fc1_signed)     # (64, 96) int8
        cs_fc2 = self.tuned['fc2']
        fc2_int4 = cpu_requant_after_vta(
            fc2_out8, cs_fc2, S['fc1_out'], S['fc2_w'], S['mlp_residual'], -8, 7)
        if rec: timings.setdefault('vta_fc2', []).append(time.perf_counter() - t0)

        # ============ 12. MLP residual add ============
        if rec: t0 = time.perf_counter()
        skip_float_mlp = attn_block_out.astype(np.float64) * S['attn_residual']
        skip_int_mlp = quant_signed(skip_float_mlp, S['mlp_residual'], -8, 7).astype(np.int32)
        mlp_block_out = fc2_int4.astype(np.int32) + skip_int_mlp
        if rec: timings.setdefault('residual_mlp', []).append(time.perf_counter() - t0)

        # ============ 13. Classifier (CPU GAP + float matmul + argmax) ============
        if rec: t0 = time.perf_counter()
        mlp_float = mlp_block_out.astype(np.float64) * S['mlp_residual']
        gap = mlp_float.mean(axis=0)                                # (96,)
        logits = gap @ self.W_cls_float.T + self.b_cls_f64         # (24,)
        pred = int(np.argmax(logits))
        if rec: timings.setdefault('classifier', []).append(time.perf_counter() - t0)

        if rec: timings.setdefault('total', []).append(time.perf_counter() - t0_total)
        return pred


# ============================================================
# CLI
# ============================================================
def fmt_ms(seconds):
    return f"{seconds * 1000:.3f} ms"


def print_timings(timings, n_calls):
    print(f"\nTiming summary over {n_calls} inferences (ms / inference):")
    order = ['patch_embed', 'positional', 'bn_attn', 'vta_qkv', 'cpu_head_split',
             'attention_heads', 'vta_o_proj', 'residual_attn', 'bn_mlp',
             'vta_fc1', 'vta_fc2', 'residual_mlp', 'classifier', 'total']
    print(f"  {'section':22s}  {'mean ms':>10s}  {'std ms':>8s}  {'min':>8s}  {'max':>8s}  {'pct':>6s}")
    total_mean = float(np.mean(timings['total'])) if timings.get('total') else 1e-9
    for k in order:
        if k not in timings: continue
        arr = np.array(timings[k])
        mean = arr.mean() * 1000
        std  = arr.std()  * 1000
        mn   = arr.min()  * 1000
        mx   = arr.max()  * 1000
        pct  = (arr.mean() / total_mean) * 100 if k != 'total' else 100.0
        print(f"  {k:22s}  {mean:>10.3f}  {std:>8.3f}  {mn:>8.3f}  {mx:>8.3f}  {pct:>5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-dir', default='/home/xilinx/transformer_export',
                    help='Path to transformer_export/ on the board')
    ap.add_argument('--data', default='/home/xilinx/data/radioml2018_eval_snr_filtered.npz',
                    help='Path to RadioML eval npz')
    ap.add_argument('--validate', action='store_true',
                    help='Run full eval, report accuracy')
    ap.add_argument('--benchmark', type=int, default=0,
                    help='Run N inferences with timing breakdown')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--reference-preds', default=None,
                    help='Path to mode-E-tuned argmax .npy for match check')
    ap.add_argument('--bitstream-extra', default=None,
                    help='Optional extra path to look for the .bit file')
    ap.add_argument('--out-json', default=None,
                    help='Where to write the result summary JSON')
    args = ap.parse_args()

    extra = [args.bitstream_extra] if args.bitstream_extra else None
    setup_board(extra)

    print(f"\n[init] loading model from {args.export_dir}")
    model = TransformerVTA(args.export_dir)

    print(f"[data] loading {args.data}")
    signals, labels = load_radioml(args.data)
    N_total = len(signals)
    print(f"[data] {N_total} samples, signal shape {signals.shape[1:]}, "
          f"label range {labels.min()}-{labels.max()}")

    summary = {
        "export_dir": args.export_dir,
        "data_path":  args.data,
        "n_samples":  N_total,
        "tuned_shifts": model.tuned,
        "predicted_accuracy_phase3_mode_e_tuned": model.cfg['predicted_accuracy']['mode_e_tuned'],
    }

    if args.validate:
        ref = None
        if args.reference_preds:
            ref = np.load(args.reference_preds)
            print(f"[ref] loaded {len(ref)} reference predictions from {args.reference_preds}")
            assert len(ref) >= N_total, (
                f"reference preds ({len(ref)}) shorter than eval ({N_total})")

        # Warmup
        print(f"[warmup] {args.warmup} samples")
        for _ in range(args.warmup):
            model.infer(signals[0])

        print(f"[run] full validation, {N_total} samples")
        preds = np.empty(N_total, dtype=np.int32)
        correct = 0
        match_ref = 0
        t0 = time.time()
        for i in range(N_total):
            preds[i] = model.infer(signals[i])
            if preds[i] == labels[i]:
                correct += 1
            if ref is not None and preds[i] == ref[i]:
                match_ref += 1
            if (i + 1) % 10000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{i+1}/{N_total}]  acc={100*correct/(i+1):.2f}%  "
                      f"({rate:.1f} samp/s)")
                gc.collect()
        elapsed = time.time() - t0
        acc = 100 * correct / N_total
        fps = N_total / elapsed
        latency_ms = elapsed / N_total * 1000
        print(f"\n[validate]  accuracy: {acc:.2f}%  ({correct}/{N_total})")
        print(f"[validate]  throughput: {fps:.1f} fps   latency: {latency_ms:.2f} ms/sample")
        if ref is not None:
            print(f"[validate]  match-vs-mode-E-tuned: {match_ref}/{N_total} "
                  f"({100*match_ref/N_total:.2f}%)")
        summary["validate"] = {
            "accuracy_pct": acc, "correct": correct, "total": N_total,
            "elapsed_s": elapsed, "throughput_fps": fps,
            "latency_ms_per_sample": latency_ms,
        }
        if ref is not None:
            summary["validate"]["match_vs_reference"] = int(match_ref)
            summary["validate"]["match_vs_reference_pct"] = 100 * match_ref / N_total

        # Save full per-sample predictions
        preds_out = os.path.join(args.export_dir, 'board_validate_preds.npy')
        np.save(preds_out, preds)
        summary["validate"]["preds_file"] = preds_out
        print(f"[save] {preds_out}")

    if args.benchmark > 0:
        print(f"\n[warmup] {args.warmup} samples")
        for _ in range(args.warmup):
            model.infer(signals[0])
        print(f"[benchmark] {args.benchmark} timed inferences")
        timings = {}
        for i in range(args.benchmark):
            model.infer(signals[i % N_total], timings=timings)
        print_timings(timings, args.benchmark)
        summary["benchmark"] = {
            "n_runs": args.benchmark,
            "timings_ms_mean": {k: float(np.mean(v) * 1000) for k, v in timings.items()},
            "timings_ms_std":  {k: float(np.std(v)  * 1000) for k, v in timings.items()},
            "fps_from_total_mean": (1.0 / float(np.mean(timings['total'])))
                                    if 'total' in timings else None,
        }

    out_path = args.out_json or os.path.join(args.export_dir, 'board_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == '__main__':
    main()
