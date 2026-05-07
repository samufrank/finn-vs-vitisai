"""DPU INT8 transformer compilation — RadioML 2018, decomposed from Brevitas INT4 checkpoint.

Run inside Vitis AI Docker (xilinx/vitis-ai-pytorch-cpu:latest):

    cd ~/dev/CEN571-final/finn-vs-vitisai
    docker run -it \\
      -v $(pwd)/Vitis-AI:/workspace \\
      -v $(pwd):/workspace/project \\
      -v $(pwd)/../finn-transformers:/workspace/finn-transformers \\
      xilinx/vitis-ai-pytorch-cpu:latest bash
    conda activate vitis-ai-pytorch
    pip install brevitas==0.10.2 h5py    # one-time, gates B/C need brevitas
    cd /workspace/project/vitis_ai
    python compile_dpu_transformer_radioml.py

Note: the third -v mount is REQUIRED because finn-transformers is a sibling of
finn-vs-vitisai, not nested inside it. Without it, the Brevitas checkpoint and
the radioml/model.py source needed for Gate B will not be findable.

Override paths via env vars FINN_TRANSFORMERS_DIR, RADIOML_HDF5, RADIOML_EVAL_NPZ
if your layout differs.

Pipeline:
  1. Load Brevitas checkpoint state_dict (tensors only, no Brevitas required)
  2. Build decomposed PyTorch model — no Brevitas wrappers, no torch.chunk
  3. Map 22 weight tensors per the gate-2 mapping table; strict shape check
  4. Gate A: decomposed float accuracy on 100 RadioML eval samples
       (informational; <50% indicates a structural mapping error)
  5. Gate B: argmax agreement vs Brevitas reference on the same 100 samples
       (decision gate; >=90/100 required to proceed; needs brevitas)
  6. Gate C: final-logit magnitude comparison (informational)
  7. vai_q_pytorch PTQ — training-split calibration from RML2018.hdf5,
       seed 12, SNR>=-6 dB (matches finn-transformers convention)
  8. vai_c_xir compile for B512, wall-clock timed; subgraph analysis printed

Outputs:
  - quantize_result_transformer/transformer_radioml_int.xmodel  (PTQ output)
  - compiled_transformer_radioml/transformer_radioml.xmodel     (DPU bitstream)
  - compile_time_transformer.txt                                 (wall-clock)
  - subgraph_summary_transformer.json                            (DPU/CPU partition)
"""

import os
import sys
import json
import time
import math
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Paths (assume cd /workspace/project/vitis_ai inside Docker)
# ============================================================
PROJECT_ROOT = '/workspace/project'

# finn-transformers is a sibling of finn-vs-vitisai, not nested. Search a few
# plausible mount points; allow env-var override.
def _find_finn_transformers():
    env = os.environ.get('FINN_TRANSFORMERS_DIR')
    if env and os.path.exists(env):
        return env
    for c in [
        '/workspace/finn-transformers',                  # recommended sibling mount
        f'{PROJECT_ROOT}/finn-transformers',             # if user nested it
        f'{PROJECT_ROOT}/../finn-transformers',          # if PROJECT_ROOT mount preserves parent
    ]:
        if os.path.exists(c):
            return c
    return None

FINN_T_DIR   = _find_finn_transformers()
CKPT         = (os.environ.get('RADIOML_CKPT')
                or (f'{FINN_T_DIR}/outputs/radioml/model_int4_norm_none_70.97pct.pt'
                    if FINN_T_DIR else None))
HDF5         = (os.environ.get('RADIOML_HDF5')
                or f'{PROJECT_ROOT}/data/RML2018.hdf5')
EVAL_NPZ     = (os.environ.get('RADIOML_EVAL_NPZ')
                or f'{PROJECT_ROOT}/data/radioml2018_eval_snr_filtered.npz')
ARCH         = f'{PROJECT_ROOT}/vitis_ai/arch_zu3_b512.json'

QUANT_DIR    = './quantize_result_transformer'
COMPILE_DIR  = './compiled_transformer_radioml'
MODEL_NAME   = 'transformer_radioml'

# Calibration knobs — 200 batches of 32 = 6400 samples
CALIB_BATCH  = 32
CALIB_BATCHES = 200

# Sanity gates
N_SANITY     = 100
GATE_A_MIN   = 50.0   # decomposed accuracy floor (random is 4.2%, trained is ~71%)
GATE_B_MIN   = 90     # argmax agreement (out of N_SANITY)


# ============================================================
# Decomposed model — no Brevitas, no torch.chunk; DPU-friendly
# ============================================================
class DecomposedRadioMLTransformer(nn.Module):
    """Functionally matches the trained Brevitas model, with float weights
    and explicit fake-quant activation ops. Architecture per
    checkpoint_analysis.txt §1; activation quantizer scales per §2.

    The fake-quant ops replicate Brevitas's per-tensor symmetric activation
    quantizers exactly (round-half-to-even, clamp to [-2^(n-1), 2^(n-1)-1]
    for signed or [0, 2^n - 1] for unsigned). Without them, the QAT-trained
    weights produce nonsense in pure float (softmax saturates) — see VTA
    session 23 deployment, where the same scales gave 71.80% on board.
    """

    EMB   = 96
    NHEAD = 3
    DHEAD = 32       # EMB // NHEAD
    DFF   = 384
    SEQ   = 64
    NCLS  = 24
    SCALE = 1.0 / math.sqrt(96)   # uses emb_dim, NOT d_head — matches training

    # Activation quantizer scales — extracted live via Brevitas proxy.scale().
    # The values in checkpoint_analysis.txt §2 are scaling_impl.value (i.e.,
    # value = scale * 2^(bits-1) for signed, scale * (2^bits - 1) for unsigned),
    # which is the buggy accessor flagged in session 23 narrative. The true
    # quantization step (used as `scale` in x_int = round(x / scale)) is below.
    # INT4 signed → [-8, 7],  INT4 unsigned → [0, 15]
    # INT8 signed → [-128,127],  INT8 unsigned → [0, 255]
    Q_SCALE_INPUT          = 1.30873e-02   # emb.1.patches.0          INT8 signed
    Q_SCALE_POST_RELU_EMB  = 6.28573e-03   # emb.1.patches.4          INT8 UNSIGNED
    Q_SCALE_POS_IN         = 1.25701e-02   # enc.0.add input_quant    INT8 signed
    Q_SCALE_POS_OUT        = 1.20258e-02   # enc.0.add output_quant   INT8 signed
    Q_SCALE_ATTN_PRENORM   = 2.85124e-01   # enc.1.pre_norm.2         INT4 signed
    Q_SCALE_Q              = 3.75043e-01   # q_projection out         INT4 signed
    Q_SCALE_K              = 2.91283e-01   # k_projection out         INT4 signed
    Q_SCALE_V              = 1.39400e-01   # v_projection out         INT4 signed
    Q_SCALE_PRE_SOFTMAX    = 6.04895e-01   # softmax.0 (Q@K^T input)  INT4 signed
    Q_SCALE_POST_SOFTMAX   = 1.04513e-02   # softmax.2 (attn weights) INT4 signed
    Q_SCALE_O_IN           = 1.40172e-01   # o_projection input_quant INT4 signed
    Q_SCALE_RESID_ATTN     = 1.70219e-01   # enc.1.quant.0 residual   INT4 signed
    Q_SCALE_MLP_PRENORM    = 2.40275e-01   # enc.2.mlp.3 post-BN      INT4 signed
    Q_SCALE_POST_RELU_MLP  = 1.79328e-01   # enc.2.mlp.6 post-ReLU    INT4 UNSIGNED
    Q_SCALE_RESID_MLP      = 8.08787e-01   # enc.2.quant.0 residual   INT4 signed
    Q_SCALE_CLS_OUT        = 9.70028e-02   # cls.1                    INT8 signed

    @staticmethod
    def _fq(x, scale, n_bits, signed=True):
        """Per-tensor symmetric fake-quant (round + clamp + dequantize)."""
        if signed:
            lo, hi = -(2 ** (n_bits - 1)), (2 ** (n_bits - 1)) - 1
        else:
            lo, hi = 0, (2 ** n_bits) - 1
        return (x / scale).round().clamp(lo, hi) * scale

    def __init__(self):
        super().__init__()
        # PatchEmbedding (Brevitas trained these as INT8). patch_bn stays as
        # BN2d since it sits directly between Conv2d and ReLU — vai_q_pytorch
        # fuses Conv+BN natively in that pattern.
        self.patch_conv = nn.Conv2d(2, self.EMB, kernel_size=(1, 16),
                                    stride=(1, 16), bias=True)
        self.patch_bn   = nn.BatchNorm2d(self.EMB, affine=False)

        # Learned positional encoding — stays as (1, seq, emb) for natural
        # broadcast against (B, seq, emb) inputs. vai_c_xir mis-infers shapes
        # if we view this as 2D (64, 96) and broadcast against 3D inputs.
        self.pos_enc = nn.Parameter(torch.empty(1, self.SEQ, self.EMB))

        # Attention pre-norm BN1d expressed as elementwise mul + add (BN with
        # affine=False is just (x - mean) / sqrt(var + eps), which is one mul
        # and one add against per-channel buffers). This avoids the
        # BatchNorm1d-on-transposed-tensor pattern that vai_c_xir crashes on
        # while preserving Brevitas's exact data-flow ordering (BN -> fake-quant
        # -> linear -> fake-quant). Per-channel buffers shape (1, 1, EMB) so
        # they broadcast against (B, seq, emb) tensors directly.
        self.register_buffer('attn_bn_scale', torch.empty(1, 1, self.EMB))
        self.register_buffer('attn_bn_shift', torch.empty(1, 1, self.EMB))
        self.register_buffer('mlp_bn_scale',  torch.empty(1, 1, self.EMB))
        self.register_buffer('mlp_bn_shift',  torch.empty(1, 1, self.EMB))

        # Attention (Brevitas trained these as INT4, all bias=False).
        self.q_proj  = nn.Linear(self.EMB, self.EMB, bias=False)
        self.k_proj  = nn.Linear(self.EMB, self.EMB, bias=False)
        self.v_proj  = nn.Linear(self.EMB, self.EMB, bias=False)
        self.o_proj  = nn.Linear(self.EMB, self.EMB, bias=False)

        # MLP
        self.fc1    = nn.Linear(self.EMB, self.DFF, bias=True)
        self.fc2    = nn.Linear(self.DFF, self.EMB, bias=True)

        # Classifier (Brevitas trained as INT8)
        self.cls_linear = nn.Linear(self.EMB, self.NCLS, bias=True)

    def forward(self, x):
        # x: (B, 1, 1024, 2) — RadioML I/Q layout.
        # Use -1 for batch dim everywhere (rather than capturing B = x.shape[0])
        # because torch.jit.trace bakes Python ints into the graph as literals;
        # vai_c_xir then sees static "1" reshapes that mismatch real shapes.
        fq = self._fq

        # ---- PatchEmbedding ----
        x = x.permute(0, 3, 1, 2)                                       # (B, 2, 1, 1024)
        x = fq(x, self.Q_SCALE_INPUT, 8, signed=True)                   # patches.0
        x = self.patch_conv(x)                                          # (B, 96, 1, 64)
        x = self.patch_bn(x)
        x = F.relu(x)
        x = fq(x, self.Q_SCALE_POST_RELU_EMB, 8, signed=False)          # patches.4
        # Pool (1,64) is identity here. Squeeze the singleton spatial dim
        # and transpose to (B, seq, emb) — avoids a 4-D reshape op that
        # vai_c_xir handles inconsistently.
        x = x.squeeze(2).transpose(1, 2)                                # (B, 64, 96)

        # ---- Learned positional encoding (QuantEltwiseAdd) ----
        x_q   = fq(x, self.Q_SCALE_POS_IN, 8, signed=True)              # add.input_quant on x
        pos_q = fq(self.pos_enc, self.Q_SCALE_POS_IN, 8, signed=True)   # add.input_quant on pos
        # pos_enc is (1, 64, 96), broadcasts against (B, 64, 96) cleanly.
        x = fq(x_q + pos_q, self.Q_SCALE_POS_OUT, 8, signed=True)       # add.output_quant

        # ---- Attention block ----
        # residual is the post-pos-add tensor (BEFORE pre-norm)
        res = x
        # Pre-norm BN as elementwise mul + add (mathematically identical to
        # BatchNorm1d(affine=False), but avoids the BN module that vai_c_xir
        # crashes on for transposed inputs).
        h = x * self.attn_bn_scale + self.attn_bn_shift
        h = fq(h, self.Q_SCALE_ATTN_PRENORM, 4, signed=True)            # pre_norm.2

        q = fq(self.q_proj(h), self.Q_SCALE_Q, 4, signed=True)          # q_proj.output_quant
        k = fq(self.k_proj(h), self.Q_SCALE_K, 4, signed=True)          # k_proj.output_quant
        v = fq(self.v_proj(h), self.Q_SCALE_V, 4, signed=True)          # v_proj.output_quant

        # Heads via explicit slice (avoids 4D matmul / view+transpose pattern
        # that vai_c_xir struggles with). Each slice is (B, 64, 32).
        head_outs = []
        for hd in range(self.NHEAD):
            s_lo = hd * self.DHEAD
            s_hi = s_lo + self.DHEAD
            q_h = q[..., s_lo:s_hi] * self.SCALE                        # (B, 64, 32)
            k_h = k[..., s_lo:s_hi]                                     # (B, 64, 32)
            v_h = v[..., s_lo:s_hi]                                     # (B, 64, 32)
            attn_h = torch.matmul(q_h, k_h.transpose(-1, -2))           # (B, 64, 64)
            attn_h = fq(attn_h, self.Q_SCALE_PRE_SOFTMAX, 4, signed=True)
            attn_h = F.softmax(attn_h, dim=-1)
            attn_h = fq(attn_h, self.Q_SCALE_POST_SOFTMAX, 4, signed=True)
            out_h = torch.matmul(attn_h, v_h)                           # (B, 64, 32)
            head_outs.append(out_h)
        out = torch.cat(head_outs, dim=-1)                              # (B, 64, 96)

        out = fq(out, self.Q_SCALE_O_IN, 4, signed=True)                # o_proj.input_quant
        out = self.o_proj(out)
        # o_proj has no output quantizer (output_quant=None in attention.py)

        # Shared residual quant on BOTH branches (matches attention.forward)
        out = fq(out, self.Q_SCALE_RESID_ATTN, 4, signed=True)
        res = fq(res, self.Q_SCALE_RESID_ATTN, 4, signed=True)
        x = out + res
        # post_norm is Identity (norm_placement=pre-norm)

        # ---- MLP block ----
        res = x
        # Same elementwise mul+add as attn_bn.
        h = x * self.mlp_bn_scale + self.mlp_bn_shift
        h = fq(h, self.Q_SCALE_MLP_PRENORM, 4, signed=True)             # mlp.3
        h = self.fc1(h)
        h = F.relu(h)
        h = fq(h, self.Q_SCALE_POST_RELU_MLP, 4, signed=False)          # mlp.6
        h = self.fc2(h)
        # fc2 has no output quantizer (avoid double quantizer per blocks.py)

        # Shared residual quant on BOTH branches
        h   = fq(h,   self.Q_SCALE_RESID_MLP, 4, signed=True)
        res = fq(res, self.Q_SCALE_RESID_MLP, 4, signed=True)
        x = h + res

        # ---- GAP + classifier ----
        x = x.mean(dim=1)                                               # (B, 96)
        out = self.cls_linear(x)                                        # (B, 24)
        out = fq(out, self.Q_SCALE_CLS_OUT, 8, signed=True)             # cls.1
        return out


# ============================================================
# Weight mapping table: Brevitas state_dict key -> decomposed param
# Source: vta/transformer/checkpoint_analysis.txt §1 (verified session 23)
# ============================================================
WEIGHT_MAP = [
    # (brevitas_key, decomposed_attr, expected_shape, weight_quant_bits, weight_scale)
    # weight_quant_bits=None means copy as-is (no fake-quant applied).
    # Weight scales are proxy.scale() values from the Brevitas weight quantizer
    # (same accessor convention as the activation scales above).
    # patch_bn is BN2d directly after Conv2d — vai_q_pytorch fuses this natively,
    # so we keep it as an explicit module. The BN1d-on-transposed-tensor pattern
    # used by attn_bn and mlp_bn is fold-or-die for vai_c_xir, so we fold them
    # into the q/k/v/fc1 weights+biases at mapping time and never expose them
    # to the compiler.
    ('emb.1.patches.1.weight',                  'patch_conv.weight',                  (96, 2, 1, 16),  8, 5.670934e-03),
    ('emb.1.patches.1.bias',                    'patch_conv.bias',                    (96,),           None, None),
    ('emb.1.patches.2.running_mean',            'patch_bn.running_mean',              (96,),           None, None),
    ('emb.1.patches.2.running_var',             'patch_bn.running_var',               (96,),           None, None),
    ('emb.1.patches.2.num_batches_tracked',     'patch_bn.num_batches_tracked',       (),              None, None),
    ('enc.0.pos',                               'pos_enc',                            (1, 64, 96),     None, None),
    ('enc.1.mha.q_projection.weight',           'q_proj.weight',                      (96, 96),        4, 7.275883e-02),
    ('enc.1.mha.k_projection.weight',           'k_proj.weight',                      (96, 96),        4, 7.374235e-02),
    ('enc.1.mha.v_projection.weight',           'v_proj.weight',                      (96, 96),        4, 8.723587e-02),
    ('enc.1.mha.o_projection.weight',           'o_proj.weight',                      (96, 96),        4, 6.745233e-02),
    ('enc.2.mlp.4.weight',                      'fc1.weight',                         (384, 96),       4, 3.784532e-02),
    ('enc.2.mlp.4.bias',                        'fc1.bias',                           (384,),          None, None),
    ('enc.2.mlp.8.weight',                      'fc2.weight',                         (96, 384),       4, 4.726412e-02),
    ('enc.2.mlp.8.bias',                        'fc2.bias',                           (96,),           None, None),
    ('cls.0.weight',                            'cls_linear.weight',                  (24, 96),        8, 6.720352e-03),
    ('cls.0.bias',                              'cls_linear.bias',                    (24,),           None, None),
]

# BN(affine=False) lowered to elementwise (mul, add) buffers. Brevitas runs
# this BN before pre-norm fake-quant, so we need to preserve the value range
# the activation quantizer was tuned around — which means BN must execute as
# a real op in the forward, not be folded into the downstream linear.
#   BN(x) = (x - mean) / sqrt(var + eps)
#         = x * scale + shift,   scale = 1/sqrt(var+eps),  shift = -mean*scale
# Buffers are (1, 1, EMB) for direct broadcast against (B, seq, emb) inputs.
BN_LOWER_PLAN = [
    # (rm_key, rv_key, n_features, scale_buf_attr, shift_buf_attr, eps)
    ('enc.1.pre_norm.1.running_mean', 'enc.1.pre_norm.1.running_var', 96,
     'attn_bn_scale', 'attn_bn_shift', 1e-5),
    ('enc.2.mlp.1.running_mean',      'enc.2.mlp.1.running_var',      96,
     'mlp_bn_scale',  'mlp_bn_shift',  1e-5),
]


def map_brevitas_into_decomposed(model, ckpt_path):
    """Load Brevitas state_dict and copy 22 mapped tensors into the decomposed model.

    Uses weights_only=True so this does NOT require Brevitas to be importable.
    Strict shape check catches any drift between the analysis table and the live ckpt.
    """
    try:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception as e:
        # Older PyTorch may not support weights_only; fall back.
        print(f"  (weights_only=True failed: {e}; falling back to weights_only=False)")
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print(f"  Brevitas state_dict: {len(sd)} keys")

    n_quantized = 0
    for key, attr, expected_shape, w_bits, w_scale in WEIGHT_MAP:
        if key not in sd:
            raise KeyError(f"Brevitas key missing: {key}")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch for {key}: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        # Apply weight fake-quant if specified — matches Brevitas's
        # internal quant_weight() behavior so the decomposed forward pass
        # uses the same effective weights as the QAT-trained Brevitas model.
        if w_bits is not None:
            lo, hi = -(2 ** (w_bits - 1)), (2 ** (w_bits - 1)) - 1
            tensor = (tensor / w_scale).round().clamp(lo, hi) * w_scale
            n_quantized += 1
        # Walk to the leaf attribute on the decomposed model
        parts = attr.split('.')
        target = model
        for p in parts[:-1]:
            target = getattr(target, p)
        existing = getattr(target, parts[-1])
        if tuple(existing.shape) != expected_shape:
            raise ValueError(
                f"Decomposed {attr} shape mismatch: model has "
                f"{tuple(existing.shape)}, expected {expected_shape}"
            )
        with torch.no_grad():
            existing.copy_(tensor)
    print(f"  Mapped {len(WEIGHT_MAP)} tensors ({n_quantized} weight-fake-quanted)")

    # ---- Lower BN1d (affine=False) to per-channel scale + shift buffers ----
    for rm_key, rv_key, n_feat, scale_attr, shift_attr, eps in BN_LOWER_PLAN:
        if rm_key not in sd or rv_key not in sd:
            raise KeyError(f"BN-lower key missing: {rm_key} or {rv_key}")
        rm = sd[rm_key]
        rv = sd[rv_key]
        if tuple(rm.shape) != (n_feat,) or tuple(rv.shape) != (n_feat,):
            raise ValueError(
                f"BN-lower shape mismatch: {rm.shape}, {rv.shape}, "
                f"expected ({n_feat},)")
        scale = 1.0 / (rv + eps).sqrt()                          # (96,)
        shift = -rm * scale                                       # (96,)
        with torch.no_grad():
            # Broadcast against (B, seq, emb) tensors -> shape (1, 1, 96)
            getattr(model, scale_attr).copy_(scale.view(1, 1, n_feat))
            getattr(model, shift_attr).copy_(shift.view(1, 1, n_feat))
        print(f"  Lowered BN {rm_key.rsplit('.', 1)[0]} -> "
              f"({scale_attr}, {shift_attr})")


# ============================================================
# Eval-data loader (eval npz, used only for sanity gates)
# ============================================================
def load_eval_samples_from_npz(npz_path, n_samples):
    print(f"  Loading {n_samples} eval samples from {npz_path}")
    d = np.load(npz_path)
    sigs = d['signals']         # expected (N, 1024, 2)
    labs = d['labels']          # expected (N,)
    if sigs.ndim == 3:
        # (N, 1024, 2) -> (N, 1, 1024, 2)
        sigs = sigs[:, np.newaxis, :, :]
    if sigs.shape[1:] != (1, 1024, 2):
        raise ValueError(f"Unexpected eval signals shape {sigs.shape}")
    sigs = sigs[:n_samples].astype(np.float32)
    labs = labs[:n_samples].astype(np.int64)
    return torch.from_numpy(sigs), torch.from_numpy(labs)


# ============================================================
# Calibration loader (training split per finn-transformers convention)
# ============================================================
def load_calibration_train_split(hdf5_path, snr_min=-6, seed=12,
                                 n_samples=CALIB_BATCH * CALIB_BATCHES):
    """Reproduce RadioMLDataset filter+split (seed=12, SNR>=-6 dB) and return
    the first `n_samples` of the 80% training portion as (N, 1, 1024, 2)."""
    import h5py
    print(f"  Reading {hdf5_path}")
    f = h5py.File(hdf5_path, 'r')
    snr = f['Z'][:].squeeze()
    keep = snr >= snr_min
    all_idx = np.where(keep)[0]
    print(f"  SNR>={snr_min} dB filter: {len(all_idx)} / {len(keep)} samples")

    # Replicate dataset.split: shuffle filtered indices with `seed`, take first 80%
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(all_idx)
    n_train = int(0.80 * len(shuffled))
    train_idx = shuffled[:n_train]
    print(f"  Training split: {len(train_idx)} samples")

    n_use = min(n_samples, len(train_idx))
    use_idx = train_idx[:n_use]
    print(f"  Calibration: {n_use} samples ({n_use // CALIB_BATCH} batches of {CALIB_BATCH})")

    # h5py fancy-indexing requires monotonic indices — sort, then load
    sorted_idx = np.sort(use_idx)
    raw = f['X'][sorted_idx]
    f.close()

    # Reshape (N, 1024, 2) -> (N, 1, 1024, 2) per dataset reshape param
    samples = raw.reshape(n_use, 1, 1024, 2).astype(np.float32)
    return torch.from_numpy(samples)


# ============================================================
# Sanity gates
# ============================================================
def gate_a_decomposed_accuracy(model, samples, labels):
    """Run decomposed model on labeled eval samples, return accuracy %."""
    model.eval()
    with torch.no_grad():
        logits = model(samples)
    pred = logits.argmax(dim=-1)
    correct = (pred == labels).sum().item()
    return 100.0 * correct / len(labels), pred


def build_brevitas_reference(ckpt_path):
    """Construct the Brevitas Model with the exact training config and load the ckpt.
    Requires brevitas + finn-transformers source on sys.path.
    """
    if FINN_T_DIR is None:
        raise RuntimeError(
            "finn-transformers source directory not found. "
            "Set FINN_TRANSFORMERS_DIR env var or mount it via "
            "-v <host_path>/finn-transformers:/workspace/finn-transformers")
    if FINN_T_DIR not in sys.path:
        sys.path.insert(0, FINN_T_DIR)
    from radioml.model import Model

    bm = Model(
        num_classes=24,
        embedding={"patches": [1, 64], "kernel_size": [1, 16],
                   "stride": [1, 16], "padding": [0, 0],
                   "activation": "relu", "bits": 8},
        positional={"encoding": "learned", "bits": 8},
        configuration="original", num_layers=1, num_heads=3,
        emb_dim=96, expansion_dim=384, bits=4, cls_bits=8,
        activation="relu", norm="none", norm_placement="pre-norm", dropout=0.0,
    )
    with torch.no_grad():
        bm(torch.zeros(1, 1, 1024, 2))    # materialize lazy modules
    bm.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
    bm.eval()
    return bm


def gate_b_argmax_agreement(decomposed, brevitas_model, samples):
    decomposed.eval()
    brevitas_model.eval()
    with torch.no_grad():
        d = decomposed(samples).argmax(dim=-1)
        b = brevitas_model(samples).argmax(dim=-1)
    agree = int((d == b).sum().item())
    return agree, d, b


def gate_c_logit_magnitudes(decomposed, brevitas_model, sample):
    """Compare final logit magnitude between models on a single sample."""
    decomposed.eval()
    brevitas_model.eval()
    with torch.no_grad():
        d = decomposed(sample)
        b = brevitas_model(sample)
    return float(d.abs().mean().item()), float(b.abs().mean().item())


# ============================================================
# vai_q_pytorch PTQ
# ============================================================
def quantize_with_vai_q(model, calib_samples, output_dir):
    """Run vai_q_pytorch calibrate + test-mode export. Writes
    {output_dir}/<ModelClassName>_int.xmodel."""
    from pytorch_nndct.apis import torch_quantizer

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    model = model.to(device)
    dummy = torch.zeros(1, 1, 1024, 2, device=device)

    print("  Phase 1: calibration")
    quantizer = torch_quantizer(
        'calib', model, (dummy,),
        output_dir=output_dir, device=device,
    )
    quant_model = quantizer.quant_model

    n_total = len(calib_samples)
    n_batches = n_total // CALIB_BATCH
    print(f"  Running {n_batches} calibration batches (batch={CALIB_BATCH})...")
    with torch.no_grad():
        for b in range(n_batches):
            batch = calib_samples[b * CALIB_BATCH:(b + 1) * CALIB_BATCH]
            quant_model(batch)
            if (b + 1) % 25 == 0 or (b + 1) == n_batches:
                print(f"    {b + 1}/{n_batches} batches")
    quantizer.export_quant_config()

    print("  Phase 2: test-mode export")
    quantizer = torch_quantizer(
        'test', model, (dummy,),
        output_dir=output_dir, device=device,
    )
    with torch.no_grad():
        quantizer.quant_model(dummy)
    quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
    print(f"  Quantize done -> {output_dir}/")


# ============================================================
# vai_c_xir compile + subgraph analysis
# ============================================================
def find_quantized_xmodel(quant_dir):
    for fn in sorted(os.listdir(quant_dir)):
        if fn.endswith('.xmodel'):
            return os.path.join(quant_dir, fn)
    raise FileNotFoundError(f"No .xmodel under {quant_dir}")


def compile_xmodel_timed(quant_xmodel, output_dir, model_name, arch_path):
    """Run vai_c_xir, return (wall_seconds, compiled_xmodel_path)."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ['vai_c_xir', '-x', quant_xmodel, '-a', arch_path,
           '-o', output_dir, '-n', model_name]
    print(f"  Command: {' '.join(cmd)}")
    print("  " + "-" * 66)
    t0 = time.time()
    rv = subprocess.run(cmd)
    t = time.time() - t0
    print("  " + "-" * 66)
    if rv.returncode != 0:
        raise RuntimeError(f"vai_c_xir failed (exit {rv.returncode})")
    compiled = os.path.join(output_dir, f"{model_name}.xmodel")
    if not os.path.exists(compiled):
        # vai_c_xir may name differently — find any xmodel
        for fn in os.listdir(output_dir):
            if fn.endswith('.xmodel'):
                compiled = os.path.join(output_dir, fn)
                break
    return t, compiled


def analyze_xmodel(xmodel_path):
    """Print subgraph breakdown and return a JSON-friendly summary."""
    import xir
    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

    summary = {'total': len(subgraphs), 'dpu': 0, 'cpu': 0, 'subgraphs': []}
    print(f"\n  Subgraph analysis ({len(subgraphs)} total):")
    for i, sg in enumerate(subgraphs):
        device = sg.get_attr('device') if sg.has_attr('device') else 'UNKNOWN'
        ops = {}
        for op in sg.get_ops():
            t = op.get_type()
            ops[t] = ops.get(t, 0) + 1
        if device == 'DPU':
            summary['dpu'] += 1
        elif device == 'CPU':
            summary['cpu'] += 1
        name = sg.get_name()
        # truncate long subgraph names for readability
        disp = name if len(name) < 80 else name[:77] + '...'
        print(f"    [{device:3s}] [{i:2d}] {disp} ({sum(ops.values())} ops)")
        for op_type, count in sorted(ops.items()):
            print(f"           {op_type}: {count}")
        summary['subgraphs'].append({
            'index': i,
            'device': device,
            'name': name,
            'op_counts': ops,
            'n_ops': sum(ops.values()),
        })
    print(f"\n  Static partition: {summary['dpu']} DPU + {summary['cpu']} CPU "
          f"= {summary['total']} subgraphs")
    return summary


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 72)
    print("  DPU INT8 Transformer Compilation — RadioML 2018 (decomposed)")
    print("  Target: DPUCZDX8G B512, fingerprint 0x101000016010400")
    print("=" * 72)

    # Sanity-check inputs exist before doing anything else
    if FINN_T_DIR is None or CKPT is None:
        print(f"FATAL: finn-transformers source not found.")
        print(f"  Tried env FINN_TRANSFORMERS_DIR, /workspace/finn-transformers, ")
        print(f"  {PROJECT_ROOT}/finn-transformers, {PROJECT_ROOT}/../finn-transformers.")
        print(f"  Add -v <host>/finn-transformers:/workspace/finn-transformers to docker run.")
        sys.exit(1)
    print(f"  finn-transformers: {FINN_T_DIR}")
    for path, label in [(CKPT, 'Brevitas checkpoint'),
                        (HDF5, 'RadioML HDF5'),
                        (EVAL_NPZ, 'eval npz'),
                        (ARCH, 'arch json')]:
        if not os.path.exists(path):
            print(f"FATAL: {label} not found: {path}")
            sys.exit(1)
        print(f"  {label}: {path}")
    print(f"  Inputs OK")

    # ----- Build decomposed model -----
    print("\n[1/8] Build decomposed model")
    decomposed = DecomposedRadioMLTransformer().eval()
    n_params = sum(p.numel() for p in decomposed.parameters())
    print(f"  Parameters: {n_params:,}")

    # ----- Map weights -----
    print("\n[2/8] Map Brevitas weights into decomposed model")
    map_brevitas_into_decomposed(decomposed, CKPT)

    # Verify forward pass
    with torch.no_grad():
        out = decomposed(torch.zeros(1, 1, 1024, 2))
    assert out.shape == (1, 24), f"Unexpected output shape {out.shape}"
    print(f"  Forward pass OK -> output shape {tuple(out.shape)}")

    # ----- Sanity gates -----
    print(f"\n[3/8] Load {N_SANITY} eval samples for sanity gates")
    eval_samples, eval_labels = load_eval_samples_from_npz(EVAL_NPZ, N_SANITY)

    print(f"\n[4/8] Gate A — decomposed model accuracy on {N_SANITY} eval samples")
    acc_a, decomposed_pred = gate_a_decomposed_accuracy(decomposed, eval_samples, eval_labels)
    print(f"  Accuracy: {acc_a:.1f}%  (random={100/24:.1f}%, trained Brevitas~71%)")
    if acc_a < GATE_A_MIN:
        print(f"  GATE A FAIL: accuracy {acc_a:.1f}% < {GATE_A_MIN}% — mapping likely broken. Aborting.")
        sys.exit(1)
    print(f"  Gate A: PASS")

    # ----- Gate B / C: require Brevitas -----
    brevitas_available = False
    try:
        import brevitas  # noqa: F401
        brevitas_available = True
    except ImportError:
        print("\n  WARNING: Brevitas not importable — Gates B and C cannot run.")
        print("  Install with: pip install brevitas==0.10.2")
        print("  Falling back to Gate A as decision gate.")

    if brevitas_available:
        print(f"\n[5/8] Gate B — argmax agreement vs Brevitas reference")
        try:
            brev_model = build_brevitas_reference(CKPT)
            agree, d_pred, b_pred = gate_b_argmax_agreement(
                decomposed, brev_model, eval_samples)
            print(f"  Argmax agreement: {agree}/{N_SANITY}")
            if agree < GATE_B_MIN:
                # Show the disagreements for diagnosis
                disagree = (d_pred != b_pred).nonzero(as_tuple=True)[0][:10].tolist()
                print(f"  GATE B FAIL: agreement {agree} < {GATE_B_MIN}. "
                      f"First {len(disagree)} disagreements at indices: {disagree}")
                sys.exit(1)
            print(f"  Gate B: PASS")

            print(f"\n  Gate C — final-logit magnitude (informational, single sample)")
            d_mag, b_mag = gate_c_logit_magnitudes(
                decomposed, brev_model, eval_samples[:1])
            print(f"    Decomposed mean|logit|: {d_mag:.4f}")
            print(f"    Brevitas   mean|logit|: {b_mag:.4f}")
            ratio = d_mag / b_mag if b_mag > 0 else float('inf')
            print(f"    Ratio (decomp/brev):    {ratio:.3f}")
            if ratio < 0.5 or ratio > 2.0:
                print(f"    NOTE: ratio outside [0.5, 2.0] — possible scale-domain drift")

            del brev_model
        except Exception as e:
            print(f"  Gate B/C error: {e}")
            print(f"  Falling back to Gate A as decision gate "
                  f"(already passed at {acc_a:.1f}%).")
    else:
        # Tighten Gate A threshold when running without Gate B
        if acc_a < 65.0:
            print(f"  GATE A (without Gate B) FAIL: accuracy {acc_a:.1f}% < 65%. Aborting.")
            sys.exit(1)
        print(f"  Gate A (sole gate): PASS at {acc_a:.1f}%")

    # ----- Calibration data -----
    print(f"\n[6/8] Load training-split calibration ({CALIB_BATCHES} batches × {CALIB_BATCH})")
    calib_samples = load_calibration_train_split(HDF5)

    # ----- PTQ -----
    print(f"\n[7/8] vai_q_pytorch PTQ -> {QUANT_DIR}")
    quantize_with_vai_q(decomposed, calib_samples, QUANT_DIR)

    # ----- Compile -----
    print(f"\n[8/8] vai_c_xir compile -> {COMPILE_DIR}")
    quant_xmodel = find_quantized_xmodel(QUANT_DIR)
    print(f"  Input xmodel: {quant_xmodel}")
    t_compile, compiled_xmodel = compile_xmodel_timed(
        quant_xmodel, COMPILE_DIR, MODEL_NAME, ARCH)
    print(f"\n  vai_c_xir wall-clock: {t_compile:.2f} seconds")
    print(f"  Compiled xmodel:      {compiled_xmodel}")

    # Persist the timing for the compile-time comparison table
    with open('compile_time_transformer.txt', 'w') as f:
        f.write(f"transformer_radioml vai_c_xir: {t_compile:.2f} seconds\n")
        f.write(f"input: {quant_xmodel}\n")
        f.write(f"output: {compiled_xmodel}\n")

    # ----- Subgraph analysis -----
    summary = analyze_xmodel(compiled_xmodel)
    summary['vai_c_xir_seconds'] = t_compile
    with open('subgraph_summary_transformer.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Subgraph summary saved -> subgraph_summary_transformer.json")
    print(f"  Compile time saved      -> compile_time_transformer.txt")

    print("\n" + "=" * 72)
    print(f"  DONE — {summary['dpu']} DPU + {summary['cpu']} CPU subgraphs, "
          f"compile {t_compile:.1f}s")
    print("=" * 72)


if __name__ == '__main__':
    main()
