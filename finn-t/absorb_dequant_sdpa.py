"""AbsorbDequantIntoSDPA: delete pre-SDPA ElementwiseMul dequant nodes and
rescale the QK thresholds AND post-SDPA thresholds to compensate.

Root cause: in the Brevitas-exported graph, the per-head dequant-scale Muls
sit between the QKV projection's Split and the SDPA Q/K/V inputs. Those Muls
convert INT4 tensors to FLOAT32, which propagates to SDPA's QType/KType/VType
attributes. The SDPA HLS template (attention.hpp) accesses QType::width / etc.
and therefore requires ap_int types -- float is a compile-time error.

Fix:

1. Delete the three ElementwiseMul_hls dequant Muls feeding each SDPA's Q, K,
   V inputs and rewire the Split outputs directly into SDPA.
2. Set QType, KType, VType = INT4 on each SDPA.
3. Rescale QK thresholds (sdpa.input[3]):
       param0_new = param0_old / (s_Q * s_K)
   This preserves bin indices under the now-larger integer accumulator:
   QK_int = Q_int * K_int^T is 1/(s_Q * s_K) times larger than the float
   Q_fp * K_fp^T that the original thresholds were compared against.
4. Widen the QK-threshold dtype annotation to fit the rescaled range.
5. DequantSoftmax: unchanged. It scales the bin index, not the raw
   accumulator; bin indices are preserved by (3).

6. Rescale post-SDPA thresholds by 1/s_V, AND widen SDPA OType + downstream
   intermediate dtypes. Why this is needed (corrected from earlier analysis):

   The post-SDPA Thresholding (commonly Thresholding_rtl_9 after
   StreamingConcat_hls_0) was calibrated by streamline+convert_to_hw to bin
   the AV-matmul-output as if V were already dequantized:
       expected SDPA output = bin_softmax * V_fp summed
   and the post-AV scale (post_softmax_bin_spacing, e.g. ~0.01045) was
   absorbed into the Thresholding by multiplying its threshold values by
   1/post_av_scale (≈ 95). After absorption the thresholds match the
   *float* AV-matmul output magnitude — but s_V was NEVER absorbed.

   Once we delete Mul_2 (the V dequant), SDPA computes
       actual SDPA output = bin_softmax * V_int summed = (expected) / s_V
   which is 1/s_V times larger (e.g. ~7.18x for s_V=0.139). Without
   compensation:
   (a) The post-SDPA Thresholding bins everything into extreme bins because
       the input is 7x its calibrated range.
   (b) The SDPA OType=INT8 (set upstream by AnnotateSDPAOutputDtype) is now
       far too narrow for the actual integer accumulator, so the cast from
       AccAVMatMul=FLOAT32 down to INT8 saturates/wraps and destroys data.

   To preserve correctness we must:
   - Multiply the post-SDPA Thresholding's threshold values by 1/s_V.
   - Widen the threshold dtype annotation to fit the new range.
   - Widen each SDPA's OType from INT8 to a width that fits the 1/s_V-larger
     AV-matmul output (use the smallest integer that fits the rescaled
     threshold max_abs, with the same +1-for-sign formula as the QK
     thresholds — values outside the threshold range simply saturate to the
     highest/lowest bin, which is the intended behavior for outliers).
   - Widen StreamingConcat_hls input/output dtypes and the post-SDPA
     Thresholding's inputDataType / weightDataType attributes to match.

   The earlier comment in this file claimed "spacing 13 in the post-SDPA
   thresholds evidences prior streamliner absorption of s_V". That was
   wrong: the spacing-13 ratio matches 1/post_av_scale only (≈ 95.7×
   streamlined spacing), not the 1/(s_V * post_av_scale) ≈ 686× that would
   be required for s_V absorption. Verified numerically against
   step_specialize_layers.onnx for the trained 70.97% model:
       streamline MT_15 thresh max  = 0.9111
       HW Thresholding_rtl_9 max    = 88
       ratio actual = 88/0.9111     = 96.6
       ratio if Mul_12 only         = 1/0.01045 = 95.7    ← matches
       ratio if both Mul_12 + s_V   = 1/(0.01045*0.139) = 686  ← does NOT match

Guard: the pass only fires when the direct producers of SDPA's Q, K, V
inputs are ElementwiseMul_hls with scalar float initializers. No-op on
graphs where the dequants are already absorbed (Paderborn DummyTransformer).

Constraint: per-head s_V values must be equal (so a single scalar threshold
rescale suffices). For our trained transformer all three heads share
s_V = 0.139 (Brevitas QuantMultiheadAttention with shared per-tensor
scales). If a future model has per-head s_V values that differ, the
threshold expansion to channelwise (NumChannels, numSteps) would be needed
instead — the pass raises in that case so the silent-corruption mode can't
recur.
"""
import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


def _smallest_signed_int_dtype(max_abs_value):
    """Return the narrowest signed-INT DataType whose representable range
    covers +/- max_abs_value. Mirrors the bitwidth formula used for the
    QK threshold widening: ceil(log2(max_abs + 1)) + 1 (the +1 reserves
    the sign bit). Floors at INT9 to match the existing QK code path
    (avoids unnecessary narrowing when the rescaled max_abs is small)."""
    bitwidth = max(9, int(np.ceil(np.log2(max_abs_value + 1))) + 1)
    return DataType[f"INT{bitwidth}"]


class AbsorbDequantIntoSDPA(Transformation):
    def __init__(self):
        super().__init__()
        # Accumulates across the (model.transform-driven) repeated apply calls
        # so the change log survives the idempotent second call that returns
        # (model, False).
        self.changes = []

    def apply(self, model: ModelWrapper):
        graph = model.graph
        modified = False

        sdpa_nodes = [n for n in graph.node
                      if n.op_type == "ScaledDotProductAttention_hls"]
        if not sdpa_nodes:
            return model, False

        producer = {o: n for n in graph.node for o in n.output}
        consumers = {}
        for n in graph.node:
            for inp in n.input:
                consumers.setdefault(inp, []).append(n)

        nodes_to_delete = []
        orphan_inits = []
        # Collected for the post-SDPA V-side compensation (step 6 above).
        sdpa_output_tensors = []
        s_V_values = []

        for sdpa in sdpa_nodes:
            if len(sdpa.input) < 4:
                continue

            q_in, k_in, v_in = sdpa.input[0], sdpa.input[1], sdpa.input[2]
            q_mul = producer.get(q_in)
            k_mul = producer.get(k_in)
            v_mul = producer.get(v_in)

            if not all(m is not None and m.op_type == "ElementwiseMul_hls"
                       for m in (q_mul, k_mul, v_mul)):
                continue

            s_Q = model.get_initializer(q_mul.input[1])
            s_K = model.get_initializer(k_mul.input[1])
            s_V = model.get_initializer(v_mul.input[1])

            if any(s is None for s in (s_Q, s_K, s_V)):
                continue
            if not all(s.size == 1 for s in (s_Q, s_K, s_V)):
                continue

            s_Q_val = float(s_Q.item())
            s_K_val = float(s_K.item())
            s_V_val = float(s_V.item())
            qk_scale = s_Q_val * s_K_val

            sdpa.input[0] = q_mul.input[0]
            sdpa.input[1] = k_mul.input[0]
            sdpa.input[2] = v_mul.input[0]

            for a in sdpa.attribute:
                if a.name in ("QType", "KType", "VType"):
                    a.s = b"INT4"

            qk_thr_name = sdpa.input[3]
            qk_thr = model.get_initializer(qk_thr_name)
            if qk_thr is None:
                continue
            old_qk_range = (float(qk_thr.min()), float(qk_thr.max()))
            new_qk_thr = (qk_thr / qk_scale).astype(qk_thr.dtype)
            model.set_initializer(qk_thr_name, new_qk_thr)

            old_qk_dt = model.get_tensor_datatype(qk_thr_name)
            qk_max_abs = float(max(abs(new_qk_thr.min()),
                                   abs(new_qk_thr.max())))
            new_qk_dt = _smallest_signed_int_dtype(qk_max_abs)
            model.set_tensor_datatype(qk_thr_name, new_qk_dt)

            nodes_to_delete.extend([q_mul, k_mul, v_mul])
            orphan_inits.extend([q_mul.input[1], k_mul.input[1], v_mul.input[1]])

            sdpa_output_tensors.append(sdpa.output[0])
            s_V_values.append(s_V_val)

            self.changes.append({
                "sdpa": sdpa.name,
                "deleted_muls": [q_mul.name, k_mul.name, v_mul.name],
                "scales": {"s_Q": s_Q_val, "s_K": s_K_val, "s_V": s_V_val,
                           "s_Q*s_K": qk_scale},
                "qk_thr_tensor": qk_thr_name,
                "qk_thr_old_range": old_qk_range,
                "qk_thr_new_range": (float(new_qk_thr.min()),
                                     float(new_qk_thr.max())),
                "qk_thr_dtype": f"{old_qk_dt.name} -> {new_qk_dt.name}",
            })
            modified = True

        for m in nodes_to_delete:
            if m in graph.node:
                graph.node.remove(m)

        for init_name in orphan_inits:
            init = next((i for i in graph.initializer
                         if i.name == init_name), None)
            if init is not None:
                graph.initializer.remove(init)

        # ---- Post-SDPA V-side compensation (step 6) ----
        if not sdpa_output_tensors:
            return model, modified

        # Per-head s_V values must agree so a single scalar threshold rescale
        # is sufficient. Otherwise the post-SDPA Thresholding would need to
        # become per-channel (numChannels, numSteps), which is a larger
        # surgery — refuse rather than corrupt silently.
        if not all(np.isclose(s, s_V_values[0], rtol=1e-6, atol=1e-9)
                   for s in s_V_values):
            raise RuntimeError(
                "AbsorbDequantIntoSDPA: per-head s_V values differ "
                f"({s_V_values}); per-channel threshold rescale not "
                "implemented. Either equalize s_V across heads in training "
                "or extend this pass to expand thresholds channelwise.")
        s_V_shared = s_V_values[0]
        v_inv_scale = 1.0 / s_V_shared

        # Walk SDPA.out -> consumer (StreamingConcat_hls) -> consumer
        # (Thresholding_rtl) and collect the post-SDPA Thresholding nodes.
        # All SDPAs typically converge on a single shared Concat which feeds a
        # single Thresholding; dedupe just in case.
        post_sdpa_thr_nodes = []
        post_sdpa_thr_seen = set()
        affected_concats = []
        affected_concats_seen = set()
        for sdpa_out in sdpa_output_tensors:
            for c in consumers.get(sdpa_out, []):
                if c.op_type != "StreamingConcat_hls":
                    continue
                if id(c) not in affected_concats_seen:
                    affected_concats.append(c)
                    affected_concats_seen.add(id(c))
                concat_out = c.output[0]
                for cc in consumers.get(concat_out, []):
                    if cc.op_type not in ("Thresholding_rtl", "Thresholding_hls",
                                          "Thresholding"):
                        continue
                    if id(cc) not in post_sdpa_thr_seen:
                        post_sdpa_thr_nodes.append(cc)
                        post_sdpa_thr_seen.add(id(cc))

        if not post_sdpa_thr_nodes:
            # Nothing to compensate. Leave SDPA OType as-is (INT8) to avoid
            # destabilizing downstream, and report the gap loudly so the
            # board accuracy can't silently regress.
            self.changes.append({
                "warning": ("post-SDPA Thresholding not found via "
                            "SDPA->StreamingConcat->Thresholding; V-side "
                            "compensation skipped — board accuracy will be "
                            "wrong. Inspect graph topology."),
                "sdpa_outputs": sdpa_output_tensors,
            })
            return model, modified

        for thr in post_sdpa_thr_nodes:
            thr_param_name = thr.input[1]
            thr_old = model.get_initializer(thr_param_name)
            if thr_old is None:
                continue
            # Round to integers BEFORE casting back to the original dtype.
            # The input to a Thresholding op (post-rescale) is integer-valued
            # (SDPA produces an integer accumulator), and FINN's
            # MinimizeAccumulatorWidth in step_minimize_bit_width requires
            # threshold values to be representable in the integer dtype it
            # picks for the threshold tensor (np.vectorize(tdt.allowed) check
            # in finn/custom_op/fpgadataflow/thresholding.py:176). Without
            # rounding here we'd hit "Thresholds can't be expressed with type
            # INTn" because non-integer floats are not "allowed" values of an
            # integer DataType.
            thr_new = np.round(thr_old * v_inv_scale).astype(thr_old.dtype)
            model.set_initializer(thr_param_name, thr_new)

            old_thr_dt = model.get_tensor_datatype(thr_param_name)
            thr_max_abs = float(max(abs(thr_new.min()), abs(thr_new.max())))
            new_thr_dt = _smallest_signed_int_dtype(thr_max_abs)
            model.set_tensor_datatype(thr_param_name, new_thr_dt)

            # Update Thresholding attributes that mirror the dtype annotations.
            # The HLS/RTL templates read these attribute strings (not the tensor
            # annotation) at codegen time, so both must be kept in sync.
            for a in thr.attribute:
                if a.name == "weightDataType":
                    a.s = new_thr_dt.name.encode()
                elif a.name == "inputDataType":
                    a.s = new_thr_dt.name.encode()

            # The Thresholding's input tensor will also need its datatype
            # widened — handled in the SDPA-output / Concat-output loop below
            # since that tensor IS the StreamingConcat output.

            self.changes.append({
                "post_sdpa_threshold": thr.name,
                "thr_tensor": thr_param_name,
                "thr_old_range": (float(thr_old.min()), float(thr_old.max())),
                "thr_new_range": (float(thr_new.min()), float(thr_new.max())),
                "thr_dtype": f"{old_thr_dt.name} -> {new_thr_dt.name}",
                "rescale_factor": v_inv_scale,
            })

        # Pick the SDPA OType width to match the post-SDPA threshold dtype.
        # Values outside this range will saturate to the extreme bins of the
        # Thresholding — the correct behavior for outliers.
        new_otype = new_thr_dt  # last threshold processed; all SDPAs share scale
        new_otype_name = new_otype.name
        new_otype_bytes = new_otype_name.encode()

        # Update each SDPA's OType attribute and output tensor annotation.
        for sdpa in sdpa_nodes:
            for a in sdpa.attribute:
                if a.name == "OType":
                    a.s = new_otype_bytes
            model.set_tensor_datatype(sdpa.output[0], new_otype)

        # Update each StreamingConcat: inputDataTypes attribute and output
        # tensor annotation. The HLS template carries arbitrary bit-widths
        # (a transparent bus), so widening here is safe.
        for c in affected_concats:
            for a in c.attribute:
                if a.name == "inputDataTypes":
                    a.strings[:] = [new_otype_bytes for _ in a.strings]
            model.set_tensor_datatype(c.output[0], new_otype)

        self.changes.append({
            "sdpa_otype": f"INT8 -> {new_otype_name}",
            "v_inv_scale": v_inv_scale,
            "s_V_shared": s_V_shared,
            "concats_widened": [c.name for c in affected_concats],
            "thresholdings_widened": [t.name for t in post_sdpa_thr_nodes],
        })

        return model, modified
