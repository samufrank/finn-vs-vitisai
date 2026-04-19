#!/bin/bash
# Patches finn-plus 1.4.0 for transformer support.
# Run with finn-t venv activated.
#
# These patches fix bugs in the transformer code path only.
# CNN workflows are unaffected. See finn_t_setup_guide.md for details.
set -euo pipefail

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "Patching finn-plus at: $SITE"

# Patch 1: InferShapes crash on FINN custom op domain (attention.py)
# Previously used `sed -i 's/<8-space indent>model = .../try:\n<12-space>model
# = .../'` but sed regex is not line-anchored — the 8-space pattern matches
# inside the already-12-space-indented post-patch line, so a second run wraps
# an already-wrapped try: and produces IndentationError. Use Python with
# substring guards: skip if the replacement target already shows up inside a
# try: block.
python3 - "$SITE/finn/transformation/fpgadataflow/attention.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
src = path.read_text()
# Anchor on preceding newline so an 8-space pristine line is not detected
# inside a 12-space post-patch line (the bug we are fixing).
OLD = '\n        model = model.transform(InferShapes())  # noqa: Shadows model\n'
NEW = (
    '\n        try:\n'
    '            model = model.transform(InferShapes())  # noqa: Shadows model\n'
    '        except Exception:\n'
    '            pass  # ONNX shape inference fails on FINN custom ops\n'
)
if OLD in src:
    src = src.replace(OLD, NEW)  # all occurrences (file has 2)
    path.write_text(src)
    print("  applied")
else:
    print("  pristine pattern not found (already patched); no-op")
PY
echo "  Patched attention.py (InferShapes x2)"

# Patch 2: InferShapes crash in transformer_adhoc.py
# Same idempotence bug as Patch 1: the original sed `c\` address was an
# unanchored regex that also matched the 8-space-indented post-patch line
# and wrapped again. Rewrite with Python substring guard.
ADHOC="$SITE/finn/builder/custom_step_library/transformer_adhoc.py"
python3 - "$ADHOC" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
src = path.read_text()
# Anchor on preceding newline so a 4-space pristine line is not detected
# inside an 8-space post-patch line (the bug we are fixing).
OLD = '\n    model = model.transform(InferShapes())\n'
NEW = (
    '\n    try:\n'
    '        model = model.transform(InferShapes())\n'
    '    except Exception:\n'
    '        pass\n'
)
if OLD in src:
    src = src.replace(OLD, NEW, 1)
    path.write_text(src)
    print("  applied")
else:
    print("  pristine pattern not found (already patched); no-op")
PY
echo "  Patched transformer_adhoc.py (InferShapes)"

# Patch 3: numpy.int64 rejected by set_nodeattr in transformer_adhoc.py
sed -i 's/inst.set_nodeattr("EmbFold", math.gcd(qkdim, vdim))/inst.set_nodeattr("EmbFold", int(math.gcd(qkdim, vdim)))/' "$ADHOC"
sed -i 's/inst.set_nodeattr("SeqFold", kvlen)/inst.set_nodeattr("SeqFold", int(kvlen))/' "$ADHOC"
sed -i 's/inst.set_nodeattr("EmbFold", fold)/inst.set_nodeattr("EmbFold", int(fold))/g' "$ADHOC"
sed -i 's/inst.set_nodeattr("SeqFold", fold)/inst.set_nodeattr("SeqFold", int(fold))/g' "$ADHOC"
echo "  Patched transformer_adhoc.py (numpy int casts)"

# Patch 4: PosixPath + string concatenation in transformer_adhoc.py
sed -i 's/cfg\.output_dir + "/str(cfg.output_dir) + "/g' "$ADHOC"
echo "  Patched transformer_adhoc.py (PosixPath)"

# Patch 5: numpy.int64 in convert_to_hw_layers.py
sed -i 's/inst.set_nodeattr("axes", list(axes))/inst.set_nodeattr("axes", [int(x) for x in axes])/' \
  "$SITE/finn/transformation/fpgadataflow/convert_to_hw_layers.py"
echo "  Patched convert_to_hw_layers.py (numpy int cast)"

# Patch 6: PosixPath + string in build_dataflow_steps.py
sed -i 's/cfg\.output_dir + "/str(cfg.output_dir) + "/g' \
  "$SITE/finn/builder/build_dataflow_steps.py"
echo "  Patched build_dataflow_steps.py (PosixPath)"

echo ""
echo "All patches applied."

# --- Patch 7: gate Alveo-only deployment package code on shell_flow_type ---
# step_deployment_package unconditionally runs Alveo-only path rewriting
# (looking for finn-accel.xclbin and acceleratorconfig.json) when CPP_DRIVER
# is in generate_outputs. This crashes vivado_zynq builds where neither file
# exists. Gate it on shell_flow_type.
sed -i \
  's|if DataflowOutputType.CPP_DRIVER in cfg.generate_outputs:|if DataflowOutputType.CPP_DRIVER in cfg.generate_outputs and cfg.shell_flow_type == "vivado_alveo":|' \
  "$SITE/finn/builder/build_dataflow_steps.py"
echo "  Patched build_dataflow_steps.py (Alveo-only deployment gate)"

# --- Patch 8: MoveTransposePastEltwise - scalar-skip guard (reorder.py) ---
# Bug: StreamlinePlus oscillates infinitely between MoveTransposePastEltwise
# and MoveScalarLinearPastInvariants when an Add/Mul has a scalar or
# effectively-scalar (all-ones-shape) initializer. The two passes are
# exact inverses on that pattern; neither converges, and only Patch 10's
# iteration cap prevents a hang. Session 17, "Bug 1 / streamliner
# oscillation" investigation.
# Fix: skip scalar-shaped initializers in MoveTransposePastEltwise so
# MoveScalarLinearPastInvariants owns the pattern exclusively.
#
# Implementation note: this is a 13-line block with indent changes. sed
# handles this only via -z and heavy escaping; python heredoc string
# replace is cleaner and stays inline. Idempotent via `if OLD in src`:
# a second run against an already-patched file no-ops silently, matching
# the de-facto no-op-on-rerun behavior of patches 1-7 (session 13 note:
# 1-7's idempotence comes from pristine-pattern-absence after first run,
# not from an explicit guard; same net effect here).
python3 - "$SITE" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1]) / "finn/transformation/streamline/reorder.py"
src = path.read_text()
OLD = '''                        # Do not transpose scalar or effectively scalar
                        # initializers
                        # fmt: off
                        if not (value.shape is None or all(
                                x == 1 for x in value.shape)):
                            # fmt: on
                            # Transpose the initializer and re-insert into the
                            # model
                            # fmt: off
                            model.set_initializer(
                                a, value.transpose(inverse_perm)
                            )
                            # fmt: on
'''
NEW = '''                        # Scalar/effectively-scalar initializers are the canonical
                        # responsibility of MoveScalarLinearPastInvariants; rewiring
                        # them here reverses that pass and produces an infinite
                        # oscillation inside StreamlinePlus.
                        # fmt: off
                        if value.shape is None or all(
                                x == 1 for x in value.shape):
                            continue
                        # fmt: on
                        # Transpose the initializer and re-insert into the
                        # model
                        # fmt: off
                        model.set_initializer(
                            a, value.transpose(inverse_perm)
                        )
                        # fmt: on
'''
if OLD in src:
    path.write_text(src.replace(OLD, NEW, 1))
    print("  applied")
else:
    print("  pristine pattern not found (likely already patched); no-op")
PY
echo "  Patched reorder.py (MoveTransposePastEltwise scalar-skip guard)"

# --- Patch 9: MoveTransposePastEltwise - rank-deficient guard (reorder.py) ---
# Bug: after a Squeeze strips a degenerate dim, an Add initializer's rank
# can be lower than the transpose permutation rank. value.transpose(perm)
# then raises "axes don't match array" inside MoveTransposePastEltwise and
# aborts the build. Session 14.
# Fix: if inverse_perm is set and value.ndim < len(inverse_perm), skip
# this initializer - it cannot be validly transposed by perm.
#
# ORDERING: this patch depends on Patch 8 having run first. It anchors on
# the post-#8 form of the scalar-skip block (24-space indented `if value.
# shape is None`). If Patch 8 is reordered after or removed, Patch 9's
# anchor will not match and the patch will silently no-op.
# Idempotence: re-running against an already-patched file no-ops (OLD not
# found), same convention as patches 1-7.
python3 - "$SITE" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1]) / "finn/transformation/streamline/reorder.py"
src = path.read_text()
OLD = '''                        if value.shape is None or all(
                                x == 1 for x in value.shape):
                            continue
                        # fmt: on
                        # Transpose the initializer and re-insert into the
'''
NEW = '''                        if value.shape is None or all(
                                x == 1 for x in value.shape):
                            continue
                        # fmt: on
                        # PATCHED: also skip rank-deficient inits
                        if (inverse_perm is not None
                                and value.ndim < len(inverse_perm)):
                            continue
                        # Transpose the initializer and re-insert into the
'''
if OLD in src:
    path.write_text(src.replace(OLD, NEW, 1))
    print("  applied")
else:
    print("  anchor pattern not found (did Patch 8 run? likely already patched); no-op")
PY
echo "  Patched reorder.py (MoveTransposePastEltwise rank-deficient guard)"

# --- Patch 10: ComposedTransformation iteration cap (qonnx/composed.py) ---
# Bug: ComposedTransformation.apply() runs each inner transformation under
# `while True:` until the pass returns graph_modified=False. A pair of
# inverse passes (or a single oscillating pass like pre-Patch-8
# MoveTransposePastEltwise) never converges, and the build hangs
# indefinitely. Session 14.
# Fix: bound the inner loop at 100 iterations. Transformation-level
# non-termination then surfaces as a budget overrun in subsequent passes
# rather than a process hang.
#
# Uniqueness: `while True:` appears exactly once in composed.py (line 45
# of pristine qonnx 1.0.0); the 2-line context `# the graph\n            while
# True:` is unique enough to anchor safely. sed -z matches across the
# newline in a single substitution.
# Idempotence: the anchor line `# the graph` (without the PATCHED suffix)
# no longer appears after first run, so re-invocations no-op. Same
# convention as patches 1-7.
sed -zi 's|            # the graph\n            while True:|            # the graph (PATCHED: max iterations guard)\n            _max_iter = 100\n            for _iter in range(_max_iter):|' \
  "$SITE/qonnx/transformation/composed.py"
echo "  Patched qonnx/composed.py (ComposedTransformation iteration cap)"

# --- Patch 11: set_folding.py numpy.int64 coercion ---
# Bug: SetFolding.apply()'s StreamingConcat_hls/StreamingSplit_hls branch
# iterates common_divisors(...) — which uses np.intersect1d internally and
# yields numpy.int64 scalars — and passes them directly to
# set_nodeattr("SIMD", simd_val). onnx.helper rejects numpy types:
# "Attribute SIMD expects int, got <class 'numpy.int64'>". Same bug exists
# in the LayerNorm_rtl branch. transformer_adhoc.py already wraps
# int(fold) at its own callsites (lines 156/165); these two in
# set_folding.py were missed. Session 18, trained transformer build
# step 12.
#
# Fix: wrap both set_nodeattr("SIMD", simd_val) calls with int(simd_val).
# Idempotence: after patching, the old pattern no longer matches.
python3 - "$SITE/finn/transformation/fpgadataflow/set_folding.py" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()
applied = 0
OLD1 = '                    for simd_val in common_divisors(channels_per_stream):\n                        node_inst.set_nodeattr("SIMD", simd_val)'
NEW1 = ('                    # PATCHED: int(simd_val) — common_divisors yields numpy.int64\n'
        '                    for simd_val in common_divisors(channels_per_stream):\n'
        '                        node_inst.set_nodeattr("SIMD", int(simd_val))')
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1); applied += 1
OLD2 = '                        if dim // simd_val > 12:\n                            node_inst.set_nodeattr("SIMD", simd_val)'
NEW2 = ('                        if dim // simd_val > 12:\n'
        '                            # PATCHED: int(simd_val) for consistency\n'
        '                            node_inst.set_nodeattr("SIMD", int(simd_val))')
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1); applied += 1
if applied:
    path.write_text(src)
    print(f"  applied ({applied}/2 spots)")
else:
    print("  anchor patterns not found; already patched; no-op")
PY
echo "  Patched set_folding.py (numpy.int64 → int for SIMD)"

