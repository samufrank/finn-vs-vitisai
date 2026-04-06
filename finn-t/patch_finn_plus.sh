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
sed -i 's/        model = model.transform(InferShapes())  # noqa: Shadows model/        try:\n            model = model.transform(InferShapes())  # noqa: Shadows model\n        except Exception:\n            pass  # ONNX shape inference fails on FINN custom ops/' \
  "$SITE/finn/transformation/fpgadataflow/attention.py"
echo "  Patched attention.py (InferShapes x2)"

# Patch 2: InferShapes crash in transformer_adhoc.py
ADHOC="$SITE/finn/builder/custom_step_library/transformer_adhoc.py"
sed -i '/    model = model.transform(InferShapes())/c\    try:\n        model = model.transform(InferShapes())\n    except Exception:\n        pass' "$ADHOC"
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
