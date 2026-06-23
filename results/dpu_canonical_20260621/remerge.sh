#!/usr/bin/env bash
# remerge.sh — re-merge FNB58 power for ONE DPU canonical model.
#
# Usage:   ./remerge.sh <model>
#   model ∈ cnn_tiny cnn_small cnn_medium cnn_large cnn_deep_3
#           mlp_tiny mlp_tiny_plus mlp_small mlp_small_plus mlp_medium
#           mlp_large mlp_original mlp_tfc resnet8_cifar10
#
# Looks the model up in the embedded re-merge map (kept in sync with MANIFEST.md),
# then re-runs merge_power.py against that model's OWN raw board JSON + CSV with
# --clock-offset 0. So a re-merge can never pair the wrong files.
#
# It OVERWRITES merged/<file> (the point of a re-merge is to fix a bad merge).
# The raw JSONs and CSVs in raw/ are immutable inputs — never edited.
# Run from this directory (results/dpu_canonical_20260621/).
#
# This is a helper for FUTURE recovery; it was NOT run during the organize pass.
set -euo pipefail

SESS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SESS/../.." && pwd)"
MERGE="$REPO/board/merge_power.py"
OFFSET=0

# model -> "raw_json_basename csv_basename"   (matches MANIFEST.md)
declare -A MAP=(
  [cnn_tiny]="cnn_tiny_mnist_b1_20260621_014924.json cnn_tiny_power.csv"
  [cnn_small]="cnn_small_mnist_b1_20260621_015926.json dpu_canonical_power.csv"
  [cnn_medium]="cnn_medium_mnist_b1_20260621_020004.json dpu_canonical_power.csv"
  [cnn_large]="cnn_large_mnist_b1_20260621_020043.json dpu_canonical_power.csv"
  [cnn_deep_3]="cnn_deep_3_mnist_b1_20260621_020120.json dpu_canonical_power.csv"
  [mlp_tiny]="mlp_tiny_mnist_b1_20260621_020155.json dpu_canonical_power.csv"
  [mlp_tiny_plus]="mlp_tiny_plus_mnist_b1_20260621_020231.json dpu_canonical_power.csv"
  [mlp_small]="mlp_small_mnist_b1_20260621_020307.json dpu_canonical_power.csv"
  [mlp_small_plus]="mlp_small_plus_mnist_b1_20260621_020343.json dpu_canonical_power.csv"
  [mlp_medium]="mlp_medium_mnist_b1_20260621_020421.json dpu_canonical_power.csv"
  [mlp_large]="mlp_large_mnist_b1_20260621_020501.json dpu_canonical_power.csv"
  [mlp_original]="mlp_original_mnist_b1_20260621_020540.json dpu_canonical_power.csv"
  [mlp_tfc]="mlp_tfc_mnist_b1_20260621_020616.json dpu_canonical_power.csv"
  [resnet8_cifar10]="resnet8_cifar10_cifar10_b1_20260621_020700.json dpu_canonical_power.csv"
)

model="${1:-}"
if [[ -z "$model" || -z "${MAP[$model]:-}" ]]; then
  echo "usage: $0 <model>"; echo "models: ${!MAP[*]}"; exit 1
fi
read -r RAWJSON CSV <<<"${MAP[$model]}"

echo "Re-merging $model"
echo "  benchmark : raw/$RAWJSON"
echo "  power     : raw/$CSV"
echo "  output    : merged/$RAWJSON   (clock-offset $OFFSET)"
python3 "$MERGE" \
  --benchmark "$SESS/raw/$RAWJSON" \
  --power     "$SESS/raw/$CSV" \
  --clock-offset "$OFFSET" \
  --output    "$SESS/merged/$RAWJSON"

echo "Done. Verify summary.dynamic_power_w is non-null in merged/$RAWJSON."
