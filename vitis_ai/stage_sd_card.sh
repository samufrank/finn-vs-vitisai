#!/bin/bash
# Stages everything that needs to land on the PetaLinux SD card for the
# Gate 1 board session (DPU transformer + size-sweep models + scripts +
# RadioML eval data).
#
# This script does NOT mount the SD card. It only builds a host-side
# staging tree that mirrors the target structure under /home/petalinux/,
# then prints the manual mount + rsync + chown + umount commands you
# should run by hand once the card is plugged in.
#
# Usage:
#   bash vitis_ai/stage_sd_card.sh [staging_dir]
# Default staging_dir: /tmp/sd_stage_petalinux

set -e
set -o pipefail

REPO=/home/samu/dev/CEN571-final/finn-vs-vitisai
STAGE="${1:-/tmp/sd_stage_petalinux}"

if [ -e "$STAGE" ]; then
    echo "ERROR: staging dir already exists: $STAGE"
    echo "  remove it first or pass a different path: bash $0 /path/to/staging"
    exit 1
fi

echo "========================================================================"
echo "  Staging SD-card payload"
echo "    repo:    $REPO"
echo "    staging: $STAGE"
echo "========================================================================"

mkdir -p "$STAGE/home/petalinux/models/dpu/transformer_radioml"
mkdir -p "$STAGE/home/petalinux/models/dpu/baseline"
mkdir -p "$STAGE/home/petalinux/data"
mkdir -p "$STAGE/home/petalinux/results"

# --- 1. Transformer xmodel + metadata -------------------------------------
echo ""
echo "[1/5] Transformer xmodel + subgraph summary"
cp -v "$REPO/vitis_ai/compiled_transformer_radioml/transformer_radioml.xmodel" \
      "$STAGE/home/petalinux/models/dpu/transformer_radioml/"
cp -v "$REPO/vitis_ai/compiled_transformer_radioml/meta.json" \
      "$STAGE/home/petalinux/models/dpu/transformer_radioml/"
cp -v "$REPO/vitis_ai/compiled_transformer_radioml/md5sum.txt" \
      "$STAGE/home/petalinux/models/dpu/transformer_radioml/"
cp -v "$REPO/vitis_ai/subgraph_summary_transformer.json" \
      "$STAGE/home/petalinux/models/dpu/transformer_radioml/"

# --- 2. 12 size-sweep dirs ------------------------------------------------
echo ""
echo "[2/5] Size-sweep DPU models (12 dirs)"
for d in "$REPO"/vitis_ai/compiled/*/; do
    name=$(basename "$d")
    target="$STAGE/home/petalinux/models/dpu/$name"
    mkdir -p "$target"
    cp -rv "$d"* "$target/" | head -10    # head: keep stdout sane
done

# --- 3. Tiny baselines (separate dir to keep them distinct from sweeps) ---
echo ""
echo "[3/5] Baseline tiny models (already on board, included for regression)"
cp -v "$REPO/vitis_ai/zu3_b512/compiled/mlp_mnist_tiny.xmodel" \
      "$STAGE/home/petalinux/models/dpu/baseline/"
cp -v "$REPO/vitis_ai/zu3_b512/compiled/cnn_mnist_tiny.xmodel" \
      "$STAGE/home/petalinux/models/dpu/baseline/"
cp -v "$REPO/vitis_ai/zu3_b512/compiled/meta.json" \
      "$STAGE/home/petalinux/models/dpu/baseline/" 2>/dev/null || true
cp -v "$REPO/vitis_ai/zu3_b512/compiled/md5sum.txt" \
      "$STAGE/home/petalinux/models/dpu/baseline/" 2>/dev/null || true

# --- 4. RadioML eval data --------------------------------------------------
echo ""
echo "[4/5] RadioML eval data (large, ~1.4 GB)"
NPZ="$REPO/data/radioml2018_eval_snr_filtered.npz"
NPZ_SIZE=$(stat -c%s "$NPZ")
echo "  source size: $(numfmt --to=iec $NPZ_SIZE) ($NPZ_SIZE bytes)"
cp -v "$NPZ" "$STAGE/home/petalinux/data/"

# --- 5. Scripts -----------------------------------------------------------
echo ""
echo "[5/5] Scripts"
cp -v "$REPO/board/benchmark.py"             "$STAGE/home/petalinux/"
cp -v "$REPO/board/probe_dpu_transformer.py" "$STAGE/home/petalinux/"
cp -v "$REPO/board/profile_dpu_subgraphs.py" "$STAGE/home/petalinux/"

# --- Summary --------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  Staging complete."
echo "  Total staged size:"
du -sh "$STAGE"
echo ""
echo "  Layout (top 3 levels):"
find "$STAGE/home/petalinux" -maxdepth 4 -type d | sort | sed "s|$STAGE||"
echo ""
echo "  Files at /home/petalinux:"
ls -la "$STAGE/home/petalinux/" | grep -v '^d' | grep -v '^total'
echo "========================================================================"

# --- Manual SD-card workflow ----------------------------------------------
cat <<'INSTRUCTIONS'

==========================================================================
  Manual SD-card mount + copy workflow (host-side, run by hand)
==========================================================================

1. Power off the board, pull the PetaLinux SD card, plug it into this host.

2. Identify the device — DO NOT GUESS:

     lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT
     # find the SD: you want partition 2 (rootfs, ext4) of the card.
     # typical names: /dev/sdX2 where X is the new letter that just appeared.

3. Mount partition 2 read-write:

     sudo mount /dev/sdX2 /mnt
     ls /mnt/home/petalinux              # SANITY CHECK: must show petalinux home

4. Copy the staged tree (use rsync; faster than cp for the 1.4 GB npz,
   and shows a progress bar):

     sudo rsync -av --progress \
         STAGING_DIR/home/petalinux/ \
         /mnt/home/petalinux/

   (If rsync isn't available, tar/cp work too; rsync just gives you
    progress + skips unchanged files on re-runs.)

5. Fix ownership — files copied as root must be owned by petalinux uid:

     sudo chown -R 1000:1000 /mnt/home/petalinux/models/
     sudo chown -R 1000:1000 /mnt/home/petalinux/data/
     sudo chown 1000:1000 /mnt/home/petalinux/benchmark.py
     sudo chown 1000:1000 /mnt/home/petalinux/probe_dpu_transformer.py
     sudo chown 1000:1000 /mnt/home/petalinux/profile_dpu_subgraphs.py

6. Verify integrity (especially the big npz):

     ls -la /mnt/home/petalinux/data/radioml2018_eval_snr_filtered.npz
     md5sum /mnt/home/petalinux/models/dpu/transformer_radioml/transformer_radioml.xmodel
     # (cross-check against the md5sum.txt in the same dir)

7. Unmount cleanly:

     sudo umount /mnt
     sync

8. Re-insert SD card into the board, power on, log in over serial console:

     screen /dev/ttyUSB1 115200
     # user: petalinux  pass: zu3
     ~/boot_setup.sh
     sudo date -s "$(date -u +'%Y-%m-%d %H:%M:%S')"   # clock sync
     ls /home/petalinux/models/dpu/                     # verify files there

INSTRUCTIONS

# Patch the printed instructions with the actual staging dir path
echo "(replace STAGING_DIR with: $STAGE)"
