#!/usr/bin/env bash
# Downloads and unpacks portable headless Blender 4.1 into /tmp/blender
# Optimized for Kaggle CPU/GPU environments (stores in /tmp to preserve 19.5GB quota)
set -euo pipefail

BLENDER_DIR="/tmp/blender"
mkdir -p "$BLENDER_DIR"

if [ -f "$BLENDER_DIR/blender" ]; then
    echo "[Notice] Blender already installed at $BLENDER_DIR/blender"
    "$BLENDER_DIR/blender" --version
    exit 0
fi

echo "[Setup] Downloading portable Blender 4.1.0 for Linux x64..."
wget -q -O /tmp/blender-4.1.0-linux-x64.tar.xz https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz

echo "[Setup] Extracting to $BLENDER_DIR (~500MB headroom in /tmp)..."
tar -xf /tmp/blender-4.1.0-linux-x64.tar.xz -C "$BLENDER_DIR" --strip-components=1
rm -f /tmp/blender-4.1.0-linux-x64.tar.xz

if [ -w /usr/local/bin ]; then
    ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender
    echo "[Setup] Linked $BLENDER_DIR/blender to /usr/local/bin/blender"
fi

echo "[Setup] Headless Blender ready for UE5 Live Link FBX packaging:"
"$BLENDER_DIR/blender" --version
