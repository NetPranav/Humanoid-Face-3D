#!/usr/bin/env bash
# Automated headless Blender FBX packaging script
set -euo pipefail

BLENDER_BIN="${1:-blender}"
if ! command -v "$BLENDER_BIN" &> /dev/null; then
    echo "[Error] Blender not found. Please install Blender or provide path as argument."
    exit 1
fi

"$BLENDER_BIN" --background --python "/Users/pranav/Project Folder/3d Model /scripts/blender_export.py" -- \
    --mesh "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/head_mesh_neutral.obj" \
    --blendshapes "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/blendshapes_arkit52.json" \
    --output "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/head_mesh_ue5_livelink.fbx" \
    --armature "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/armature_rig.json" \
    --lods "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/lod_manifest.json" \
    --normal_map "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/film_normal.png" \
    --displacement_map "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/film_displacement_16bit.png" \
    --render_preview "/Users/pranav/Project Folder/3d Model /outputs/online_test/elon/preview_render.png"
