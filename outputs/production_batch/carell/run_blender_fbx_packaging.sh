#!/usr/bin/env bash
# Automated headless Blender FBX packaging script
set -euo pipefail

BLENDER_BIN="${1:-blender}"
if ! command -v "$BLENDER_BIN" &> /dev/null; then
    echo "[Error] Blender not found. Please install Blender or provide path as argument."
    exit 1
fi

"$BLENDER_BIN" --background --python "/Users/pranav/Project Folder/3d Model /scripts/blender_export.py" -- \
    --mesh "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/head_mesh_neutral.obj" \
    --blendshapes "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/blendshapes_arkit52.json" \
    --output "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/head_mesh_ue5_livelink.fbx" \
    --armature "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/armature_rig.json" \
    --lods "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/lod_manifest.json" \
    --hair_cards "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/facial_hair/facial_hair_cards.obj" \
    --hair_skinning "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/facial_hair/hair_cards_skinning.json" \
    --normal_map "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/film_normal.png" \
    --displacement_map "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/film_displacement_16bit.png" \
    --render_preview "/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/preview_render.png"
