#!/bin/bash
# Automated Film-Grade Blender Cycles Render Script
blender -b -P '/Users/pranav/Project Folder/3d Model /scripts/render_blender_film.py' -- \
    --mesh '/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/face_lod0.obj' \
    --output '/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/film_render_cycles.png' \
    --save_blend '/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell/studio_scene.blend' \
    --samples '16' \
    --resolution '2048' '2048' \
    --device 'AUTO' \
    --textures_dir '/Users/pranav/Project Folder/3d Model /outputs/production_batch/carell' \

