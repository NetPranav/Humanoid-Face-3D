#!/bin/bash
# Automated Film-Grade Blender Cycles Render Script
blender -b -P '/Users/pranav/Project Folder/3d Model /scripts/render_blender_film.py' -- \
    --mesh '/Users/pranav/Project Folder/3d Model /outputs/production_batch/connelly/face_lod0.obj' \
    --output '/Users/pranav/Project Folder/3d Model /outputs/production_batch/connelly/film_render_cycles.png' \
    --save_blend '/Users/pranav/Project Folder/3d Model /outputs/production_batch/connelly/studio_scene.blend' \
    --samples '128' \
    --resolution '2048' '2048' \
    --device 'AUTO' \
    --textures_dir '/Users/pranav/Project Folder/3d Model /outputs/production_batch/connelly' \

