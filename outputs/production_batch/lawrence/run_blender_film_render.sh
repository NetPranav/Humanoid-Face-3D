#!/bin/bash
# Automated Film-Grade Blender Cycles Render Script
blender -b -P '/Users/pranav/Project Folder/3d Model /scripts/render_blender_film.py' -- \
    --mesh '/Users/pranav/Project Folder/3d Model /outputs/production_batch/lawrence/face_lod0.obj' \
    --output '/Users/pranav/Project Folder/3d Model /outputs/production_batch/lawrence/film_render_cycles.png' \
    --save_blend '/Users/pranav/Project Folder/3d Model /outputs/production_batch/lawrence/studio_scene.blend' \
    --samples '128' \
    --resolution '2048' '2048' \
    --device 'AUTO' \
    --textures_dir '/Users/pranav/Project Folder/3d Model /outputs/production_batch/lawrence' \

