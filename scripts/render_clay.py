"""
Clay renders of a mesh with Blender's Workbench engine (grey studio shading + cavity), the way
a sculpt is reviewed in ZBrush. Shows geometry only: no textures and no normal maps, so what
you see is detail that is actually in the mesh.

  python3 scripts/render_clay.py --mesh out/head_mesh_detail.obj --out out/eval/clay.png \
      [--focus X Y Z --focus_size 0.08]      # FLAME coordinates (metres) for a close-up

Runs Blender headless (BLENDER env var, or `blender` on PATH, or /opt/homebrew/bin/blender).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BLENDER_SCRIPT = r'''
import bpy, sys, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
mesh, out, res = argv[0], argv[1], int(argv[2])
yaw = float(argv[3]); fx, fy, fz, fsize = map(float, argv[4:8])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=mesh)          # Y-up OBJ -> Blender Z-up: FLAME (x, y, z) -> (x, -z, y)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
bpy.ops.object.shade_smooth()
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading
sh.light = "STUDIO"
sh.color_type = "SINGLE"
sh.single_color = (0.62, 0.62, 0.62)
sh.show_cavity = True
sh.cavity_type = "BOTH"
sh.curvature_ridge_factor = 0.6
sh.curvature_valley_factor = 1.0
sh.show_specular_highlight = True
sc.render.resolution_x = sc.render.resolution_y = res
sc.render.film_transparent = False
world = bpy.data.worlds.new("w"); sc.world = world
sc.display.shading.background_type = "VIEWPORT"
sc.display.shading.background_color = (0.2, 0.2, 0.21)

bb = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
if fsize > 0:
    center = mathutils.Vector((fx, -fz, fy)); size = fsize
else:
    center = sum(bb, mathutils.Vector()) / 8.0
    size = max((max(v[i] for v in bb) - min(v[i] for v in bb)) for i in range(3)) * 0.85
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 135; cam_data.sensor_width = 36
cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
dist = size / 36.0 * 135 * 1.05
a = math.radians(yaw)
cam.location = center + mathutils.Vector((math.sin(a) * dist, -math.cos(a) * dist, 0.0))
direction = center - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
cam_data.clip_start = dist * 0.1; cam_data.clip_end = dist * 10
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
'''


def blender_bin() -> str:
    for c in (os.environ.get("BLENDER"), shutil.which("blender"), "/opt/homebrew/bin/blender",
              "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("Blender not found. Set BLENDER=/path/to/blender.")


def render_clay(mesh: str, out: str, res: int = 1024, yaw: float = 0.0, focus=None, focus_size: float = 0.0) -> str:
    fx, fy, fz = focus if focus is not None else (0.0, 0.0, 0.0)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(BLENDER_SCRIPT)
        script = fh.name
    cmd = [blender_bin(), "-b", "--factory-startup", "-P", script, "--", str(Path(mesh).resolve()),
           str(Path(out).resolve()), str(res), str(yaw), str(fx), str(fy), str(fz), str(focus_size)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(script)
    if r.returncode != 0 or not Path(out).exists():
        raise RuntimeError(f"Blender clay render failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--focus", type=float, nargs=3, default=None)
    ap.add_argument("--focus_size", type=float, default=0.0)
    a = ap.parse_args()
    print(render_clay(a.mesh, a.out, a.res, a.yaw, a.focus, a.focus_size))
    sys.exit(0)
