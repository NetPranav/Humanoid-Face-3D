"""
Ground-truth test photographs from a textured head scan (Blender Cycles, random-walk skin SSS).

Renders a scan (default: the CC-BY 3.0 Lee Perry-Smith head, data/external/lee_perry_smith)
with a known camera, and writes
    <out>/photo.jpg        8-bit sRGB "photograph" with EXIF focal data (so the normal pipeline path runs)
    <out>/camera.json      exact intrinsics/extrinsics, scene units, light setup
    <out>/albedo.png       base colour seen through the same camera (ground-truth pigment)
Variants (--sss 0 disables subsurface scattering; --yaw/--pitch/--light rotate things) let the
same scan produce many controlled test photos.

  python3 scripts/render_gt_scan.py --out outputs/gt_scan/lps_front --yaw 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLENDER_SCRIPT = r'''
import bpy, sys, json, math, mathutils
a = json.loads(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a["mesh"])
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.object.shade_smooth()

# fine relief: subdivide and displace with the scan's own displacement map
if a["disp"] and a["disp_strength"] > 0:
    sub = obj.modifiers.new("sub", "SUBSURF"); sub.levels = a["subdiv"]; sub.render_levels = a["subdiv"]
    img = bpy.data.images.load(a["disp"]); img.colorspace_settings.name = "Non-Color"
    tex = bpy.data.textures.new("disp", "IMAGE"); tex.image = img
    d = obj.modifiers.new("disp", "DISPLACE"); d.texture = tex; d.texture_coords = "UV"
    d.strength = a["disp_strength"]; d.mid_level = a["disp_mid"]

mat = bpy.data.materials.new("skin"); mat.use_nodes = True
nt = mat.node_tree; nodes, links = nt.nodes, nt.links
bsdf = nodes["Principled BSDF"]
col = nodes.new("ShaderNodeTexImage"); col.image = bpy.data.images.load(a["albedo"])
links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.45
if "Specular IOR Level" in bsdf.inputs: bsdf.inputs["Specular IOR Level"].default_value = 0.35
if a["sss"] > 0:
    bsdf.subsurface_method = "RANDOM_WALK_SKIN" if hasattr(bsdf, "subsurface_method") else bsdf.subsurface_method
    bsdf.inputs["Subsurface Weight"].default_value = a["sss"]
    bsdf.inputs["Subsurface Radius"].default_value = a["sss_radius_rgb"]
    bsdf.inputs["Subsurface Scale"].default_value = a["sss_scale"]
obj.data.materials.clear(); obj.data.materials.append(mat)

sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = a["samples"]; sc.cycles.use_denoising = True
sc.cycles.device = "CPU"
sc.render.resolution_x, sc.render.resolution_y = a["res"]
sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.18, 0.18, 0.2, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = a["ambient"]

C = mathutils.Vector(a["target"])
def add_light(name, energy, az, el, dist, size):
    L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o)
    az, el = math.radians(az), math.radians(el)
    o.location = C + dist * mathutils.Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    o.rotation_euler = (C - o.location).to_track_quat("-Z", "Y").to_euler()
for L in a["lights"]:
    add_light(*L)

cam = bpy.data.cameras.new("cam"); cam.lens = a["lens_mm"]; cam.sensor_width = 36.0; cam.sensor_fit = "HORIZONTAL"
co = bpy.data.objects.new("cam", cam); sc.collection.objects.link(co); sc.camera = co
yaw, pitch = math.radians(a["yaw"]), math.radians(a["pitch"])
co.location = C + a["dist"] * mathutils.Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
co.rotation_euler = (C - co.location).to_track_quat("-Z", "Y").to_euler()
cam.clip_start = 0.1; cam.clip_end = 1000

sc.render.filepath = a["out"] + "/render.png"
bpy.ops.render.render(write_still=True)

# ground-truth albedo: same camera, emission of the base colour only (no lighting, no SSS)
em = nodes.new("ShaderNodeEmission"); links.new(col.outputs["Color"], em.inputs["Color"])
out_node = [n for n in nodes if n.type == "OUTPUT_MATERIAL"][0]
links.new(em.outputs["Emission"], out_node.inputs["Surface"])
for o in list(sc.objects):
    if o.type == "LIGHT": bpy.data.objects.remove(o)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
sc.cycles.samples = 4
sc.render.filepath = a["out"] + "/albedo.png"
bpy.ops.render.render(write_still=True)

M = co.matrix_world.inverted()
json.dump({"world_to_camera_blender": [list(r) for r in M], "lens_mm": a["lens_mm"], "sensor_width_mm": 36.0,
           "res": a["res"], "yaw": a["yaw"], "pitch": a["pitch"], "dist": a["dist"], "target": a["target"]},
          open(a["out"] + "/camera_blender.json", "w"), indent=1)
'''


def blender_bin() -> str:
    import shutil
    for c in (os.environ.get("BLENDER"), shutil.which("blender"), "/opt/homebrew/bin/blender"):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("Blender not found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=str(ROOT / "data/external/lee_perry_smith"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--lens-mm", type=float, default=85.0)
    ap.add_argument("--res", type=int, nargs=2, default=[1600, 2000])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--sss", type=float, default=1.0)
    ap.add_argument("--sss-scale", type=float, default=0.06, help="scene units; LPS units are ~5 cm")
    ap.add_argument("--key-az", type=float, default=-35.0)
    ap.add_argument("--key-el", type=float, default=30.0)
    ap.add_argument("--disp-strength", type=float, default=0.06)
    ap.add_argument("--albedo", default=None, help="override albedo texture (e.g. with added pigment)")
    a = ap.parse_args()
    scan = Path(a.scan)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = [0.0, 0.0, 1.9]          # LPS face centre (scene units, Blender Z-up after glTF import)
    dist = 14.0 * a.lens_mm / 85.0    # keeps the head the same size in frame across lenses
    args = {
        "mesh": str(scan / "LeePerrySmith.glb"), "albedo": a.albedo or str(scan / "Map-COL_blender.png"),
        "disp": str(scan / "Disp_blender.png"), "disp_strength": a.disp_strength,
        "disp_mid": 0.62, "subdiv": 2, "sss": a.sss, "sss_radius_rgb": [1.0, 0.35, 0.2], "sss_scale": a.sss_scale,
        "samples": a.samples, "res": a.res, "ambient": 0.35, "target": target,
        "lights": [["key", 2500.0, a.key_az, a.key_el, 12.0, 4.0], ["fill", 700.0, 50.0, 10.0, 12.0, 6.0],
                   ["rim", 900.0, 160.0, 35.0, 12.0, 3.0]],
        "lens_mm": a.lens_mm, "yaw": a.yaw, "pitch": a.pitch, "dist": dist, "out": str(out),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(BLENDER_SCRIPT)
        script = fh.name
    r = subprocess.run([blender_bin(), "-b", "--factory-startup", "-P", script, "--", json.dumps(args)],
                       capture_output=True, text=True)
    os.unlink(script)
    if r.returncode != 0 or not (out / "render.png").exists():
        raise RuntimeError(f"Blender failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    # JPEG "photo" with EXIF focal data so the real pipeline path is exercised
    from PIL import Image
    im = Image.open(out / "render.png").convert("RGB")
    W = a.res[0]
    exif = Image.Exif()
    exif[0x8769] = {0x920A: float(a.lens_mm), 0xA20E: float(W / 36.0), 0xA210: 4}   # px per mm on a 36 mm sensor
    im.save(out / "photo.jpg", quality=95, exif=exif)
    print(f"rendered {out / 'photo.jpg'}")


if __name__ == "__main__":
    main()
