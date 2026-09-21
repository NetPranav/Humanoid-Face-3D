"""
Headless Blender script for game-engine FBX export.
Integrates neutral base mesh, creates ARKit-52 shape keys (blendshapes),
builds a 5-joint skeletal armature with skinning weights, binds facial hair cards,
wires Unreal Engine 5 material slots, and exports FBX assets cleanly structured
for Unreal Engine 5 Live Link.

Invoked via:
  blender --background --python scripts/blender_export.py -- \
      --mesh <mesh.obj> \
      --blendshapes <blendshapes.json> \
      --output <output.fbx> \
      [--armature <armature.json>] \
      [--lods <lod_manifest.json>] \
      [--hair_cards <hair_cards.obj>] \
      [--hair_skinning <hair_skinning.json>] \
      [--normal_map <normal.png>] \
      [--render_preview <preview.png>]
"""
import sys
import os
import argparse
import json
from pathlib import Path


def parse_args():
    # Blender passes everything after '--' to Python
    try:
        args_idx = sys.argv.index('--') + 1
        cli_args = sys.argv[args_idx:]
    except ValueError:
        cli_args = sys.argv[1:]

    # Backwards compatibility check for positional arguments: <mesh.obj> <blendshapes.json> <output.fbx>
    if len(cli_args) >= 3 and not cli_args[0].startswith('-'):
        parser = argparse.ArgumentParser(description="Headless Blender FBX Export")
        parser.add_argument("mesh", help="Path to base neutral OBJ mesh")
        parser.add_argument("blendshapes", help="Path to ARKit-52 blendshapes JSON")
        parser.add_argument("output", help="Path to output FBX file")
        parser.add_argument("--armature", help="Path to armature JSON", default=None)
        parser.add_argument("--lods", help="Path to LOD manifest JSON", default=None)
        parser.add_argument("--hair_cards", help="Path to hair cards OBJ", default=None)
        parser.add_argument("--hair_skinning", help="Path to hair cards skinning JSON", default=None)
        parser.add_argument("--normal_map", help="Path to normal map PNG", default=None)
        parser.add_argument("--displacement_map", help="Path to displacement PNG", default=None)
        parser.add_argument("--render_preview", help="Path to output render preview PNG", default=None)
        return parser.parse_args(cli_args)

    parser = argparse.ArgumentParser(description="Headless Blender FBX Export")
    parser.add_argument("--mesh", required=True, help="Path to base neutral OBJ mesh")
    parser.add_argument("--blendshapes", required=True, help="Path to ARKit-52 blendshapes JSON")
    parser.add_argument("--output", required=True, help="Path to output FBX file")
    parser.add_argument("--armature", help="Path to armature JSON", default=None)
    parser.add_argument("--lods", help="Path to LOD manifest JSON", default=None)
    parser.add_argument("--hair_cards", help="Path to hair cards OBJ", default=None)
    parser.add_argument("--hair_skinning", help="Path to hair cards skinning JSON", default=None)
    parser.add_argument("--normal_map", help="Path to normal map PNG", default=None)
    parser.add_argument("--displacement_map", help="Path to displacement PNG", default=None)
    parser.add_argument("--render_preview", help="Path to output render preview PNG", default=None)
    return parser.parse_args(cli_args)


def create_or_get_material(name: str):
    import bpy
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
    return mat


def setup_skin_material(mat, normal_map_path: str = None, disp_map_path: str = None):
    import bpy
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return

    # Base skin tone
    bsdf.inputs["Base Color"].default_value = (0.75, 0.62, 0.53, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.45
    if "Subsurface" in bsdf.inputs:
        bsdf.inputs["Subsurface"].default_value = 0.15

    if normal_map_path and Path(normal_map_path).exists():
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(str(Path(normal_map_path).resolve()))
        tex_node.image.colorspace_settings.name = "Non-Color"

        normal_node = nodes.new("ShaderNodeNormalMap")
        links.new(tex_node.outputs["Color"], normal_node.inputs["Color"])
        links.new(normal_node.outputs["Normal"], bsdf.inputs["Normal"])


def setup_hair_card_material(mat, alpha_map_path: str = None, normal_map_path: str = None):
    import bpy
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return

    # Dark facial hair base color
    bsdf.inputs["Base Color"].default_value = (0.05, 0.04, 0.03, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.65

    # Alpha mask setup
    if alpha_map_path and Path(alpha_map_path).exists():
        alpha_node = nodes.new("ShaderNodeTexImage")
        alpha_node.image = bpy.data.images.load(str(Path(alpha_map_path).resolve()))
        alpha_node.image.colorspace_settings.name = "Non-Color"
        links.new(alpha_node.outputs["Color"], bsdf.inputs["Alpha"])

        if hasattr(mat, "blend_method"):
            mat.blend_method = "CLIP"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "CLIP"

    # Tangent normal map
    if normal_map_path and Path(normal_map_path).exists():
        norm_tex = nodes.new("ShaderNodeTexImage")
        norm_tex.image = bpy.data.images.load(str(Path(normal_map_path).resolve()))
        norm_tex.image.colorspace_settings.name = "Non-Color"

        normal_node = nodes.new("ShaderNodeNormalMap")
        links.new(norm_tex.outputs["Color"], normal_node.inputs["Color"])
        links.new(normal_node.outputs["Normal"], bsdf.inputs["Normal"])


def build_armature(armature_path: str):
    import bpy
    from mathutils import Vector

    arm_file = Path(armature_path)
    if not arm_file.exists():
        return None, {}, []

    print(f"[Blender Export] Constructing 5-joint armature from: {arm_file.name}")
    with open(arm_file, 'r') as f:
        arm_data = json.load(f)

    arm_data_block = bpy.data.armatures.new("Face_Armature_Data")
    armature_obj = bpy.data.objects.new("Face_Armature", arm_data_block)
    bpy.context.scene.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    joints = arm_data.get("joints", {})
    edit_bones = {}

    for joint_name, j_info in joints.items():
        pos = Vector(j_info["position"])
        bone = arm_data_block.edit_bones.new(joint_name)
        bone.head = pos
        if joint_name == "jaw":
            bone.tail = pos + Vector([0.0, -10.0, 15.0])
        elif "eye" in joint_name:
            bone.tail = pos + Vector([0.0, 0.0, 15.0])
        else:
            bone.tail = pos + Vector([0.0, 20.0, 0.0])
        edit_bones[joint_name] = bone

    for joint_name, j_info in joints.items():
        parent_name = j_info.get("parent")
        if parent_name and parent_name in edit_bones:
            edit_bones[joint_name].parent = edit_bones[parent_name]

    bpy.ops.object.mode_set(mode='OBJECT')

    weights = arm_data.get("skinning_weights", [])
    joint_names = arm_data.get("joint_names", list(joints.keys()))

    return armature_obj, arm_data, joint_names


def bind_skinning_weights(mesh_obj, armature_obj, weights, joint_names):
    import bpy

    if not weights or not armature_obj:
        return

    # Create vertex groups for each bone
    vgroups = {}
    for j_name in joint_names:
        vgroups[j_name] = mesh_obj.vertex_groups.new(name=j_name)

    for v_idx, v_weights in enumerate(weights):
        for j_idx, w_val in enumerate(v_weights):
            if w_val > 0.001 and j_idx < len(joint_names):
                vgroups[joint_names[j_idx]].add([v_idx], float(w_val), 'REPLACE')

    mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = armature_obj
    mod.use_vertex_groups = True


def apply_blendshapes(mesh_obj, blendshapes_path: str):
    import bpy
    from mathutils import Vector

    bs_file = Path(blendshapes_path) if blendshapes_path else None
    if not bs_file or not bs_file.exists():
        return

    mesh_obj.shape_key_add(name="Basis", from_mix=False)

    with open(bs_file, 'r') as f:
        bs_raw = json.load(f)

    bs_dict = bs_raw.get("blendshapes", bs_raw)
    print(f"[Blender Export] Applying {len(bs_dict)} ARKit shape key(s) to {mesh_obj.name}...")

    for bs_name, deltas in bs_dict.items():
        kb = mesh_obj.shape_key_add(name=bs_name, from_mix=False)
        for idx, d in enumerate(deltas):
            if idx < len(kb.data):
                kb.data[idx].co = kb.data[idx].co + Vector(d)


def import_obj_mesh(mesh_path: str, obj_name: str):
    import bpy

    if hasattr(bpy.ops.wm, 'obj_import'):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        bpy.ops.import_scene.obj(filepath=mesh_path)

    selected = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not selected:
        raise RuntimeError(f"Could not import mesh from {mesh_path}")

    mesh_obj = selected[0]
    mesh_obj.name = obj_name
    return mesh_obj


def setup_lighting_and_render_preview(output_render_path: str, target_mesh):
    import bpy
    from mathutils import Vector

    print(f"[Blender Export] Rendering neutral 3-point lighting preview...")

    # Calculate center and bounds
    bbox = [target_mesh.matrix_world @ Vector(corner) for corner in target_mesh.bound_box]
    center = sum(bbox, Vector((0, 0, 0))) / 8.0
    height = max(p.y for p in bbox) - min(p.y for p in bbox)

    # 1. Camera: positioned in front of face along -Y looking at center
    cam_data = bpy.data.cameras.new("Preview_Camera")
    cam_data.lens = 50.0  # Portrait lens
    cam_obj = bpy.data.objects.new("Preview_Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_dist = height * 2.2
    cam_obj.location = Vector([center.x, center.y - cam_dist, center.z + height * 0.05])
    # Look at target
    direction = center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # 2. Key Light (Warm, 45 deg right, high)
    key_light_data = bpy.data.lights.new(name="Key_Light", type='POINT')
    key_light_data.energy = 450.0
    key_light_data.color = (1.0, 0.96, 0.92)
    key_light = bpy.data.objects.new("Key_Light", key_light_data)
    key_light.location = Vector([center.x + height * 1.0, center.y - height * 1.5, center.z + height * 1.0])
    bpy.context.scene.collection.objects.link(key_light)

    # 3. Fill Light (Cool, 45 deg left, softer)
    fill_light_data = bpy.data.lights.new(name="Fill_Light", type='POINT')
    fill_light_data.energy = 180.0
    fill_light_data.color = (0.90, 0.94, 1.0)
    fill_light = bpy.data.objects.new("Fill_Light", fill_light_data)
    fill_light.location = Vector([center.x - height * 1.0, center.y - height * 1.2, center.z + height * 0.3])
    bpy.context.scene.collection.objects.link(fill_light)

    # 4. Rim Light (Backlight separating silhouette)
    rim_light_data = bpy.data.lights.new(name="Rim_Light", type='POINT')
    rim_light_data.energy = 600.0
    rim_light_data.color = (1.0, 1.0, 1.0)
    rim_light = bpy.data.objects.new("Rim_Light", rim_light_data)
    rim_light.location = Vector([center.x, center.y + height * 1.5, center.z + height * 1.2])
    bpy.context.scene.collection.objects.link(rim_light)

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.filepath = str(Path(output_render_path).resolve())
    scene.render.image_settings.file_format = 'PNG'

    bpy.ops.render.render(write_still=True)
    print(f"[Blender Export] Saved preview render to: {Path(output_render_path).name}")


def export_fbx_pipeline(args):
    import bpy

    # 1. Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import neutral base mesh
    print(f"[Blender Export] Importing base neutral mesh: {args.mesh}")
    head_mesh = import_obj_mesh(args.mesh, "Face_Neutral_Base")

    # 3. ARKit-52 Blendshapes
    apply_blendshapes(head_mesh, args.blendshapes)

    # 4. 5-Joint Armature Rigging
    armature_obj = None
    if args.armature and Path(args.armature).exists():
        armature_obj, arm_data, joint_names = build_armature(args.armature)
        if armature_obj:
            bind_skinning_weights(
                head_mesh,
                armature_obj,
                arm_data.get("skinning_weights", []),
                joint_names
            )

    # 5. UE5 Skin Material Slot & Textures
    skin_mat = create_or_get_material("M_Head_Skin")
    setup_skin_material(skin_mat, normal_map_path=args.normal_map, disp_map_path=args.displacement_map)
    if head_mesh.data.materials:
        head_mesh.data.materials[0] = skin_mat
    else:
        head_mesh.data.materials.append(skin_mat)

    # 6. Static Hair Cards Sub-mesh (if provided)
    hair_mesh = None
    if args.hair_cards and Path(args.hair_cards).exists():
        print(f"[Blender Export] Importing static facial hair cards: {args.hair_cards}")
        hair_mesh = import_obj_mesh(args.hair_cards, "Facial_Hair_Cards")

        # Material
        alpha_p = str(Path(args.hair_cards).parent / "hair_card_alpha.png")
        normal_p = str(Path(args.hair_cards).parent / "hair_card_normal.png")
        hair_mat = create_or_get_material("M_FacialHair_Cards")
        setup_hair_card_material(hair_mat, alpha_map_path=alpha_p, normal_map_path=normal_p)

        if hair_mesh.data.materials:
            hair_mesh.data.materials[0] = hair_mat
        else:
            hair_mesh.data.materials.append(hair_mat)

        # Bind hair cards to same armature
        if armature_obj and args.hair_skinning and Path(args.hair_skinning).exists():
            with open(args.hair_skinning, 'r') as f:
                hair_skin_data = json.load(f)
            bind_skinning_weights(
                hair_mesh,
                armature_obj,
                hair_skin_data.get("skinning_weights", []),
                hair_skin_data.get("bone_names", ["neck", "head", "jaw", "eye_L", "eye_R"])
            )

    # 7. Render Preview (if requested)
    if args.render_preview:
        setup_lighting_and_render_preview(args.render_preview, head_mesh)

    # 8. Export Main UE5 Live Link FBX
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    head_mesh.select_set(True)
    if hair_mesh:
        hair_mesh.select_set(True)
    if armature_obj:
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
    else:
        bpy.context.view_layer.objects.active = head_mesh

    print(f"[Blender Export] Writing production FBX: {out_path.name}...")
    bpy.ops.export_scene.fbx(
        filepath=str(out_path.resolve()),
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        bake_space_transform=True,
        object_types={'MESH', 'ARMATURE'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        add_leaf_bones=False,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        bake_anim=False,
        path_mode='COPY',
        embed_textures=True
    )
    print(f"[Blender Export] Successfully generated UE5 Live Link FBX: {out_path.name}")

    # 9. Export Separate LOD FBX Files if requested
    if args.lods and Path(args.lods).exists():
        with open(args.lods, 'r') as f:
            lod_data = json.load(f)
        lod_dir = out_path.parent / "lod"
        lod_dir.mkdir(parents=True, exist_ok=True)

        for lod_tier, info in lod_data.items():
            if lod_tier == "LOD0":
                continue
            mesh_p = info.get("mesh_obj")
            bs_p = info.get("blendshapes_json")
            if not mesh_p or not Path(mesh_p).exists():
                continue

            print(f"[Blender Export] Packaging {lod_tier} FBX...")
            lod_mesh = import_obj_mesh(mesh_p, f"Face_{lod_tier}")
            apply_blendshapes(lod_mesh, bs_p)
            if lod_mesh.data.materials:
                lod_mesh.data.materials[0] = skin_mat
            else:
                lod_mesh.data.materials.append(skin_mat)

            if armature_obj and args.armature:
                # Interpolated skinning for decimated topology
                bind_skinning_weights(
                    lod_mesh,
                    armature_obj,
                    arm_data.get("skinning_weights", [])[:len(lod_mesh.data.vertices)],
                    joint_names
                )

            lod_fbx_path = lod_dir / f"{lod_tier.lower()}.fbx"
            bpy.ops.object.select_all(action='DESELECT')
            lod_mesh.select_set(True)
            if armature_obj:
                armature_obj.select_set(True)
                bpy.context.view_layer.objects.active = armature_obj
            else:
                bpy.context.view_layer.objects.active = lod_mesh

            bpy.ops.export_scene.fbx(
                filepath=str(lod_fbx_path.resolve()),
                use_selection=True,
                apply_scale_options='FBX_SCALE_ALL',
                bake_space_transform=True,
                object_types={'MESH', 'ARMATURE'},
                use_mesh_modifiers=True,
                mesh_smooth_type='FACE',
                add_leaf_bones=False,
                bake_anim=False
            )
            # Remove lod mesh from scene to keep clean
            bpy.data.objects.remove(lod_mesh, do_unlink=True)
            print(f"[Blender Export] Saved {lod_tier} to: {lod_fbx_path.name}")


if __name__ == '__main__':
    args = parse_args()
    export_fbx_pipeline(args)
