"""
Headless Blender script for game-engine FBX export.
Integrates neutral base mesh, creates ARKit-52 shape keys (blendshapes),
builds a 5-joint skeletal armature with skinning weights, and exports an FBX
cleanly structured for Unreal Engine 5 Live Link.

Invoked via:
  blender --background --python scripts/blender_export.py -- <mesh.obj> <blendshapes.json> <output.fbx> [--armature <armature.json>]
"""
import sys
import os
import json
from pathlib import Path


def export_fbx(
    mesh_path: str,
    blendshapes_path: str,
    output_fbx: str,
    armature_path: str = None
):
    import bpy
    from mathutils import Vector

    # 1. Reset and clear current scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import neutral base OBJ mesh
    if hasattr(bpy.ops.wm, 'obj_import'):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        bpy.ops.import_scene.obj(filepath=mesh_path)

    selected = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not selected:
        print(f"[Blender Export Error] No mesh imported from {mesh_path}")
        return
    mesh_obj = selected[0]
    mesh_obj.name = "Face_Neutral_Base"

    # Ensure mesh object is active and selected
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # 3. Add basis shape key
    mesh_obj.shape_key_add(name="Basis", from_mix=False)

    # 4. Add ARKit-52 blendshape shape keys if available
    bs_file = Path(blendshapes_path) if blendshapes_path else None
    if bs_file and bs_file.exists():
        with open(bs_file, 'r') as f:
            bs_raw = json.load(f)

        # Support both wrapped and unwrapped JSON schema
        bs_dict = bs_raw.get("blendshapes", bs_raw)

        print(f"[Blender Export] Applying {len(bs_dict)} shape key(s)...")
        for bs_name, deltas in bs_dict.items():
            kb = mesh_obj.shape_key_add(name=bs_name, from_mix=False)
            # Apply delta displacements to vertices using mathutils.Vector
            for idx, d in enumerate(deltas):
                if idx < len(kb.data):
                    kb.data[idx].co = kb.data[idx].co + Vector(d)

    # 5. Build 5-joint armature and skinning weights if available
    arm_file = Path(armature_path) if armature_path else None
    armature_obj = None
    if arm_file and arm_file.exists():
        print(f"[Blender Export] Building armature from: {arm_file.name}")
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

        # First pass: create bones with positions
        for joint_name, j_info in joints.items():
            pos = Vector(j_info["position"])
            bone = arm_data_block.edit_bones.new(joint_name)
            bone.head = pos
            # Offset tail slightly upwards or forward to give bone a positive length
            if joint_name == "jaw":
                bone.tail = pos + Vector([0.0, -10.0, 15.0])
            elif "eye" in joint_name:
                bone.tail = pos + Vector([0.0, 0.0, 15.0])
            else:
                bone.tail = pos + Vector([0.0, 20.0, 0.0])
            edit_bones[joint_name] = bone

        # Second pass: set parent relationships
        for joint_name, j_info in joints.items():
            parent_name = j_info.get("parent")
            if parent_name and parent_name in edit_bones:
                edit_bones[joint_name].parent = edit_bones[parent_name]

        bpy.ops.object.mode_set(mode='OBJECT')

        # Bind skinning weights to mesh
        weights = arm_data.get("skinning_weights", [])
        joint_names = arm_data.get("joint_names", list(joints.keys()))

        if weights:
            # Create vertex groups for each bone
            vgroups = {}
            for j_name in joint_names:
                vgroups[j_name] = mesh_obj.vertex_groups.new(name=j_name)

            for v_idx, v_weights in enumerate(weights):
                for j_idx, w_val in enumerate(v_weights):
                    if w_val > 0.001 and j_idx < len(joint_names):
                        vgroups[joint_names[j_idx]].add([v_idx], float(w_val), 'REPLACE')

            # Add Armature modifier to mesh
            mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
            mod.object = armature_obj
            mod.use_vertex_groups = True

    # 6. Select mesh and armature for export
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    if armature_obj:
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
    else:
        bpy.context.view_layer.objects.active = mesh_obj

    # 7. Export clean FBX for Unreal Engine 5 Live Link
    out_path = Path(output_fbx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.fbx(
        filepath=str(out_path.resolve()),
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        bake_space_transform=True,
        object_types={'MESH', 'ARMATURE'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='COPY'
    )
    print(f"[Blender Export] Clean FBX export completed: {out_path.name}")


if __name__ == '__main__':
    try:
        args_idx = sys.argv.index('--') + 1
        cli_args = sys.argv[args_idx:]
    except ValueError:
        cli_args = sys.argv[1:]

    if len(cli_args) >= 3:
        mesh_arg = cli_args[0]
        bs_arg = cli_args[1]
        out_arg = cli_args[2]
        arm_arg = None
        if '--armature' in cli_args:
            arm_idx = cli_args.index('--armature') + 1
            if arm_idx < len(cli_args):
                arm_arg = cli_args[arm_idx]

        export_fbx(
            mesh_path=mesh_arg,
            blendshapes_path=bs_arg,
            output_fbx=out_arg,
            armature_path=arm_arg
        )
    else:
        print("Usage: blender --background --python scripts/blender_export.py -- <mesh.obj> <blendshapes.json> <output.fbx> [--armature <armature.json>]")
