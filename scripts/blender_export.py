"""
Headless Blender script for game-engine FBX export.
Integrates neutral base mesh, creates ARKit-52 shape keys (blendshapes),
and exports an FBX cleanly structured for Unreal Engine 5 Live Link.
Invoked via:
  blender --background --python scripts/blender_export.py -- <mesh.obj> <blendshapes.json> <output.fbx>
"""
import sys
from pathlib import Path

def export_fbx(mesh_path: str, blendshapes_path: str, output_fbx: str):
    import bpy

    # 1. Reset and clear current scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import neutral base OBJ mesh
    if hasattr(bpy.ops.wm, 'obj_import'):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        bpy.ops.import_scene.obj(filepath=mesh_path)

    selected = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not selected:
        print(f"Error: No mesh imported from {mesh_path}")
        return
    mesh_obj = selected[0]
    mesh_obj.name = "Face_Neutral_Base"

    # 3. Add basis shape key
    mesh_obj.shape_key_add(name="Basis", from_mix=False)

    # 4. Add ARKit-52 blendshape shape keys if available
    bs_file = Path(blendshapes_path)
    if bs_file.exists() and bs_file.suffix == '.json':
        import json
        with open(bs_file, 'r') as f:
            bs_data = json.load(f)
        for bs_name, deltas in bs_data.items():
            kb = mesh_obj.shape_key_add(name=bs_name, from_mix=False)
            # Apply delta displacements to vertices
            for idx, d in enumerate(deltas):
                if idx < len(kb.data):
                    kb.data[idx].co += d

    # 5. Export clean FBX for Unreal Engine 5
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
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
    print(f"[Blender Export] FBX export completed: {output_fbx}")

if __name__ == '__main__':
    # Parse arguments after '--'
    try:
        args_idx = sys.argv.index('--') + 1
        args = sys.argv[args_idx:]
        if len(args) >= 3:
            export_fbx(mesh_path=args[0], blendshapes_path=args[1], output_fbx=args[2])
        else:
            print("Usage: blender --background --python scripts/blender_export.py -- <mesh.obj> <blendshapes.json> <output.fbx>")
    except ValueError:
        print("No script arguments provided after '--'.")
