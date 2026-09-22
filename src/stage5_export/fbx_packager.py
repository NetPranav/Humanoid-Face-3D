"""
Stage 5 Master FBX Packager Engine.

Orchestrates headless Blender execution to package neutral base geometry,
ARKit-52 blendshape shape keys, 5-joint skeletal armature rigging,
static hair card geometry, Unreal Engine 5 material slots, and multi-tier LODs
into a unified production FBX asset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import os
import shutil
import subprocess
import json
import sys


class FBXPackager:
    """
    Production packager invoking headless Blender to compile UE5 Live Link FBX files.
    """
    def __init__(self, blender_path: Optional[str] = None):
        self.blender_bin = blender_path or self.find_blender_binary()

    @staticmethod
    def find_blender_binary() -> Optional[str]:
        """
        Locates the Blender executable across standard operating systems and Kaggle paths.
        """
        # 1. PATH environment variable
        found = shutil.which("blender")
        if found:
            return found

        # 2. Known standard paths
        candidates = [
            "/usr/local/bin/blender",
            "/usr/bin/blender",
            "/tmp/blender/blender",
            "/tmp/blender-4.1.0-linux-x64/blender",
            "/kaggle/working/blender-4.1.0-linux-x64/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ]
        for c in candidates:
            if Path(c).is_file() and os.access(c, os.X_OK):
                return c

        return None

    def package_fbx(
        self,
        mesh_obj: Optional[Union[str, Path]] = None,
        blendshapes_json: Optional[Union[str, Path]] = None,
        output_fbx: Optional[Union[str, Path]] = None,
        armature_json: Optional[Union[str, Path]] = None,
        lod_manifest: Optional[Union[str, Path]] = None,
        hair_cards_obj: Optional[Union[str, Path]] = None,
        hair_skinning_json: Optional[Union[str, Path]] = None,
        normal_map: Optional[Union[str, Path]] = None,
        displacement_map: Optional[Union[str, Path]] = None,
        render_preview: Optional[Union[str, Path]] = None,
        timeout_seconds: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Packages production FBX asset via headless Blender.

        Returns:
            dict containing execution status, output paths, and generated asset manifest.
        """
        mesh_obj = mesh_obj or kwargs.get("mesh_obj_path")
        blendshapes_json = blendshapes_json or kwargs.get("blendshapes_json_path")
        output_fbx = output_fbx or kwargs.get("output_fbx_path")
        armature_json = armature_json or kwargs.get("armature_json_path")
        lod_manifest = lod_manifest or kwargs.get("lod_manifest_path")

        if mesh_obj is None or blendshapes_json is None or output_fbx is None:
            raise ValueError("mesh_obj, blendshapes_json, and output_fbx are required arguments.")

        out_path = Path(output_fbx)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mesh_p = Path(mesh_obj)
        bs_p = Path(blendshapes_json)

        if not mesh_p.exists():
            raise FileNotFoundError(f"Base neutral mesh OBJ not found: {mesh_p}")
        if not bs_p.exists():
            raise FileNotFoundError(f"Blendshapes JSON not found: {bs_p}")

        blender_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "blender_export.py"
        if not blender_script.exists():
            raise FileNotFoundError(f"Blender export script not found: {blender_script}")

        # Assemble CLI arguments
        cmd_args = [
            "--mesh", str(mesh_p.resolve()),
            "--blendshapes", str(bs_p.resolve()),
            "--output", str(out_path.resolve()),
        ]

        if armature_json and Path(armature_json).exists():
            cmd_args.extend(["--armature", str(Path(armature_json).resolve())])
        if lod_manifest and Path(lod_manifest).exists():
            cmd_args.extend(["--lods", str(Path(lod_manifest).resolve())])
        if hair_cards_obj and Path(hair_cards_obj).exists():
            cmd_args.extend(["--hair_cards", str(Path(hair_cards_obj).resolve())])
        if hair_skinning_json and Path(hair_skinning_json).exists():
            cmd_args.extend(["--hair_skinning", str(Path(hair_skinning_json).resolve())])
        if normal_map and Path(normal_map).exists():
            cmd_args.extend(["--normal_map", str(Path(normal_map).resolve())])
        if displacement_map and Path(displacement_map).exists():
            cmd_args.extend(["--displacement_map", str(Path(displacement_map).resolve())])
        if render_preview:
            cmd_args.extend(["--render_preview", str(Path(render_preview).resolve())])

        # If Blender binary is available, execute headless packaging
        if self.blender_bin:
            print(f"[FBX Packager] Executing headless Blender ({self.blender_bin})...")
            full_cmd = [
                self.blender_bin,
                "--background",
                "--python", str(blender_script.resolve()),
                "--"
            ] + cmd_args

            try:
                res = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                if res.returncode == 0:
                    print(f"[FBX Packager] Successfully packaged FBX asset: {out_path.name}")
                    return {
                        "status": "success",
                        "fbx_file": str(out_path.resolve()),
                        "has_armature": bool(armature_json and Path(armature_json).exists()),
                        "has_blendshapes": True,
                        "has_hair_cards": bool(hair_cards_obj and Path(hair_cards_obj).exists()),
                        "render_preview": str(Path(render_preview).resolve()) if render_preview and Path(render_preview).exists() else None,
                        "blender_executable": self.blender_bin,
                    }
                else:
                    err_msg = res.stderr[:500] or res.stdout[:500]
                    print(f"[FBX Packager Warning] Blender returned code {res.returncode}:\n{err_msg}")
                    return {
                        "status": "blender_error",
                        "error_code": res.returncode,
                        "error_message": err_msg,
                        "fbx_file": None,
                    }
            except Exception as e:
                print(f"[FBX Packager Warning] Subprocess execution failed: {e}")
                return {
                    "status": "subprocess_error",
                    "error_message": str(e),
                    "fbx_file": None,
                }

        # If Blender is not available, generate actionable shell script and detailed packaging instructions
        print("[FBX Packager Notice] Blender binary not detected in PATH.")
        sh_script = out_path.parent / "run_blender_fbx_packaging.sh"
        sh_content = (
            "#!/usr/bin/env bash\n"
            "# Automated headless Blender FBX packaging script\n"
            "set -euo pipefail\n\n"
            "BLENDER_BIN=\"${1:-blender}\"\n"
            "if ! command -v \"$BLENDER_BIN\" &> /dev/null; then\n"
            "    echo \"[Error] Blender not found. Please install Blender or provide path as argument.\"\n"
            "    exit 1\n"
            "fi\n\n"
            f"\"$BLENDER_BIN\" --background --python \"{blender_script.resolve()}\" -- \\\n"
        )
        for i in range(0, len(cmd_args), 2):
            val = cmd_args[i+1] if i+1 < len(cmd_args) else ""
            sh_content += f"    {cmd_args[i]} \"{val}\" \\\n"
        sh_content = sh_content.rstrip(" \\\n") + "\n"

        with open(sh_script, 'w') as f:
            f.write(sh_content)
        sh_script.chmod(0o755)

        print(f"[FBX Packager Notice] Created executable packaging script: {sh_script.name}")
        return {
            "status": "dry_run_script_generated",
            "fbx_file": None,
            "shell_script": str(sh_script.resolve()),
            "message": "Blender binary not in PATH. Generated run_blender_fbx_packaging.sh for single-click execution on workstation or Kaggle.",
        }
