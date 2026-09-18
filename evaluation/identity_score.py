import cv2
import numpy as np
from pathlib import Path
from typing import Optional

class IdentityEvaluator:
    """
    Measures facial identity retention using ArcFace cosine similarity
    between input photographic portraits and rendered output geometry.
    """
    def __init__(self, ctx_id: int = 0):
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.has_insightface = True
        except Exception as e:
            raise RuntimeError(
                "InsightFace is required for Identity verification. "
                "Install with: pip install insightface onnxruntime-gpu"
            ) from e

    def extract_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        faces = self.app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected in target image for identity verification.")
        emb = faces[0].embedding
        return emb / np.linalg.norm(emb)

    def evaluate(self, render_bgr: np.ndarray, photo_bgr: np.ndarray) -> float:
        emb_render = self.extract_embedding(render_bgr)
        emb_photo = self.extract_embedding(photo_bgr)
        return float(np.dot(emb_render, emb_photo))

def render_neutral_preview(mesh_path: str, out_png: str, size: int = 512) -> str:
    """
    Renders an orthographic frontal preview of the reconstructed neutral mesh.
    Uses trimesh and OpenCV polygon rendering without requiring an X11 display server.
    """
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, process=False)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
    except ImportError:
        # Zero-dependency OBJ loader fallback
        verts_list = []
        faces_list = []
        with open(mesh_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    verts_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.strip().split()
                    face = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                    faces_list.append(face)
        vertices = np.array(verts_list, dtype=np.float32)
        faces = np.array(faces_list, dtype=np.int64)

    # Frontal orthographic projection: center in canvas
    v_xy = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    scale = np.abs(v_xy).max() * 1.1 + 1e-6
    v_proj = ((v_xy / scale + 1.0) * 0.5 * (size - 1)).astype(np.int32)
    # Flip Y for image coordinate convention
    v_proj[:, 1] = (size - 1) - v_proj[:, 1]

    canvas = np.full((size, size, 3), 35, dtype=np.uint8)
    for f in faces:
        pts = v_proj[f].reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], (180, 180, 180))

    cv2.imwrite(str(out_png), canvas)
    return str(out_png)

def compute_identity_score(
    mesh_path: str,
    photo_path: str,
    evaluator: Optional[IdentityEvaluator] = None
) -> float:
    """
    Computes ArcFace cosine similarity score for a generated mesh against the source photo.
    Requires that an actual rendered mesh preview exists at {mesh_name}.png.
    Refuses to evaluate if the rendered preview is missing (no self-grading).
    """
    preview_path = Path(mesh_path).with_suffix('.png')
    if not preview_path.exists():
        raise FileNotFoundError(
            f"No rendered preview found at {preview_path}. "
            "An identity score cannot be computed without a rendered mesh image. "
            "Call render_neutral_preview(mesh_path, preview_path) first."
        )

    if evaluator is None:
        evaluator = IdentityEvaluator()

    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise FileNotFoundError(f"Could not read source photograph at: {photo_path}")

    render = cv2.imread(str(preview_path))
    if render is None:
        raise FileNotFoundError(f"Could not read rendered preview image at: {preview_path}")

    return evaluator.evaluate(render, photo)
