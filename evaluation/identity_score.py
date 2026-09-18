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
    scale = np.abs(v_xy).max() * 1.15 + 1e-6
    v_proj = ((v_xy / scale + 1.0) * 0.5 * (size - 1)).astype(np.int32)
    # Flip Y for standard image coordinate convention
    v_proj[:, 1] = (size - 1) - v_proj[:, 1]

    # Directional Lambertian shading & Painter's algorithm depth ordering
    # Triangles: v0, v1, v2
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    fn = fn / fn_norm

    # Key light: slightly elevated frontal directional source
    light_dir = np.array([0.0, 0.20, 0.98], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    diffuse = np.clip(np.sum(fn * light_dir, axis=1), 0.0, 1.0)
    # Ambient 0.20 + Diffuse 0.80 mapped to 0..235 grey range
    shades = np.clip(0.20 + 0.80 * diffuse, 0.0, 1.0) * 230.0

    # Depth sorting: mean Z per triangle (draw back-to-front)
    mean_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    face_order = np.argsort(mean_z)

    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    for idx in face_order:
        # Backface culling: skip faces pointing strictly away from camera
        if fn[idx, 2] < -0.05:
            continue
        col = int(shades[idx])
        pts = v_proj[faces[idx]].reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], (col, col, col), lineType=cv2.LINE_AA)

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

def calibrate_identity_threshold(
    positive_pairs: list,
    negative_pairs: Optional[list] = None,
    evaluator: Optional[IdentityEvaluator] = None
) -> dict:
    """
    Calibrates empirical ArcFace cosine similarity threshold from known pairs.
    positive_pairs: list of (render_path, photo_path) for the SAME individual.
    negative_pairs: optional list of (render_path, photo_path) for DIFFERENT individuals.
    Returns dictionary with mean, min, max, std, and recommended threshold.
    """
    if evaluator is None:
        evaluator = IdentityEvaluator()

    pos_scores = []
    for render_p, photo_p in positive_pairs:
        score = compute_identity_score(render_p, photo_p, evaluator=evaluator)
        pos_scores.append(score)

    pos_arr = np.array(pos_scores, dtype=np.float32)
    pos_mean = float(pos_arr.mean()) if len(pos_arr) > 0 else 0.0
    pos_std = float(pos_arr.std()) if len(pos_arr) > 0 else 0.0
    pos_min = float(pos_arr.min()) if len(pos_arr) > 0 else 0.0

    # Recommended conservative gate threshold: mean - 2*std or 0.40 minimum
    recommended_thresh = max(0.40, pos_mean - 2.0 * pos_std)

    neg_scores = []
    if negative_pairs:
        for render_p, photo_p in negative_pairs:
            score = compute_identity_score(render_p, photo_p, evaluator=evaluator)
            neg_scores.append(score)

    return {
        'positive_count': len(pos_scores),
        'positive_mean': pos_mean,
        'positive_std': pos_std,
        'positive_min': pos_min,
        'recommended_threshold': recommended_thresh,
        'negative_scores': neg_scores,
    }
