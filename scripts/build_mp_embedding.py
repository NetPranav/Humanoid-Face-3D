"""
Builds our own MediaPipe(478) -> FLAME surface correspondence (commercial path, DOCS/07 C2).

The FLAME<->MediaPipe embeddings that ship with research code are non-commercial, so this
derives one from data: for every subject with a fitted head (manifest from the pipeline),
the fitted mesh is rasterized in the photo, the surface point under each MediaPipe landmark
is read (face id + barycentrics), mapped to the FLAME template, and the points are
aggregated across subjects (coordinate-wise median). The median is snapped to the template
surface. The per-landmark spread across subjects (mm) becomes the landmark's fitting weight:
semantically stable points (eye corners, nose, lips) get high weight; hairline/silhouette
points that land on different anatomy per person get low weight.

Run again after the Phase 1 fitter has produced better fits to refine the embedding.

  python3 scripts/build_mp_embedding.py --runs outputs/golden_phase3a4a --golden data/golden_set

--mode geometric (recommended) registers MediaPipe's *neutral canonical face mesh* onto the
neutral FLAME template instead (similarity alignment on the stable data-driven correspondences,
then Laplacian-regularized non-rigid ICP, then closest-point snapping). This removes the
expression bias of the data-driven mode: most portraits smile, which drags lip/jaw
correspondences. The data-driven spread is still used for per-landmark weights.

  python3 scripts/build_mp_embedding.py --mode geometric --datadriven data/flame_model/mediapipe_embedding_datadriven.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.render.soft_raster import rasterize  # noqa: E402
from src.render.soft_renderer import project  # noqa: E402
from src.stage0_preprocess.landmarks_mp import MediaPipeLandmarker  # noqa: E402
from src.utils.flame_torch import FLAMETorch, load_flame  # noqa: E402


def fitted_vertices_cam(flame_layer: FLAMETorch, man: dict, view: int = 0) -> tuple:
    vf = man["view_fits"][view]
    beta = torch.tensor(man["beta_shape"], dtype=torch.float32)
    psi = torch.tensor(vf["psi"], dtype=torch.float32)
    pose = torch.zeros(15)
    pose[6:9] = torch.tensor(vf["jaw"], dtype=torch.float32)
    with torch.no_grad():
        v = flame_layer(beta, psi, pose).numpy().astype(np.float64)
    R, t = np.asarray(vf["R"]), np.asarray(vf["t"])
    K = np.array([[vf["intrinsics"]["fx"], 0, vf["intrinsics"]["cx"]],
                  [0, vf["intrinsics"]["fy"], vf["intrinsics"]["cy"]], [0, 0, 1.0]])
    return v @ R.T + t, K


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def geometric_embedding(flame_path: str, datadriven: str, out: str, canonical: str) -> None:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from src.stage0_preprocess.landmarks_mp import FACE_OVAL, LIPS_INNER
    from src.utils.flame_regions import load_vertex_masks

    fl = load_flame(flame_path)
    faces = fl.faces.astype(np.int64)
    tmpl = fl.v_template.astype(np.float64)
    dd = np.load(datadriven)
    can = trimesh.load(canonical, process=False)
    X = np.asarray(can.vertices, np.float64) / 100.0            # cm -> m
    cf = np.asarray(can.faces, np.int64)
    n = len(X)                                                   # 468

    # FLAME target surface without eyeballs
    vm = load_vertex_masks()
    eyes = np.zeros(len(tmpl), bool)
    eyes[np.concatenate([vm["left_eyeball"], vm["right_eyeball"]])] = True
    keep_f = ~eyes[faces].any(1)
    target = trimesh.Trimesh(tmpl, faces[keep_f], process=False)
    face_map = np.nonzero(keep_f)[0]

    # 1) similarity alignment on stable data-driven correspondences
    dd_pts = (tmpl[faces[dd["face_idx"]]] * dd["bary"][..., None]).sum(1)
    lips_outer = np.array([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185])
    stable = (np.nan_to_num(dd["spread_mm"], posinf=99) < 2.5)
    stable[:n] &= True
    stable[n:] = False
    stable[FACE_OVAL] = False
    stable[LIPS_INNER] = False
    stable[lips_outer] = False
    idx = np.nonzero(stable[:n])[0]
    s, R, t = umeyama(X[idx], dd_pts[idx])
    Y = (s * (R @ X.T)).T + t
    print(f"similarity on {len(idx)} stable points: scale {s:.4f}, rms "
          f"{np.sqrt(((Y[idx] - dd_pts[idx]) ** 2).sum(1).mean()) * 1000:.2f} mm")

    # 2) non-rigid ICP: minimize |Y+D - closest|^2 + a |L D|^2 + b |anchors|^2
    e = np.concatenate([cf[:, [0, 1]], cf[:, [1, 2]], cf[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    anchors = np.zeros(n)
    anchors[idx] = 1.0
    for alpha in (50.0, 20.0, 8.0, 3.0, 1.0):
        cp, dist, _ = trimesh.proximity.closest_point(target, Y)
        w = np.where(dist < 0.01, 1.0, 0.1)                      # ignore far (>1 cm) matches
        M = sp.diags(w) + alpha * (L.T @ L) + sp.diags(anchors * 5.0)
        rhs = sp.diags(w) @ (cp - Y) + sp.diags(anchors * 5.0) @ (dd_pts[:n] - Y)
        solve = spla.factorized(M.tocsc())
        D = np.stack([solve(rhs[:, c]) for c in range(3)], 1)
        Y = Y + D
        print(f"  NRICP alpha={alpha:5.1f}: median surface distance {np.median(dist) * 1000:.2f} mm")

    # 3) snap to the surface
    cp, dist, tri = trimesh.proximity.closest_point(target, Y)
    face_idx = np.array(dd["face_idx"]).copy()
    bary = np.array(dd["bary"]).copy()
    face_idx[:n] = face_map[tri]
    bary[:n] = trimesh.triangles.points_to_barycentric(target.triangles[tri], cp)
    moved = np.linalg.norm((tmpl[faces[face_idx[:n]]] * bary[:n, :, None]).sum(1) - dd_pts[:n], axis=1) * 1000
    np.savez(out, face_idx=face_idx, bary=bary, spread_mm=dd["spread_mm"], n_hits=dd["n_hits"],
             source=np.array("geometric: MediaPipe canonical mesh registered to FLAME template"))
    print(f"wrote {out}: geometric vs data-driven placement differs by median {np.median(moved):.2f} mm "
          f"(lips {np.median(moved[np.r_[LIPS_INNER, lips_outer]]):.2f} mm, face oval {np.median(moved[FACE_OVAL]):.2f} mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["outputs/golden_phase3a4a"])
    ap.add_argument("--golden", default="data/golden_set")
    ap.add_argument("--flame", default="data/flame_model/generic_model.pkl")
    ap.add_argument("--out", default="data/flame_model/mediapipe_embedding.npz")
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--mode", choices=["datadriven", "geometric"], default="datadriven")
    ap.add_argument("--datadriven", default="data/flame_model/mediapipe_embedding_datadriven.npz")
    ap.add_argument("--canonical", default="models_cache/mediapipe/canonical_face_model.obj")
    a = ap.parse_args()
    if a.mode == "geometric":
        geometric_embedding(a.flame, a.datadriven, a.out, a.canonical)
        return

    fl = load_flame(a.flame)
    layer = FLAMETorch(fl)
    faces = fl.faces.astype(np.int64)
    tmpl = fl.v_template.astype(np.float64)
    lmk = MediaPipeLandmarker()
    manifest = json.load(open(Path(a.golden) / "manifest.json"))

    hits = [[] for _ in range(478)]
    for run in a.runs:
        for m in manifest:
            mp = Path(run) / m["subject"] / "manifest.json"
            if not mp.exists():
                continue
            man = json.load(open(mp))
            if "view_fits" not in man:
                continue
            img = cv2.imread(str(Path(a.golden) / m["photos"][0]))
            det = lmk.detect(img)
            if det is None:
                continue
            vc, K = fitted_vertices_cam(layer, man)
            H, W = img.shape[:2]
            s = min(1.0, a.max_side / max(H, W))
            Ks = K.copy()
            Ks[:2] *= s
            xy, z = project(vc, Ks)
            r = rasterize(xy, z, faces, int(round(H * s)), int(round(W * s)))
            px = np.round(det.points[:, :2] * s - 0.5).astype(np.int64)
            ok = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < r.face_id.shape[1]) & (px[:, 1] < r.face_id.shape[0])
            for i in np.nonzero(ok)[0]:
                fid = r.face_id[px[i, 1], px[i, 0]]
                if fid < 0:
                    continue
                b = r.bary[px[i, 1], px[i, 0]].astype(np.float64)
                hits[i].append((tmpl[faces[fid]] * b[:, None]).sum(0))
            print(f"  {run}/{m['subject']}: {int(ok.sum())} landmarks on image", flush=True)

    mesh = trimesh.Trimesh(tmpl, faces, process=False)
    face_idx = np.zeros(478, np.int64)
    bary = np.zeros((478, 3))
    spread = np.full(478, np.inf)
    n_hits = np.array([len(h) for h in hits])
    med = np.zeros((478, 3))
    for i, h in enumerate(hits):
        if len(h) == 0:
            continue
        P = np.stack(h)
        med[i] = np.median(P, 0)
        spread[i] = float(np.median(np.linalg.norm(P - med[i], axis=1)) * 1000)
    have = n_hits > 0
    cp, dist, tri = trimesh.proximity.closest_point(mesh, med[have])
    face_idx[have] = tri
    bary[have] = trimesh.triangles.points_to_barycentric(mesh.triangles[tri], cp)
    np.savez(a.out, face_idx=face_idx, bary=bary, spread_mm=spread, n_hits=n_hits)
    fin = np.isfinite(spread)
    print(f"wrote {a.out}: {int(have.sum())}/478 landmarks placed; spread median {np.median(spread[fin]):.2f} mm, "
          f"p90 {np.percentile(spread[fin], 90):.2f} mm; min hits {n_hits[have].min()}")


if __name__ == "__main__":
    main()
