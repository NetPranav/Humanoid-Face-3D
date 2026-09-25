"""
Runs the pipeline on the golden set and builds a v1-vs-current comparison (DOCS/04 §9).

  python3 scripts/run_golden_set.py \
      --golden data/golden_set --out outputs/golden_phase0 \
      --baseline outputs/golden_v1_baseline          # optional: v1 outputs to compare against

Writes <out>/_summary/: contact_sheet.jpg (one row per subject), <subject>_row.jpg,
summary.json and summary.md. Baseline renders use the same renderer, turntable cameras and
identity judge as the current pipeline, so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.phase0_eval import BG, IdentityJudge, _crop_square, judge_turntable, render_turntable  # noqa: E402
from src.pipeline import FaceGeoPipeline  # noqa: E402

TILE = 384


def load_obj_vertices(path: Path) -> np.ndarray:
    return np.array([[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")], np.float64)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (40, 40, 40), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def tile(img: np.ndarray, text: str) -> np.ndarray:
    return label(cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA), text)


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden_set")
    ap.add_argument("--out", default="outputs/golden_phase0")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--baseline-label", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-run", action="store_true", help="only rebuild the summary from existing outputs")
    a = ap.parse_args()

    golden, out = Path(a.golden), Path(a.out)
    manifest = json.load(open(golden / "manifest.json"))
    pipe = FaceGeoPipeline(a.config)
    judge = IdentityJudge(pipe.detector.app)
    faces = pipe.flame.faces.astype(np.int64)
    uv, uvf = pipe.uv_layout()

    rows, summary = [], []
    for m in manifest:
        sid = m["subject"]
        photos = [str(golden / p) for p in m["photos"]]
        sub_out = out / sid
        t = time.time()
        if not a.skip_run:
            pipe.run(photos, str(sub_out))
        runtime = time.time() - t
        rep = json.load(open(sub_out / "report.json"))
        man = json.load(open(sub_out / "manifest.json"))

        im = cv2.imread(photos[0])
        det = pipe.detector.detect_single(im)
        ref = det.embedding
        entry = {
            "subject": sid, "camera": m.get("camera"), "focal_source": rep["views"][0]["focal_source"],
            "runtime_s": round(runtime, 1) if not a.skip_run else None,
            "lmk_err_iod": rep["lmk_err_iod_mean"],
            "id_input_view": rep["views"][0].get("id_input_view"),
            "v3": {k: rep.get(k) for k in ("id_yaw+0", "id_yaw+30", "id_yaw-30")},
            "v3_relit": {k.replace("_relit", ""): rep.get(k) for k in ("id_relit_yaw+0", "id_relit_yaw+30", "id_relit_yaw-30")},
            "texture_fractions": rep.get("texture_fractions"),
            "stage_status": man.get("stage_status"),
        }

        # current pipeline tiles
        cur_v = load_obj_vertices(sub_out / "head_mesh.obj")
        cur_tex = cv2.imread(str(sub_out / "textures" / "head_albedo_diffuse.png"))
        tt = render_turntable(cur_v, faces, uv, uvf, cur_tex, yaws=(0, 30, 60))
        geo = render_turntable(cur_v, faces, uv, uvf, None, yaws=(30,), textured=False)[0]
        overlay = cv2.imread(str(sub_out / "eval" / "overlay_view0.jpg"))
        ov_textured = overlay[:, 1024:1536]
        tiles = [tile(_crop_square(im, det.bbox), f"{sid}: input photo")]

        # baseline tiles
        if a.baseline and (Path(a.baseline) / sid / "head_mesh.obj").exists():
            bdir = Path(a.baseline) / sid
            b_v = load_obj_vertices(bdir / "head_mesh.obj")
            b_tex = cv2.imread(str(bdir / "textures" / "head_albedo_diffuse.png"))
            entry["v1"] = judge_turntable(judge, ref, b_v, faces, uv, uvf, b_tex)
            btt = render_turntable(b_v, faces, uv, uvf, b_tex, yaws=(0, 30, 180))
            bl = a.baseline_label
            tiles += [tile(btt[0], f"{bl}: front"), tile(btt[1], f"{bl}: 30 deg"), tile(btt[2], f"{bl}: back")]

        back = render_turntable(cur_v, faces, uv, uvf, cur_tex, yaws=(180,))[0]
        tiles += [tile(ov_textured, "new: over photo (fitted cam)"), tile(tt[0], "new: front"),
                  tile(tt[1], "new: 30 deg"), tile(back, "new: back")]
        for name, label_ in (("clay_front", "new: sculpt mesh"), ("clay_eye", "new: sculpt eye"),
                             ("clay_mouth", "new: sculpt mouth")):
            cp = sub_out / "eval" / f"{name}.png"
            if cp.exists():
                tiles.append(tile(cv2.imread(str(cp)), label_))
        if len(tiles) < 9:
            tiles += [tile(tt[2], "new: 60 deg"), tile(geo, "new: geometry 30 deg")]
        row = np.hstack(tiles)
        (out / "_summary").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "_summary" / f"{sid}_row.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(row)
        summary.append(entry)
        print(f"[golden] {sid}: lmk {100 * entry['lmk_err_iod']:.1f}% IOD | id front {a.baseline_label} "
              f"{fmt(entry.get('v1', {}).get('id_yaw+0'))} -> new {fmt(entry['v3']['id_yaw+0'])}", flush=True)

    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=BG) for r in rows])
    cv2.imwrite(str(out / "_summary" / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(summary, open(out / "_summary" / "summary.json", "w"), indent=2)

    def mean(key, side):
        vals = [s[side][key] for s in summary if side in s and s[side].get(key) is not None]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    bl = a.baseline_label
    lines = [f"| subject | focal source | landmark err (% IOD) | id front {bl} → new | id +30° {bl} → new | id −30° {bl} → new | id input view | observed / mirrored / synthesized / filled |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        v1 = s.get("v1", {})
        tf = s.get("texture_fractions") or {}
        lines.append(
            f"| {s['subject']} | {s['focal_source']} | {100 * s['lmk_err_iod']:.1f} | "
            f"{fmt(v1.get('id_yaw+0'))} → {fmt(s['v3']['id_yaw+0'])} | "
            f"{fmt(v1.get('id_yaw+30'))} → {fmt(s['v3']['id_yaw+30'])} | "
            f"{fmt(v1.get('id_yaw-30'))} → {fmt(s['v3']['id_yaw-30'])} | {fmt(s['id_input_view'])} | "
            f"{100 * tf.get('observed', 0):.0f}% / {100 * tf.get('inferred_symmetry', 0):.0f}% / "
            f"{100 * tf.get('synthesized_own_skin', 0):.0f}% / {100 * tf.get('filled_interpolated', 0):.0f}% |")
    lines.append("")
    for k in ("id_yaw+0", "id_yaw+30", "id_yaw-30"):
        b, nb = mean(k, "v1")
        c, nc = mean(k, "v3")
        r, nr = mean(k, "v3_relit")
        lines.append(f"- mean {k}: {bl} {fmt(b)} (n={nb}) → new {fmt(c)} (n={nc}); new relit with photo lighting {fmt(r)} (n={nr})")
    (out / "_summary" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
