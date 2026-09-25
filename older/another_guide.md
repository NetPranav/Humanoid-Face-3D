# Face Geometry Pipeline — Implementation Guide

> This is the **execution guide**, not the research reference. It tells you what to type, in what order. For *why* any decision was made, see `Research.md` and `DOCS/01–03`. For what is currently broken and why, see `DOCS/00_Code_Review.md`.
>
> ⚠️ **HISTORICAL AUDIT NOTICE:**
> For the postmortem on the failed Meta Multiface Detail GAN experiment, see [DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md).
> For the active film-grade MetaHuman synthesis architecture, see [DOCS/02_Metahuman_Film_Grade_Synthesis.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Metahuman_Film_Grade_Synthesis.md) and [DOCS/05_Blender_Cycles_Film_Rendering_Engine.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/05_Blender_Cycles_Film_Rendering_Engine.md).
>
> **Revision note.** The previous version of this guide described code that had never been run, using phrasing ("Copy exactly", "verified", "Day 0 is complete when…") that implied it had been. Several steps could not work. This revision fixes them, marks what is real versus aspirational, and adds the work that was missing entirely — chiefly Stage 5 retopology, blendshape generation, and the licensing audit.

---

## Status legend

Every step is tagged. Nothing is presented as working unless it has been run.

| Tag | Meaning |
|---|---|
| ✅ **WORKS** | Code exists and has been executed successfully |
| 🔧 **NEEDS FIX** | Code exists, has a known defect, fix is given inline |
| 🏗 **STUB** | File exists but does nothing useful; must be written |
| ❌ **MISSING** | Does not exist at all |

Current repo status as of this revision:

| Component | Status |
|---|---|
| `src/utils/flame_model.py` | ✅ **WORKS** (after the rewrite shipped with the review) |
| `src/stage0_preprocess/detector.py` | 🔧 fabricates detections on import failure |
| `src/utils/validation.py` | 🔧 same-person check silently never runs |
| `src/stage1_identity/inference.py` | 🔧 MICA never loads; preprocessing wrong |
| `src/stage2_expression/encoder.py` | 🔧 checkpoint load wrong; SMIRK has no detail branch |
| `src/pipeline.py` | 🔧 degenerate-mesh fallback |
| `src/stage3_detail/*` | 🔧 three convergence-blocking bugs |
| `scripts/build_uv_displacement_dataset.py` | 🏗 **STUB** — writes a hardcoded constant and exits |
| `scripts/blender_export.py` | 🔧 raises on first shape key; input file never generated |
| `src/stage4_facial_hair/` | ❌ empty |
| `src/stage5_export/` | ❌ empty — retopology, blendshapes, LODs, skeleton all missing |
| `evaluation/identity_score.py` | 🔧 compares the photo to itself; gate cannot fail |
| tests | ❌ none |

---

## Before you start — the mental model

The pipeline has 6 stages. Each is a Python module. Trained weights live in Kaggle's Model registry under your account. `production_inference.ipynb` loads all stages and runs them in sequence on a new set of photos. You never need a GPU locally.

Your local machine is for writing code, committing to git, and reviewing meshes in Blender.

```
Day 0        → Environment, FLAME loader, nvdiffrast, licence decisions
Days 1–5     → Phase 0: make the existing code fail honestly, then pass honestly
Weeks 1–2    → Phase 1: inference-only baseline with pretrained weights
Weeks 3–4    → Phase 2: fine-tune MICA
Weeks 4–5    → Phase 2.5: build the UV displacement dataset (was a stub)
Weeks 5–9    → Phase 3: detail GAN
Weeks 1–10   → Phase 4: licensing + own-capture (starts NOW, not week 8)
Weeks 5–12   → Phase 5: retopology + rig  ← the real critical path
Weeks 10–14  → Phase 6: hybrid detail, facial hair
Weeks 12–16  → Phase 7: production export, UE5 integration
Ongoing      → Phase 8: evaluation
```

**Two schedule changes versus the previous guide, and the reasons:**

- **Phase 4 (licensing) moves to week 1.** It is a calendar-time dependency, not an effort dependency. Emails, consent forms, and subject recruitment don't compress no matter how much you work. Starting it in week 8 means discovering in week 12 that the weights are unshippable.
- **Phase 5 (retopology) expands from 3 weeks to 6–8 and starts in week 5.** Previously it was one page about installing Blender. It is actually the difference between a research mesh and a game asset, and it is entirely independent of the ML — so it can run in parallel from the moment the FLAME topology is fixed.

---

## Kaggle budget — read this before planning any training

The previous guide only mentioned the 12-hour session cap. The binding constraint is the **weekly quota**.

- Kaggle gives roughly **30 GPU-hours per week**.
- On a **T4×2** accelerator, one hour of wall-clock consumes **two** quota-hours.
- So: **~15 wall-clock hours of 2×T4 per week.** A single maxed-out 12-hour session costs ~24 of your 30 hours. You get **one full session per week, plus a short one.**

Plan Phase 3 against ~15 wall-clock hours/week, not against the session limit. Keep a running ledger:

```python
# Put this at the end of every training notebook
import time, json
hrs = (time.time() - start_time) / 3600
print(f"This session: {hrs:.2f} wall-clock h = {hrs*2:.2f} quota-h on T4x2")
```

CPU-only sessions cost **zero** GPU quota. Dataset building, evaluation, retopology, and all Blender export work belong on CPU sessions. This is a large lever — use it.

---

## Day 0 — Hard blockers

### Step 1 — Kaggle account

1. kaggle.com → Settings → **Phone verification**. Without it, Internet is disabled in notebooks and you cannot pull weights.
2. Settings → API → Create New Token → `kaggle.json`.
3. Confirm you can select **GPU T4 ×2** under Accelerator.

### Step 2 — GitHub repo

Create a private repo `face-geo-pipeline`.

```bash
# Note: rename the project directory first. The trailing space in "3d Model /"
# breaks shell scripts and CI.
mv "3d Model " face-geo-pipeline
cd face-geo-pipeline
git init && git add -A && git commit -m "Initial import"
git remote add origin https://github.com/youruser/face-geo-pipeline.git
git push -u origin main
```

Commit now. The current tree has a `.git` directory with **zero commits** — no history, no baseline, no bisect.

Add to `.gitignore`:
```
__MACOSX/
.DS_Store
._*
data/flame_model/*.pkl
models_cache/
outputs/
```

### Step 3 — Kaggle → GitHub access

**Use a fine-grained PAT over HTTPS, not SSH.** The SSH route in the previous guide fails on a subtle issue: Kaggle Secrets strip trailing whitespace, and OpenSSH rejects a private key whose final line has no newline (`error in libcrypto`).

GitHub → Settings → Developer settings → Fine-grained tokens → read-only on this one repo. Store as Kaggle Secret `github_pat`.

```python
from kaggle_secrets import UserSecretsClient
import subprocess, os
tok = UserSecretsClient().get_secret("github_pat").strip()
url = f"https://x-access-token:{tok}@github.com/youruser/face-geo-pipeline.git"
subprocess.run(["git", "clone", url, "/kaggle/working/pipeline"], check=True)
print("Clone OK" if os.path.exists("/kaggle/working/pipeline/src") else "FAILED")
```

If you insist on SSH: `f.write(deploy_key.rstrip() + "\n")` and `chmod 700 ~/.ssh`.

### Step 4 — Decide your FLAME version **before writing any code**

This is a new Day-0 step and it is the most consequential decision on the page, because changing it later invalidates every checkpoint you train.

`configs/default.yaml` currently pins `FLAME2020`. FLAME 2020 is **non-commercial research only**. In November 2025, MPI released **FLAME 2023 Open under CC-BY-4.0**, alongside conversion code for translating expression parameters from FLAME 2023 to FLAME 2023 Open. Previous versions remain non-commercial.

| If your goal is… | Use |
|---|---|
| Research / prototype only, never shipped | FLAME 2020 (largest ecosystem, what MICA/SMIRK were trained against) |
| Anything you intend to ship commercially | **FLAME 2023 Open (CC-BY-4.0)** |

Verify the current licence text yourself at https://flame.is.tue.mpg.de before committing — I'm summarising a changelog entry, and terms change. This is not legal advice.

If you pick 2023 Open, note the knock-on: MICA and SMIRK's pretrained heads predict into the **2020** basis. You will need the MPI conversion code, or to fine-tune the regressor head against 2023 Open. Budget a week. Doing this *now* is cheap; doing it after Phase 3 means retraining the GAN.

Record the decision in the config and stop treating it as a detail:
```yaml
stage1:
  flame_version: FLAME2023_Open   # CC-BY-4.0. See DOCS/00_Code_Review.md licensing table.
  flame_path: data/flame_model/flame2023_Open.pkl
```

### Step 5 — Verify the FLAME loader

Download the model from https://flame.is.tue.mpg.de (registration required). You need **two** files, not one — the previous guide only mentioned the first:

- `generic_model.pkl` / `flame2023_Open.pkl` — geometry basis
- `FLAME_texture.npz` **or** `head_template.obj` — **the UV coordinates.** The `.pkl` contains no UV layout. Stage 3 cannot rasterise anything without this, and the previous guide never mentioned fetching it. Get it now.

Place both in `data/flame_model/`.

```python
import sys; sys.path.insert(0, '/kaggle/working/pipeline')
import numpy as np
from src.utils.flame_model import FLAMEModel

flame = FLAMEModel('/kaggle/working/pipeline/data/flame_model/flame2023_Open.pkl')
v, f = flame.decode_neutral(np.zeros(300, np.float32))

print(f"verts      : {v.shape}          expect (5023, 3)")
print(f"faces      : {f.shape}          expect (9976, 3)")
print(f"NaN        : {np.isnan(v).any()} expect False")
print(f"range      : {v.min():.4f} .. {v.max():.4f} {flame.units}")

# Pose round-trip: a zero pose must be the identity transform
vp, _ = flame.decode(np.zeros(300), np.zeros(100), np.zeros(15))
print(f"zero-pose deviation: {np.abs(vp - v).max():.2e}   expect ~1e-8")

# Jaw must actually move geometry
theta = np.zeros(15); theta[6] = 0.3         # jaw joint, axis-angle
vj, _ = flame.decode(np.zeros(300), np.zeros(100), theta)
print(f"jaw moves mesh: {np.abs(vj - v).max() > 1e-4}   expect True")
```

Expected:
```
verts      : (5023, 3)
faces      : (9976, 3)
NaN        : False
range      : -0.1300 .. 0.1300 m
zero-pose deviation: 1.49e-08
jaw moves mesh: True
```

**Three corrections against the previous guide, all of which made its Day-0 check impossible to pass:**

1. **There is no `exprdirs` key.** FLAME packs identity and expression into one `shapedirs` array of shape `(5023, 3, 400)` — `[..., :300]` identity, `[..., 300:400]` expression. The old loader did `data['exprdirs']` and raised `KeyError` on the first line of the first stage.

2. **The chumpy→numpy unpickler trick does not work.** Returning `np.array` from `find_class` gives:
   ```
   UnpicklingError: NEWOBJ class argument must be a type, not builtin_function_or_method
   ```
   because `np.array` is a function, not a type. The working approach is to re-add the numpy aliases chumpy needs and import real chumpy, with a stub-class unpickler as fallback. Both are implemented in the replacement `src/utils/flame_model.py`.

3. **FLAME is in metres, not millimetres.** The old guide told you to expect "roughly -150 to 150". Correct is roughly **-0.13 to 0.13 m**. If you want millimetres, pass `scale_to_mm=True` — and then be consistent everywhere, including the displacement p99, or Stage 3 will be off by 1000×.

**Do not proceed until all four lines print as expected.**

### Step 6 — Verify nvdiffrast

`pip install nvdiffrast` **does not work** — nvdiffrast is not on PyPI. The previous guide's Day-0 step and `requirements.txt` line 3 both fail.

```python
!pip install -q setuptools wheel ninja
!pip install -q git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

import torch, nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
pos = torch.tensor([[[-0.5,-0.5,0,1],[0.5,-0.5,0,1],[0,0.5,0,1]]],
                   dtype=torch.float32, device='cuda')
tri = torch.tensor([[0,1,2]], dtype=torch.int32, device='cuda')
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256,256])
assert tuple(rast.shape) == (1,256,256,4), rast.shape
print("nvdiffrast OK")
```

First call JIT-compiles CUDA extensions — expect 1–3 minutes, and a hard failure if `ninja` is absent. Licence: NVIDIA Source Code License, business enquiries go to NVIDIA Research Licensing. Add it to the licence audit.

### Step 7 — Fixed `requirements.txt`

```
torch>=2.1.0
torchvision
insightface
onnxruntime-gpu
trimesh
opencv-python
Pillow
numpy>=1.23,<2.0
scipy
scikit-learn
pyyaml
kagglehub
matplotlib
tqdm
pytest
chumpy            # needed to unpickle FLAME; import after the numpy-alias shim
ninja             # required to build nvdiffrast
```

Removed and why:
- `nvdiffrast` — not installable from PyPI; install from git (Step 6).
- `mediapipe` — listed but never imported anywhere in the codebase. Either use it for dense landmarks (recommended: InsightFace's 5-point is thin for FLAME fitting) or leave it out.
- `open3d` — never imported. It's a ~400 MB install.

**The numpy pin needs a kernel restart.** Kaggle ships numpy 2.x; downgrading mid-session leaves already-imported C extensions linked against the wrong ABI. Do the pip install in cell 1, restart, then import anything.

```python
# Cell 1 — then Run > Restart before Cell 2
!pip install -q -r /kaggle/working/pipeline/requirements.txt
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

**Not in requirements, but needed if you use SMIRK: PyTorch3D.** SMIRK depends on it, not on nvdiffrast. It is a slow, version-sensitive build on Kaggle (10–20 min). Either budget for it, or extract only SMIRK's *encoder* weights — you don't need its renderer at inference.

### Day 0 gate

- [ ] `git log` shows at least one commit
- [ ] Kaggle clone prints "Clone OK"
- [ ] FLAME prints all four expected lines, including zero-pose deviation ~1e-8
- [ ] UV coordinates file downloaded alongside the model
- [ ] nvdiffrast prints OK
- [ ] FLAME version decision recorded in `configs/default.yaml` with a reason
- [ ] `pip install -r requirements.txt` completes without error

---

## Phase 0 — Make the code fail honestly (Days 1–5)

**This phase did not exist before, and it is the most important one on the page.**

The codebase is built almost entirely out of silent fallbacks. Every module degrades to zeros when a dependency is missing and then reports success. Stacked, they let the old Phase 1 gate pass on a pipeline that has done nothing. Until this is fixed, every number you measure is meaningless.

**Goal:** remove every silent fallback, then watch the gate fail. That failure is your first real measurement.

### Step 1 — The self-grading identity score

`evaluation/identity_score.py`:
```python
preview_path = Path(mesh_path).with_suffix('.png')
if preview_path.exists(): render = cv2.imread(str(preview_path))
else:                     render = photo          # compares photo to itself → 1.000
```

Nothing in the repo ever writes that PNG. So `render is photo`, similarity is exactly 1.0, and `assert score > 0.5` **cannot fail**.

Fix — write a real renderer and refuse to score without one:

```python
def render_neutral_preview(vertices, faces, out_png, size=512):
    """Orthographic Lambertian shade of the neutral mesh, frontal view.
    nvdiffrast if you have it; a numpy z-buffer is ~30 lines and needs no GPU."""
    ...

def compute_identity_score(mesh_path, photo_path):
    preview = Path(mesh_path).with_suffix('.png')
    if not preview.exists():
        raise FileNotFoundError(
            f"No render at {preview}. An identity score without a render is not a measurement."
        )
    ...
```

**Then recalibrate the threshold.** `arcface_threshold: 0.5` was borrowed from face *verification*, where both inputs are photographs. An untextured grey shaded mesh scores far lower against a photo — that gap is a domain shift, not an identity failure. Establish your own baseline: render 20 meshes you've confirmed by eye are correct, score them, and set the threshold from that distribution. Until you do, 0.5 is a number with no provenance and it will either pass everything or fail everything.

### Step 2 — Remove the fabricated detection

`src/stage0_preprocess/detector.py` returns the whole image resized, with `det_score=0.99` and `yaw_deg=0.0`, when InsightFace fails to import. That single fallback defeats three of your five input validations at once: the yaw check passes (everything is 0.0), the angular-spread check can't fail, and MICA receives an unaligned full-frame image.

```python
except ImportError as e:
    raise RuntimeError(
        "InsightFace is required for Stage 0. "
        "pip install insightface onnxruntime-gpu"
    ) from e
```

Keep a `--allow-degraded` flag for CI smoke tests if you want, and have it stamp `"degraded": true` into `manifest.json` so a degraded run is never mistaken for a real one.

### Step 3 — Remove the degenerate-mesh fallback

`src/pipeline.py` writes 5023 coincident vertices and 9976 faces all pointing at vertex 1 when FLAME is missing. It opens in Blender and looks like an exporter bug rather than a missing model. Raise instead.

### Step 4 — Fix the same-person check, which currently never runs

`src/utils/validation.py`:
```python
arcface = insightface.model_zoo.get_model('buffalo_l')   # wrong: pack, not a file
```
`buffalo_l` is a model *pack*; `get_model` wants a path to a `.onnx`. This raises, gets swallowed by `except Exception: pass`, `embeddings` stays empty, and the guard `if len(embeddings) == len(photo_paths)` is False — so the check is skipped for every run.

Also: the ArcFace model is constructed **inside the per-photo loop**, so a 5-photo run initialises InsightFace six times, and `pipeline.run()` then makes a seventh and re-detects everything validation already detected.

```python
def validate_inputs(photo_paths, detector, app, min_photos=3, same_person_thresh=0.40):
    """Takes the already-constructed detector and FaceAnalysis app.
    Returns detections so the pipeline can reuse them instead of re-detecting."""
    ...
    faces = app.get(img)                    # face.embedding is already L2-normalised
    embeddings.append(faces[0].normed_embedding)
```

Then in `pipeline.run()`, consume `val_res.detections` rather than running Stage 0 again. Right now the result is computed, discarded, and recomputed twenty lines later.

### Step 5 — Remove the zero-tensor dataset fallback

`src/stage3_detail/data.py` returns all-zero tensors with an all-ones mask when no files are found, and `__len__` returns `max(len, 1)`. Masked L1 on zeros is exactly 0.0, so training logs look healthy for hours while learning nothing.

Note the **old guide's version was correct here** (`raise RuntimeError`) — the code regressed from it. Restore the raise.

### Step 6 — Add the tests

Seven tests, none over 20 lines, that would have caught most of the above:

```python
# tests/test_pipeline.py
def test_flame_loads(flame):
    v, f = flame.decode_neutral(np.zeros(300))
    assert v.shape == (5023, 3) and not np.isnan(v).any()
    assert -0.25 < v.min() and v.max() < 0.25          # metres

def test_neutral_is_pose_free(flame):
    b = np.random.randn(300) * 0.1
    assert np.allclose(flame.decode(b, None, None)[0], flame.decode_neutral(b)[0])

def test_distinct_identities(pipeline, two_subjects):
    b1, b2 = (pipeline.encode(s) for s in two_subjects)
    assert np.linalg.norm(b1 - b2) > 1e-3              # catches the zeros(300) fallback

def test_disp_roundtrip():
    x = np.random.uniform(-1, 1, (512, 512)).astype(np.float32)
    assert np.abs(read_disp(write_disp(x)) - x).max() < 2 / 65535

def test_generator_output(gen, batch):
    out = gen(**batch)
    assert out.shape == (batch['pos'].shape[0], 1, 512, 512)
    assert out.std(dim=0).mean() > 1e-4                # catches mode collapse

def test_ema_buffers(gen):
    ema = copy.deepcopy(gen); gen.train(); _ = gen(**batch)
    update_ema(ema, gen)
    for be, b in zip(ema.buffers(), gen.buffers()):
        assert torch.allclose(be, b)                   # catches the BN-buffer bug

def test_identity_score_requires_render(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_identity_score(str(tmp_path/'m.obj'), str(tmp_path/'p.jpg'))
```

### Phase 0 gate

- [ ] `pytest` runs, and **at least three tests fail** — if everything passes, the fallbacks are still hiding things
- [ ] The pipeline raises a clear error when FLAME, InsightFace, or MICA weights are absent
- [ ] `compute_identity_score` raises without a render
- [ ] The same-person check demonstrably rejects two photos of different people
- [ ] Every fallback path that remains stamps `"degraded": true` into the manifest

---

## Phase 1 — Inference-only baseline (Weeks 1–2)

**Goal:** 3–5 photos in, a `.obj` with the correct identity out. No training.

**Gate:** 5 test subjects produce 5 **measurably distinct** meshes, each scoring above your calibrated threshold against a **real render**.

### Step 1 — Vendor MICA properly

The old guide's loader could not work:
```python
from micalib.models import MICA
model = MICA(config=None)          # MICA needs a config object and a FLAME instance
beta = model(img)                  # not the forward signature
```
Both failures are swallowed by a `try/except` that falls back to `np.zeros(300)` — which decodes to the **mean FLAME face**. Every subject gets the identical mesh, and (before Phase 0) the gate still passed.

```bash
git submodule add https://github.com/Zielon/MICA vendor/MICA
cd vendor/MICA && git checkout <PINNED_COMMIT> && cd -
```

Pin the commit. Write a thin adapter against MICA's *actual* API rather than guessing at it, and let construction failures propagate.

### Step 2 — Fix MICA's input preprocessing

Two independent errors in `encode_single`:

```python
img = torch.from_numpy(crop_112).permute(2,0,1).float() / 255.0   # both wrong
```

- `face_align.norm_crop` returns **BGR** (OpenCV convention); ArcFace backbones expect **RGB**.
- ArcFace normalisation is `(x - 127.5) / 127.5` → `[-1, 1]`, not `x / 255` → `[0, 1]`.

A colour-swapped, mis-scaled tensor produces a plausible face that is not the right person — the most expensive kind of wrong, because it looks fine.

```python
img = crop_112[:, :, ::-1].copy()                 # BGR → RGB
img = torch.from_numpy(img).permute(2, 0, 1).float()
img = ((img - 127.5) / 127.5).unsqueeze(0).to(self.device)
```

### Step 3 — Fix multi-view fusion (it is currently a flat average)

The old guide was emphatic that flat-averaging betas degrades accuracy and prescribed softmax over `det_score`. But `det_score` lives in roughly **0.55–0.90**, and softmax over that range is nearly uniform:

```
det_score 0.90 vs 0.55  →  weights 0.587 / 0.413
```

That is a flat average to within 9%. The failure mode the guide warns about is not prevented. And `det_score` measures *detectability*, not frontality — a crisp profile often outscores a soft frontal.

**Do this instead — fuse in embedding space, which is what MICA itself does.** MICA is an ArcFace backbone plus a small regressor. ArcFace embeddings are *designed* to be averaged; 300-D PCA coefficients are not — the mean of two plausible faces in PCA space is not necessarily a plausible face.

```python
def encode_multiview(self, detections):
    """Average ArcFace embeddings across views, then regress once.
    Strictly better than averaging betas: the embedding space is built for this."""
    dets = [d for d in detections if d is not None]
    if not dets:
        raise ValueError("No valid detections for multi-view fusion.")

    embs, weights = [], []
    for d in dets:
        embs.append(self.arcface_backbone(self._preprocess(d.crop_112)))
        # weight by frontality, not detectability
        weights.append(d.det_score * max(np.cos(np.radians(d.yaw_deg)), 0.0) ** 2)

    w = np.asarray(weights, np.float32); w /= w.sum()
    fused = sum(wi * e for wi, e in zip(w, embs))
    fused = F.normalize(fused, dim=-1)              # re-normalise after averaging
    return self.regressor(fused).squeeze(0).cpu().numpy()
```

If you cannot reach inside the checkpoint to split backbone from regressor, fall back to β-space averaging but weight by `det_score · cos(yaw)²` and put the exponent in the config. Do **not** ship the softmax-over-det_score version — it does not do what its docstring claims.

### Step 4 — Decide what Stage 2 actually is

`src/stage2_expression/encoder.py` has two problems:

**Loading.** `torch.load(ckpt)` returns an `OrderedDict`, not a module. `hasattr(model,'eval')` is False, so a dict gets returned and `self.model(img)` raises `TypeError: 'OrderedDict' object is not callable` — outside the try/except, so it propagates as a confusing error. SMIRK's own demo builds the encoder first, then filters the checkpoint for `smirk_encoder.` keys and strips the prefix, because the checkpoint holds both encoder and generator. Do the same.

**Capability.** `output.get('detail', zeros)` assumes a detail branch. **SMIRK has none** — it predicts shape, expression, pose and camera. The detail displacement head is a **DECA/EMOCA** feature. `Research.md` says "DECA/EMOCA/HRN/SMIRK-style"; the old guide collapsed that to "SMIRK" while keeping the detail promise. So `coarse_detail` is permanently `zeros((128,128))`, saved to disk, recorded in the manifest, and used by nothing.

Pick one, explicitly:

| Option | Do this | Cost |
|---|---|---|
| **(a) Recommended** — Stage 2 is expression-only | Delete `coarse_detail` from code, config, manifest. Stage 3 produces all displacement. | None. Stage 3 was designed for this anyway, and a 128² coarse map feeding a 512² generator adds little. |
| (b) Keep a coarse detail branch | Add EMOCA back | EMOCA's non-commercial licence, plus a second checkpoint to maintain |

Also update `configs/default.yaml` — the comment `model: smirk  # Permissively licensed expression extractor` is not accurate. SMIRK's install pulls FLAME (registration required) and uses EMOCA's emotion model (registration and licence agreement required). It is not a licensing escape hatch.

### Step 5 — Phase 1 notebook

```python
# Cell 1 — environment (then RESTART)
from kaggle_secrets import UserSecretsClient
import subprocess
tok = UserSecretsClient().get_secret("github_pat").strip()
subprocess.run(["git","clone",
    f"https://x-access-token:{tok}@github.com/youruser/face-geo-pipeline.git",
    "/kaggle/working/pipeline"], check=True)
!pip install -q -r /kaggle/working/pipeline/requirements.txt
!pip install -q setuptools wheel ninja
!pip install -q git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

```python
# Cell 2 — Day-0 verification, every single run, no exceptions
import sys; sys.path.insert(0, '/kaggle/working/pipeline')
import numpy as np
from src.utils.flame_model import FLAMEModel
flame = FLAMEModel('/kaggle/working/pipeline/data/flame_model/flame2023_Open.pkl')
v, f = flame.decode_neutral(np.zeros(300))
vp, _ = flame.decode(np.zeros(300), np.zeros(100), np.zeros(15))
assert v.shape == (5023,3) and not np.isnan(v).any()
assert np.abs(vp - v).max() < 1e-6
print("FLAME OK")
```

```python
# Cell 3 — weights must be present. No fallbacks.
import os
for p in ['/kaggle/input/mica-pretrained/mica.tar']:
    assert os.path.exists(p), f"Missing {p}. Attach as a Kaggle Dataset input."
```

```python
# Cell 4 — run on 5 subjects and assert they differ
from src.pipeline import FaceGeoPipeline
pipe = FaceGeoPipeline('/kaggle/working/pipeline/configs/default.yaml',
                       '/kaggle/working/models')
betas = {}
for subj in SUBJECTS:                        # each is a list of 3–5 photo paths
    r = pipe.run(SUBJECTS[subj], f'/kaggle/working/outputs/phase1/{subj}/')
    betas[subj] = np.load(r['beta_path'])

import itertools
for a, b in itertools.combinations(betas, 2):
    d = float(np.linalg.norm(betas[a] - betas[b]))
    print(f"  ||beta_{a} - beta_{b}|| = {d:.4f}")
    assert d > 1e-3, f"{a} and {b} produced the same identity — MICA is not loading"
```

That last assertion is the one that matters. It is the check that catches the `np.zeros(300)` fallback, and it is worth more than the identity score.

```python
# Cell 5 — identity score, against a real render
from evaluation.identity_score import compute_identity_score, render_neutral_preview
for subj in SUBJECTS:
    mesh = f'/kaggle/working/outputs/phase1/{subj}/head_mesh.obj'
    render_neutral_preview(mesh, mesh.replace('.obj', '.png'))
    s = compute_identity_score(mesh, SUBJECTS[subj][0])
    print(f"{subj}: {s:.3f}")
```

### Phase 1 gate

- [ ] 5 subjects → 5 meshes with pairwise `‖β₁−β₂‖ > 1e-3` *(catches the mean-face fallback)*
- [ ] Each mesh has exactly 5023 verts, no NaN
- [ ] Every mesh has a rendered `.png` beside it
- [ ] Identity scores recorded, and a threshold **calibrated from this run**, not assumed
- [ ] Validation correctly rejects: no face; two different people; all photos same angle
- [ ] `manifest.json` records model versions, FLAME version, and `degraded: false`
- [ ] `pytest` green

---

## Phase 2 — Fine-tune identity (Weeks 3–4)

**Goal:** MICA fine-tuned on your target demographic.
**Gate:** NoW median error improves over pretrained MICA; ArcFace similarity improves on 20 held-out photos.

Unchanged from the previous guide, with three corrections:

**1. The `%%writefile` + `!torchrun` pattern is mandatory.** You cannot run `dist.init_process_group()` inline in a Jupyter cell; it hangs. This was right in the old guide and remains right.

**2. fp16, not bf16.** T4 is Turing (sm_75) and has no bf16 support. Correct in the old guide.

**3. Deprecated AMP API.** `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` are deprecated in torch ≥2.4 and warn *every step*, burying your actual logs:

```python
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')
with autocast('cuda', dtype=torch.float16):
    ...
```

**4. Quota.** Fine-tuning 30k steps on 2×T4 is roughly 6–10 wall-clock hours = 12–20 quota-hours. That is most of one week. Plan resume-from-checkpoint from the start, not as an emergency measure.

### Phase 2 gate

- [ ] NoW median error < pretrained MICA baseline (measure the baseline yourself; don't assume 0.90mm)
- [ ] ArcFace similarity improves on 20 held-out photos vs. Phase 1
- [ ] Checkpoint uploaded to Kaggle Models with the FLAME version in the version notes
- [ ] Quota ledger updated

---

## Phase 2.5 — Build the UV displacement dataset (Weeks 4–5)

**This is a new phase. The previous guide treated it as a one-command step with a stub script behind it.**

`scripts/build_uv_displacement_dataset.py` currently **never opens a scan, never calls its own rasteriser, and never writes a single map.** It writes `normalization_stats.json` containing a hardcoded `p99 = 1.85` labelled "Typical", prints "Dataset build structure prepared", and exits. That fabricated constant is then the denormalisation factor for the entire displacement pipeline, and the thing `upload_to_kaggle_models.py` gates uploads on.

The old guide told you to run it, budget 4–8 hours, and then verify output files that were never created.

**Four real problems to solve before this works:**

### 1. Correspondence — `scan - flame` assumes something untrue

```python
diff = scan_vertices - flame_vertices
```
FaceScape scans are **not in FLAME topology**. There is no vertex-to-vertex correspondence, so this either raises on shape mismatch or produces meaningless numbers.

Two ways to fix it:
- **Fit-then-subtract:** run FLAME fitting per scan (landmarks + ICP), then subtract. Slower, and the fit quality becomes an error source in your ground truth.
- **Ray-cast (recommended):** for each FLAME vertex, cast along its normal and take the first intersection with the scan surface. Signed distance along the normal *is* the displacement, by definition. `trimesh.ray` does this; it's already in requirements.

```python
import trimesh
scan = trimesh.load(scan_path)
rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(scan)   # pip install embreex
locs, ray_idx, _ = rmi.intersects_location(
    ray_origins=flame_v - normals * 0.01,                 # start 10mm inside
    ray_directions=normals, multiple_hits=False)
disp = np.zeros(len(flame_v), np.float32)
disp[ray_idx] = np.einsum('ij,ij->i', locs - flame_v[ray_idx], normals[ray_idx])
hit_mask = np.zeros(len(flame_v), bool); hit_mask[ray_idx] = True
```
Keep `hit_mask` — vertices with no intersection must be excluded from the validity mask, or you train on zeros that aren't zeros.

### 2. Point splatting is not rasterisation

```python
disp_map[py, px] = disp_scalar          # 5023 points into a 512² canvas
mask_map = cv2.dilate(mask_map, kernel, iterations=2)
```
That's ~5023 pixels out of 262,144 = **1.9% coverage**, not the "~60-70%" the old guide's verification asserted. Dilating by 2 doesn't bridge it.

You need barycentric rasterisation over the UV **triangles**. You're already installing nvdiffrast — use it (rasterise UV-space triangles, interpolate per-vertex displacement with `dr.interpolate`). On a CPU-only session, a scanline loop over 9976 triangles works fine and costs zero GPU quota, which matters here.

### 3. FLAME's `.pkl` has no UV coordinates

`uv_coords` and `uv_faces` are parameters with no source. They ship separately in `FLAME_texture.npz` / `head_template.obj`. Fetch them in Day 0 Step 5. Note also that the FLAME UV layout has **seams**; displacement must be continuous across them or you'll get visible ridges. Dilate the valid region by a few pixels past the mask boundary so bilinear sampling at the seam doesn't pull in zeros.

### 4. The p99 must be measured, not asserted

```python
all_disp = np.concatenate([d[m] for d, m in zip(displacements, masks)])
p99 = float(np.percentile(np.abs(all_disp), 99))
print(f"Measured p99 over {len(all_disp):,} valid samples: {p99:.4f} mm")
```
Compute it across the whole corpus in a first pass, then normalise in a second. If your measured p99 lands anywhere near 1.85mm, that is a coincidence, not a validation.

### 5. Define the storage contract, and enforce it both ways

This is the bug that makes Stage 3 untrainable. Currently:
```python
disp = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)   # 0..255
...
return self.out_act(out)                                           # tanh → [-1,1]
diff = (pred - target).abs() * mask                                # comparing [-1,1] to [0,255]
```
With `λ_recon = 100`, the generator's entire gradient signal is "increase output by ~128", which tanh cannot do. It saturates at 1.0 and stops. Nothing recovers.

Also: displacement is **signed** — wrinkle valleys are negative — and 8-bit unsigned PNG cannot represent that.

```python
# WRITE
disp_norm = np.clip(disp_mm / p99_mm, -1.0, 1.0)                  # [-1, 1]
cv2.imwrite(f"{stem}_disp.png",
            ((disp_norm + 1.0) * 0.5 * 65535).astype(np.uint16))   # 16-bit

# READ
raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
assert raw.dtype == np.uint16, f"expected 16-bit, got {raw.dtype}"
disp = (raw.astype(np.float32) / 65535.0) * 2.0 - 1.0             # [-1,1], matches tanh

# INFERENCE
disp_mm = generator_output * p99_mm
```
Add the round-trip test from Phase 0. Honestly, `.npy` float16 is simpler than 16-bit PNG and skips the quantisation argument entirely — the only reason for PNG is compression, and float16 npy + zip gets you most of that.

### 6. Vectorise the normals

The per-face Python loop is ~200× slower than `np.add.at`. At 18,760 scans that is days versus hours. `FLAMEModel.vertex_normals()` in the rewritten loader does it correctly.

### Phase 2.5 gate

- [ ] Script actually writes `*_disp.png`, `*_pos.png`, `*_norm.png`, `*_mask.png` per scan
- [ ] Mask coverage is **measured and reported** — whatever it is, not whatever the guide guessed
- [ ] `p99` is measured over the corpus and printed with its sample count
- [ ] Round-trip test passes to within 1/65535
- [ ] Visual inspection of 10 random samples: displacement maps show recognisable wrinkle structure, not noise
- [ ] Ran on a **CPU-only** session (zero GPU quota)
- [ ] `normalization_stats.json` records p99, resolution, FLAME version, scan count, and date

---

## Phase 3 — Detail GAN (Weeks 5–9)

**Gate:** visible wrinkles on held-out subjects; Chamfer improves over Stage 1+2; identity does not degrade.

The architecture is sound. There are four bugs that individually prevent convergence, and none of them announce themselves — training runs, loss curves look plausible, output is garbage.

### Bug 1 — R1 wipes the discriminator's adversarial gradient

```python
scaler_d.scale(d_loss).backward()      # accumulate adversarial gradients
if step % 16 == 0:
    opt_d.zero_grad()                  # ← throws them away
    r1_loss = r1_gradient_penalty(...)
```
Every 16th step, D updates on the R1 penalty **alone**. R1 pushes gradient norm toward zero, so on its own it drives D toward a constant function. One step in sixteen actively un-trains the discriminator.

```python
scaler_d.scale(d_loss).backward()
if step % 16 == 0:                     # lazy regularisation = accumulate, don't reset
    r1 = r1_gradient_penalty(d_raw, real_disp, gamma=HYPERPARAMS['r1_gamma'])
    scaler_d.scale(r1 * 16.0).backward()
scaler_d.step(opt_d); scaler_d.update()
```

### Bug 2 — Cross-attention destroys the bottleneck

```python
q = self.mv_attn(q, per_view_feats)              # no residual
bottleneck = q.permute(0,2,1).reshape(B,C,H,W)
```
`MultiheadAttention` returns a weighted combination of the **values** — here, 3 pooled per-view vectors. So all 256 spatial positions get overwritten by a convex combination of 3 global vectors. Every bit of spatial structure at the bottleneck is destroyed, and the AdaIN modulation applied one line earlier is discarded with it. The only information reaching the decoder is via skip connections.

```python
# in __init__
self.norm_q = nn.LayerNorm(bottleneck_c)
# in forward
q = q + self.mv_attn(self.norm_q(q), per_view_feats)     # residual preserves content
```
Two lines. They decide whether Stage 3 can work at all.

### Bug 3 — EMA drops BatchNorm buffers

```python
for p_ema, p in zip(ema_model.parameters(), model.parameters()):
```
`parameters()` excludes buffers. BatchNorm's `running_mean`/`running_var` are buffers, so `ema_generator.pt` carries EMA'd weights beside statistics frozen at `deepcopy` time — step 0, `mean=0, var=1`. Those are exactly the stats used in `eval()` mode at inference.

This matters more than usual because `upload_to_kaggle_models.py` **hard-blocks uploading anything except the EMA checkpoint**. The only artefact you're permitted to ship is the broken one.

```python
def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for pe, p in zip(ema_model.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for be, b in zip(ema_model.buffers(), model.buffers()):
            be.data.copy_(b.data)                        # buffers copied, not averaged
```

**Better: remove BatchNorm entirely.** In an adversarial generator at batch 4 under DDP, BN is wrong three ways over — it couples samples within a batch (a classic GAN artefact source), the statistics are very noisy at batch 4, and plain `BatchNorm2d` under DDP normalises per-GPU so your effective batch is 4, not 8.

```python
nn.InstanceNorm2d(out_c, affine=True)      # or nn.GroupNorm(8, out_c)
```
Batch-independent, standard for image-to-image GANs, and it makes the buffer bug moot. If you keep BN, wrap with `SyncBatchNorm.convert_sync_batchnorm()` before the DDP wrap.

### Bug 4 — The unit mismatch

Covered in Phase 2.5 item 5. Fix the dataset contract, not the loss.

### Also fix

**Subject leakage in the train/val split.** `self.disp_files[:split_idx]` splits a sorted filename list, so the same person's neutral trains while their smile validates. Your validation loss will be optimistic.

```python
subjects = sorted({p.stem.split('_')[0] for p in self.disp_files})
rng = random.Random(1337); rng.shuffle(subjects)
val = set(subjects[:max(1, len(subjects)//10)])
self.files = [p for p in self.disp_files if (p.stem.split('_')[0] in val) != is_train]
```

**`identity_preservation_loss` is dead code.** Defined in `losses.py`, documented as a headline loss with `λ=5.0`, never imported, never called — the trainer's import list omits it. Either wire it up (it needs a differentiable renderer in the loop, which is a real cost) or delete it and remove `id_loss_lambda` from the hyperparameters. Right now the guide promises a loss the code doesn't compute.

**Config duplication.** `configs/default.yaml:stage3` and `trainer.py:HYPERPARAMS` hold the same values and already disagree. Load the yaml in the trainer; delete the dict.

**`num_workers=2` × 2 GPUs on 4 vCPUs.** Each batch reads 4×(512² PNG × 4 maps). The dataloader will be your bottleneck, not the GPU. Pre-pack to a single memory-mapped `.npy` per split, or use webdataset shards. Measure it: if GPU utilisation sits below 70%, you're paying quota for I/O.

### Training health monitoring

Track every 100 steps. These thresholds are starting points — replace them with values from your own first successful run.

| Metric | Healthy | Alarm | Likely cause |
|---|---|---|---|
| D accuracy on real | 0.5–0.8 | > 0.95 | D saturated → lower D lr, raise R1 γ |
| D accuracy on fake | 0.2–0.5 | < 0.05 | G winning → more D steps |
| G loss | Decreasing | Flat 5k+ steps | Mode collapse → raise λ_recon |
| **Output std across batch** | > 0.01 | < 0.01 | Mode collapse — **the single most useful signal** |
| NaN | — | any | R1 in fp16, or the unit mismatch |
| Masked recon loss | > 0 | **exactly 0.0** | You're training on the zero-tensor fallback |

That last row exists because of the `data.py` fallback. A recon loss of exactly 0.0 is not convergence.

### Phase 3 gate

- [ ] Prove it on **50 subjects with a fixed seed** before spending quota on 938
- [ ] Displacement maps show visible, **subject-varying** wrinkles on 10 held-out subjects
- [ ] Output std across batch > 0.01 throughout training
- [ ] Chamfer improves over Stage 1+2 baseline
- [ ] ArcFace drops no more than 0.05 from Phase 2
- [ ] EMA checkpoint verified: `ema.buffers()` matches `model.buffers()`
- [ ] `normalization_stats.json` has a **measured** p99
- [ ] Quota ledger updated

---

## Phase 4 — Licensing (Weeks 1–10, starts NOW)

**Moved from week 8 to week 1.** This is a calendar dependency, not an effort dependency.

The old guide reduced this to one email to Nanjing University about FaceScape. The actual exposure is the whole stack, and almost every layer is non-commercial:

| Component | Licence | Blocks commercial ship? |
|---|---|---|
| FLAME 2020 *(currently pinned)* | Non-commercial research | **Yes** |
| **FLAME 2023 Open** | CC-BY-4.0 (released 11/2025) | **No — migrate to this** |
| MICA | Non-commercial research | Yes |
| MICA unified dataset (LYHM / FaceWarehouse / Stirling) | Mixed, per-component | Yes, individually |
| SMIRK | Research; pulls FLAME (registration) and EMOCA's emotion model (licence agreement) | Yes |
| EMOCA | Non-commercial | Yes |
| FaceScape | Explicit no-commercial-use | Yes |
| nvdiffrast | NVIDIA Source Code License | Check |
| InsightFace / ArcFace weights | Research-licensed | Check |
| ICT-FaceKit | MIT | No |

**The good news the old guide missed entirely:** MPI released **FLAME 2023 Open under CC-BY-4.0** in November 2025, with conversion code for translating expression parameters from FLAME 2023 to 2023 Open. That removes the foundation-layer blocker. The config pins FLAME2020 — the non-commercial one — for no stated reason.

**Actions, in order:**

1. **Migrate to FLAME 2023 Open now,** before any weights exist. Retraining on a different basis later invalidates every checkpoint.
2. **Audit every dependency** into the table above with a link to the actual licence text and the date you read it. Terms change.
3. **Treat MICA/SMIRK/EMOCA as teachers, not shipped components** — use them to generate supervision for your own regressor trained from scratch on clean data. Whether this satisfies the licences is genuinely contested and depends on their specific terms. Get an actual lawyer before betting the project on it.
4. **FaceScape validates the architecture only.** Never train shipped weights on it. `Research.md` §7 already says this; keep it that way.
5. **Start the own-capture track this week.** Consent forms, releases, a rig, subject recruitment — none of it compresses. 30 well-captured subjects with clean paperwork beats 938 you cannot ship. This is the long pole.
6. Email the FaceScape authors (nju3dv@gmail.com) about commercial terms. Do it in week 1 so the reply arrives while you still have options.

I am not a lawyer, and I'm summarising publicly available licence summaries. Verify each one directly before any commercial commitment.

---

## Phase 5 — Retopology and rig (Weeks 5–12) — the real critical path

**This phase did not meaningfully exist in the previous guide.** It got three weeks and one page, and the page was about installing Blender.

`configs/default.yaml` declares:
```yaml
stage5:
  topology: ict_facekit
  lod_triangle_counts: [24500, 5000, 2000, 500]
  blendshape_standard: arkit52
```
**None of this exists.** `src/stage5_export/` contains an empty `__init__.py`. What the pipeline actually exports is FLAME topology — 5023 verts, 9976 triangles. The largest declared LOD (24,500 tris) is 2.5× more triangles than the mesh has.

`Research.md` is right about this: *"Stage 5 (retopology) is a solved, mostly non-ML geometry-processing problem — doing it right is what actually makes this 'production grade,' not the ML."* It is also the part nobody has started.

Good news: it is entirely independent of the ML. Start it in week 5 and run it in parallel. It costs **zero GPU quota** — all of it is CPU work.

### 5.1 — FLAME → ICT-FaceKit correspondence

A one-time dense mapping: for each ICT vertex, the barycentric coordinates of its position on the FLAME surface.

```python
# Computed once, offline, by non-rigid ICP between the two templates.
# Output: W, a sparse (n_ict, 5023) matrix.
# Then forever after:  v_ict = W @ v_flame     — microseconds.
```

Method: align the two templates by landmarks, run non-rigid ICP (or optimal-step NICP), then for each deformed ICT vertex find the closest point on the FLAME surface and record its triangle + barycentrics. Validate by round-tripping the FLAME template through `W` and measuring per-vertex error — should be well under 1mm in the face region.

This is **several days of careful work** and it gates everything downstream. Do it first.

### 5.2 — ARKit-52 blendshapes

FLAME's expression basis is 100 PCA components. ARKit's is 52 **semantic** shapes (`jawOpen`, `browInnerUp`, `mouthSmileLeft`, …). These are different bases and the mapping exists in neither model.

**Recommended:** ICT-FaceKit ships its own ARKit-compatible blendshape set and is **MIT licensed** — the only fully clean component in your stack. Take ICT's generic blendshape deltas and transfer them onto each subject's identity via deformation transfer (Sumner & Popović), or more simply by applying the generic deltas in each vertex's local frame.

**Alternative:** solve a least-squares map from FLAME ψ to ARKit weights against a set of posed examples. Cheaper to implement, worse to animate, and it inherits FLAME's licence.

### 5.3 — Deformation transfer

Generic blendshapes on a specific identity look wrong — a `jawOpen` tuned for the mean face over-opens a narrow jaw. Deformation transfer fixes this. Standard, well-documented, a few hundred lines.

### 5.4 — LODs that preserve shape keys

**Blender's Decimate modifier does not carry shape keys across.** This is a known and genuinely annoying problem, and the previous guide's `lod_triangle_counts` line implies it's a config value.

Approach: decimate the *neutral* mesh to each LOD, then re-project the shape keys onto the decimated topology using the same barycentric machinery from 5.1. Validate each LOD by animating it against the LOD0 and measuring maximum vertex deviation.

### 5.5 — Skeleton

`blender_export.py` exports `object_types={'MESH','ARMATURE'}` with `use_armature_deform_only=True` — but **no armature exists in the scene**. UE5 Live Link needs a head/jaw/eye joint hierarchy with correct orientations and skin weights.

FLAME already has 5 joints (global, neck, jaw, eyeL, eyeR) with skinning weights in `data['weights']`. Transfer them through the same correspondence matrix and you have most of a rig.

### 5.6 — Fix `blender_export.py`

```python
kb.data[idx].co += d          # TypeError: d is a Python list from JSON
```
`bpy` vector maths needs `mathutils.Vector`. And `+=` adds to the basis, which is correct only if the JSON stores *offsets* — the format is undefined because **nothing in the repo writes that file**.

```python
from mathutils import Vector
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_obj

kb = mesh_obj.shape_key_add(name=bs_name, from_mix=False)
for idx, d in enumerate(deltas):
    if idx < len(kb.data):
        kb.data[idx].co = kb.data[idx].co + Vector(d)      # deltas, explicit
```

Define and document the `blendshapes.json` schema before writing the consumer:
```json
{"jawOpen": [[dx, dy, dz], ...],  "browInnerUp": [[dx, dy, dz], ...]}
```
Units and coordinate convention stated in the file header. Length must equal the target topology's vertex count.

### Phase 5 gate

- [ ] `W` correspondence matrix built; round-trip error < 1mm in the face region
- [ ] 52 named ARKit shapes present on the exported mesh
- [ ] Deformation transfer validated on 5 visually distinct identities
- [ ] 4 LODs generated, each retaining all 52 shape keys
- [ ] Armature present with correct joint orientations
- [ ] FBX imports into UE5 with no errors
- [ ] Live Link drives the face; `jawOpen` opens the jaw and nothing else moves
- [ ] All of it ran on **CPU-only** sessions

---

## Phase 6 — Hybrid detail + facial hair (Weeks 10–14)

Unchanged from the previous guide in substance. One caution: the "train diffusion then sharpen adversarially" plan is the right target, but it assumes Phase 3 converged. Do not start it until Phase 3's gate passes — otherwise you're layering a second unvalidated model on an unvalidated one and you won't be able to attribute failures.

Facial hair: run the 20-subject check the old guide describes. Note that if the answer is "no", Approach 2 (geometry cards) is a multi-week addition that isn't budgeted anywhere.

---

## Phase 7 — Production export (Weeks 12–16)

CPU-only. Zero GPU quota.

```bash
!wget -q https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz
!tar -xf blender-4.1.0-linux-x64.tar.xz -C /kaggle/working/
!ln -sf /kaggle/working/blender-4.1.0-linux-x64/blender /usr/local/bin/blender
!blender --version
```

Correct in the old guide: do **not** `pip install bpy` — version locking against your export script is a recurring source of breakage. The portable binary pins cleanly.

Extend `manifest.json` with the fields the old guide's `generate_manifest` had but `pipeline.py` never wrote — `run_id`, `generated_at`, `model_versions`, `displacement_p99_mm`, `lod_triangle_counts`, plus `flame_version` and `degraded`.

---

## Phase 8 — Evaluation (ongoing)

Track after every phase. **Fill these in by measuring, not by copying targets from a plan.**

| Phase | NoW median (mm) | ArcFace sim | Chamfer vs GT (mm) | Notes |
|---|---|---|---|---|
| 1 baseline | measure | measure | — | Pretrained MICA |
| 2 fine-tuned | target < phase 1 | target > phase 1 | — | |
| 3 GAN | — | ≥ phase 2 − 0.05 | measure | Detail added |
| 6 hybrid | — | ≥ phase 2 − 0.05 | target < Meshy | Best detail |

Two notes on the old table:
- The 0.90mm "pretrained MICA baseline" was quoted as fact. **Measure your own** — it depends on your preprocessing, your FLAME version, and your NoW split.
- Meshy's 81.0/79.7/59.8 figures are **vendor self-reported**, as `Research.md` correctly flags. Aim past them; don't treat them as certified.

---

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'exprdirs'` | FLAME packs expression into `shapedirs[..., 300:400]` | Use the rewritten loader |
| `UnpicklingError: NEWOBJ class argument must be a type` | `find_class → np.array` doesn't work | numpy-alias shim + real chumpy |
| `ERROR: No matching distribution for nvdiffrast` | Not on PyPI | Install from git |
| FLAME range looks 1000× wrong | FLAME is metres, not mm | `scale_to_mm`, consistently |
| **All 5 subjects produce the same mesh** | MICA failed to load; `zeros(300)` = mean face | Remove the fallback; assert β distinctness |
| **Identity score is exactly 1.000** | Comparing the photo to itself | Render the mesh; delete the fallback |
| **Recon loss is exactly 0.0** | Training on the zero-tensor dataset fallback | Restore the `raise` in `data.py` |
| Generator output saturates at ±1 | Displacement unit mismatch ([0,255] vs [-1,1]) | Fix the storage contract |
| GAN never converges, no error | R1 `zero_grad()` wiping adversarial gradients | Accumulate, don't reset |
| Detail is blurry / identity-free | Cross-attention has no residual | `q = q + attn(norm(q), feats)` |
| EMA checkpoint worse than raw | BN buffers not copied | Copy buffers, or drop BN |
| Validation never rejects anything | `get_model('buffalo_l')` raises into a bare `except: pass` | Use `FaceAnalysis` |
| DDP hangs at `init_process_group` | Distributed code inline in a cell | `%%writefile` + `!torchrun` |
| Deprecation warnings drown the logs | `torch.cuda.amp.*` | `torch.amp.autocast('cuda')` |
| Out of GPU quota mid-week | 2×T4 burns 2 quota-h per wall-clock h | Budget ~15 wall-clock h/week |
| SSH key rejected on Kaggle | Secrets strip the trailing newline | `key.rstrip() + "\n"`, or use a PAT |
| FBX has no blendshapes | Nothing generates `blendshapes.json` | Phase 5.2 |
| Wrong blendshape behaviour in UE5 | Expression baked into the base mesh | Verify `decode_neutral` is what's exported |

---

## Weekly checkpoint

**Week 1:** Does `pytest` fail? If everything passes, the silent fallbacks are still hiding things. Has the FLAME licence decision been made and recorded?

**Week 2:** Do 5 subjects give 5 meshes with `‖β₁−β₂‖ > 1e-3`? If they're identical, MICA isn't loading and the `zeros(300)` fallback is returning the mean face.

**Week 4:** Has the UV dataset script written real files with a *measured* p99? If it printed "Dataset build structure prepared", it did nothing.

**Week 6:** Do displacement maps differ across subjects? Is output std across batch above 0.01? Is the recon loss non-zero?

**Week 8:** Does `W @ v_flame` round-trip the ICT template to under 1mm? This gates everything downstream and nothing about it depends on the ML.

**Week 10:** Have you heard back on licensing? Is the own-capture pipeline capturing? If both are still pending, the ML progress may not be shippable regardless of quality.

**Week 14:** Does the FBX import into UE5 and animate under Live Link? If the blendshapes are wrong, check the base mesh is truly neutral before blaming the rig.

**Week 16:** Is surface-detail Chamfer against ground-truth scans lower than Meshy 7.1's on the same photos, using weights trained on data you can legally ship? Both halves of that sentence have to be true.