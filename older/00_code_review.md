# Code Review — Face Geometry Pipeline

Reviewed: full repo at commit-zero (`git log` shows no commits yet — everything below is uncommitted working tree).

**Overall assessment.** `Research.md` and the three `DOCS/0x_*.md` files are genuinely good: the analysis is honest, the licensing risk is called out, the decomposition is right, and the Meshy numbers are correctly flagged as vendor-published. The problem is that `Guide.md` and `src/` were written *as if the plan had already been executed*. They contain confident instructions ("Copy exactly", "verified", "Day 0 is complete when...") for code that has never been run. Several of those steps cannot work as written.

The most dangerous class of bug here is not the crashes — it's the **silent fallbacks**. Nearly every module degrades to zeros or to a trivial identity when its dependency is missing, and then reports success. Stacked together, the Phase 1 gate in `Guide.md` passes on a pipeline that has done nothing at all. That is fixed first, because until it is, no other measurement you take is trustworthy.

Severity key: **P0** blocks everything · **P1** produces wrong results silently · **P2** wrong but visible · **P3** quality/cleanup.

---

## P0 — Day-0 blockers (nothing runs until these are fixed)

### 1. `FLAME generic_model.pkl` has no `exprdirs` key
**Files:** `src/utils/flame_model.py:33`, `Guide.md` Step 4 (line ~349)

```python
self.exprdirs = np.array(data['exprdirs'], ...)   # KeyError
```

FLAME stores identity and expression in **one** array. `shapedirs` is `(5023, 3, 400)`: columns `0:300` are identity, `300:400` are expression. This is documented in the reference implementation — FLAME's own `hello_world.py` prints the identity component as `shapedirs[:,:,0:300]` and the expression component as `shapedirs[:,:,300:]`, and the fitting code defines the valid expression range in `betas` as indices 300–399.

**Fix:**
```python
shapedirs      = _as_array(data['shapedirs'])          # (5023, 3, 400)
self.shapedirs = shapedirs[:, :, :300]
self.exprdirs  = shapedirs[:, :, 300:400]
```
This line is the *first* thing the pipeline does. It has never executed.

### 2. The chumpy unpickler trick does not work
**Files:** `src/utils/flame_model.py:6-14`, `Guide.md` Step 4

Returning `np.array` from `find_class` fails. I reproduced it against a chumpy-shaped pickle:

```
UnpicklingError: NEWOBJ class argument must be a type, not builtin_function_or_method
```

`np.array` is a *function*, not a type; pickle's `NEWOBJ` opcode calls `cls.__new__(cls)` and there is no such attribute. The guide tells you to verify this on Day 0 and it will fail at that verification.

**Fix (primary):** re-add the numpy aliases chumpy needs, then let real chumpy do the work:
```python
for n, t in (("bool",bool),("int",int),("float",float),("object",object),("str",str),("unicode",str)):
    if n not in np.__dict__: setattr(np, n, t)
import chumpy
data = pickle.load(open(path,'rb'), encoding='latin1')
```
**Fix (fallback, no chumpy):** a stub `np.ndarray` subclass with `__setstate__` capturing the payload dict. Both paths are implemented and tested in the replacement `src/utils/flame_model.py` shipped alongside this review.

### 3. `pip install nvdiffrast` does not exist
**Files:** `requirements.txt:3`, `Guide.md` Day 0 Step 5

nvdiffrast is not published on PyPI. The official instructions are `pip install setuptools wheel ninja` followed by `pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation`. It JIT-compiles CUDA extensions on first use, so it needs `ninja` and a matching toolchain present. `pip install -r requirements.txt` currently aborts the whole install at line 3.

Also note the licence: nvdiffrast is released under the NVIDIA Source Code License, with business enquiries directed to NVIDIA Research Licensing — that belongs on the Phase 4 licensing list, not just FaceScape.

### 4. `numpy<2.0.0` pin needs a documented kernel restart
**File:** `requirements.txt:11`

Kaggle images ship numpy 2.x. Downgrading mid-session leaves already-imported extension modules linked against the wrong ABI. The guide never says to restart. Either pin at the top of cell 1 followed by an explicit restart, or drop the pin and fix the two places that actually need it.

---

## P1 — Silent-wrong (the expensive ones)

### 5. The Phase 1 acceptance gate grades itself
**Files:** `evaluation/identity_score.py:50-54`, `Guide.md` Phase 1 Cell 5

```python
if preview_path.exists(): render = cv2.imread(preview_path)
else:                     render = photo          # <-- compares the photo to itself
return evaluator.evaluate(render, photo)          # -> 1.000
```

Nothing in the repo ever writes `head_mesh.png`. So `preview_path` never exists, `render is photo`, cosine similarity is exactly 1.0, and the gate `assert score > 0.5` **can never fail**. Combined with items 6–8 below, you can pass the entire Phase 1 gate with no FLAME model, no MICA weights and no InsightFace installed.

**Fix:** delete the fallback. Render the mesh. Minimum viable version:
```python
def render_preview(vertices, faces, out_png, size=512):
    """Orthographic Lambertian shade of the neutral mesh, frontal view."""
    # nvdiffrast, or trimesh+pyrender offscreen, or a 20-line numpy z-buffer.
    ...
if not preview_path.exists():
    raise FileNotFoundError(
        f"No render at {preview_path}. Identity score is meaningless without one."
    )
```
And be honest about what the number means: ArcFace on an untextured grey shaded mesh scores far below its photo-to-photo range. A 0.5 threshold borrowed from face *verification* is not the right bar. Establish your own baseline by scoring 20 renders of known-correct meshes, then set the threshold from that distribution. Until you do, `arcface_threshold: 0.5` in the config is a number with no provenance.

### 6. `FaceDetector` fabricates a detection when InsightFace is missing
**File:** `src/stage0_preprocess/detector.py:35-47`

On import failure it returns the whole image resized to 112/224 with `det_score=0.99` and `yaw_deg=0.0`. Consequences:
- validation check 4 (`any(|yaw| < 30)`) passes because every yaw is 0.0;
- validation check 5 (angular spread) warns but never fails;
- the same-person check passes trivially;
- MICA gets an unaligned, uncropped image and returns garbage;
- the manifest records nothing unusual.

**Fix:** raise. A missing detector is an environment error, not a degraded mode.
```python
except ImportError as e:
    raise RuntimeError(
        "InsightFace is required for Stage 0. pip install insightface onnxruntime-gpu"
    ) from e
```
Keep a `--allow-degraded` flag if you want it for CI smoke tests, and have it stamp `"degraded": true` into `manifest.json`.

### 7. Pipeline emits a degenerate mesh when FLAME is missing
**File:** `src/pipeline.py:88-91`

```python
neutral_vertices = np.zeros((5023, 3)); faces = np.zeros((9976, 3), np.int32)
```
This writes an OBJ with 5023 coincident vertices and 9976 faces all referencing vertex 1. It opens in Blender. It looks like a bug in the exporter rather than a missing model. Same fix: raise, or at minimum refuse to write the OBJ.

### 8. `UVDisplacementDataset` trains on zeros
**File:** `src/stage3_detail/data.py:18-19, 26-38`

When no `*_disp.png` are found, `__len__` returns 1 and `__getitem__` returns all-zero tensors with an all-ones mask. The masked L1 loss on zeros is exactly 0, the discriminator sees constant input, and the training log prints healthy-looking numbers for hours. Note the guide's version is *correct* here (`raise RuntimeError`) — `data.py` regressed from it. Restore the raise.

### 9. Confidence-weighted fusion is a flat average in disguise
**Files:** `src/stage1_identity/inference.py:76-80`, `Guide.md` Step 7, `configs/default.yaml:17`

The guide is emphatic that flat-averaging betas degrades accuracy, and prescribes softmax over `det_score` as the fix. But InsightFace `det_score` lives in roughly 0.55–0.90. Softmax over that range:

```
det_score 0.90 vs 0.55  ->  weights 0.587 / 0.413
```

That is a flat average to within 9%. The stated failure mode — a bad profile beta dragging down a good frontal beta — is not prevented. Worse, `det_score` measures *detectability*, not *how frontal* the face is; a crisp profile shot often scores higher than a soft frontal one.

**Three fixes, in increasing order of correctness:**

1. **Temperature.** `weights = softmax(det_score / T)` with `T ≈ 0.05`. Cheap, and at least makes the weighting real. Put `fusion_temperature` in `configs/default.yaml` so it is tunable rather than hidden.
2. **Weight by pose, not by detectability.** `w_i ∝ det_score_i · cos(yaw_i)^k`, k≈2. This targets the actual failure mode.
3. **Fuse in embedding space, not in β space** — this is what MICA itself does, and it is the right answer. MICA is an ArcFace backbone plus a small regressor. Average the *ArcFace embeddings* across views (that is what ArcFace embeddings are designed for), then run the regressor once. Averaging 300-D PCA coefficients is averaging in a space where the mean of two plausible faces is not necessarily a plausible face.

Given MICA's architecture, option 3 is roughly 15 lines and strictly better. Recommend making it the default and keeping option 2 as the fallback when you cannot reach inside the checkpoint.

### 10. MICA input preprocessing is wrong on two axes
**File:** `src/stage1_identity/inference.py:53-55`

```python
img = torch.from_numpy(crop_112).permute(2,0,1).float() / 255.0
```
- `crop_112` comes from `face_align.norm_crop`, which returns **BGR** (OpenCV convention). ArcFace backbones expect RGB.
- ArcFace normalisation is `(x - 127.5) / 127.5`, giving `[-1, 1]`, not `x/255` giving `[0, 1]`.

Both wrong simultaneously means the identity code is being read from a colour-swapped, mis-scaled tensor. The mesh will be a plausible-looking face that is not the right person, and because of item 5 the score will still read 1.000.

**Fix:**
```python
img = crop_112[:, :, ::-1].copy()                    # BGR -> RGB
img = torch.from_numpy(img).permute(2, 0, 1).float()
img = (img - 127.5) / 127.5
```

### 11. `MICA(config=None)` will not construct
**File:** `src/stage1_identity/inference.py:36-40`

MICA's constructor needs a config object and a FLAME instance, and its forward is not `model(img)` — the repo exposes encode/decode entry points returning a dict. The `try/except` around it means this fails and falls back to `np.zeros(300)` — which decodes to the *mean FLAME face*. Every subject produces the identical mesh. The guide's Week-1 check ("5 distinct .obj files") is the right check, and it will fail — but only if items 5–7 are fixed first so it can fail loudly.

**Action:** vendor MICA at a pinned commit, write a thin adapter against its actual API, and add a unit test asserting two different subjects give `‖β₁ − β₂‖ > 0`.

### 12. SMIRK does not emit a detail map, and is not loaded correctly
**Files:** `src/stage2_expression/encoder.py:34-36, 55-62`, `configs/default.yaml:22`

Two separate problems:

- **Loading.** `torch.load(ckpt)` on a SMIRK checkpoint returns an `OrderedDict`, not a module. `hasattr(model,'eval')` is False, so the model is returned as a dict and `self.model(img)` raises `TypeError: 'collections.OrderedDict' object is not callable` — caught by nothing, since the try/except only wraps loading. SMIRK's own demo builds the encoder first, then filters the checkpoint for `smirk_encoder.` keys and strips that prefix, because the checkpoint holds both the encoder and the generator. You must do the same.
- **Capability.** `output.get('detail', zeros)` assumes a detail branch. SMIRK has none — it is an expression/pose/shape encoder. The detail displacement head is a **DECA/EMOCA** feature. So `coarse_detail` is permanently `zeros((128,128))`, it gets saved to `coarse_detail.npy`, referenced in the manifest, and is never used by anything. The guide's Stage 2 description ("Expression + coarse-detail regression") is inherited from `Research.md`, where it correctly says "DECA/EMOCA/HRN/SMIRK-style" — the guide collapsed that into "SMIRK" and kept the detail promise.

**Decision you need to make explicitly:** either
- (a) accept Stage 2 is expression-only, delete `coarse_detail` from the code, config and manifest, and let Stage 3 produce *all* the displacement; or
- (b) add EMOCA back for the detail branch, and take on its licence.

Option (a) is cleaner and I'd recommend it — Stage 3 is designed to do this job anyway, and a 128×128 coarse map feeding a 512×512 generator adds little.

**Also:** SMIRK requires **PyTorch3D**, not nvdiffrast. PyTorch3D is a slow, fragile build on Kaggle (~10–20 min, version-sensitive to the CUDA/torch pair). Neither `requirements.txt` nor the guide mentions it. Budget for it or extract only SMIRK's encoder weights and drop its renderer entirely — you only need the encoder at inference.

### 13. Displacement unit mismatch makes Stage 3 untrainable
**Files:** `src/stage3_detail/data.py:47`, `generator.py:126`, `losses.py:40`

```python
disp = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)   # 0..255 or 0..65535
...
return self.out_act(out)                                           # tanh -> [-1, 1]
...
diff = (pred - target).abs() * mask                                # comparing [-1,1] to [0,255]
```

The reconstruction loss is comparing incompatible ranges. With `λ_recon` starting at 100, the generator's entire gradient signal is "increase output by ~128", which tanh cannot do. It saturates at 1.0 and stops learning. Nothing recovers from this.

Separately: displacement is **signed** (wrinkle valleys are negative), and an 8-bit unsigned PNG cannot represent that without an explicit offset. And there's no denormalisation step anywhere at inference, despite `normalization_stats.json` being gated on at upload.

**Fix — define the contract once and enforce it in both directions:**
```python
# writing (build_uv_displacement_dataset.py)
disp_norm = np.clip(disp_mm / p99_mm, -1, 1)               # [-1, 1]
disp_u16  = ((disp_norm + 1.0) * 0.5 * 65535).astype(np.uint16)
cv2.imwrite(f"{stem}_disp.png", disp_u16)                  # 16-bit, lossless

# reading (data.py)
raw  = cv2.imread(p, cv2.IMREAD_UNCHANGED)
assert raw.dtype == np.uint16, f"expected 16-bit displacement, got {raw.dtype}"
disp = (raw.astype(np.float32) / 65535.0) * 2.0 - 1.0      # [-1, 1], matches tanh

# inference
disp_mm = generator_output * p99_mm                        # from normalization_stats.json
```
Add a round-trip unit test: `write(read(x)) ≈ x` to within 1/65535. Consider `.npy` float16 instead of PNG — simpler, and you skip the quantisation argument entirely.

### 14. R1 penalty wipes the discriminator's adversarial gradient
**Files:** `src/stage3_detail/trainer.py:113-119`, `Guide.md` Phase 3 Step 4

```python
scaler_d.scale(d_loss).backward()        # accumulate adversarial grads
if step % 16 == 0:
    opt_d.zero_grad()                    # <-- throws them away
    r1_loss = r1_gradient_penalty(...)
    scaler_d.scale(r1_loss * 16.0).backward()
scaler_d.step(opt_d)
```

Every 16th step, D is updated using the R1 penalty *alone*. R1 is a regulariser that pushes gradient norm toward zero; on its own it drives D toward a constant function. So 1 step in 16 actively un-trains the discriminator.

**Fix:** drop the `zero_grad()` and let the gradients accumulate, which is what lazy regularisation means:
```python
scaler_d.scale(d_loss).backward()
if step % 16 == 0:
    r1 = r1_gradient_penalty(d_raw, real_disp, gamma=HYPERPARAMS['r1_gamma'])
    scaler_d.scale(r1 * 16.0).backward()
scaler_d.step(opt_d); scaler_d.update()
```

### 15. Cross-attention destroys the bottleneck
**File:** `src/stage3_detail/generator.py:110-113`

```python
q = bottleneck.reshape(B, C, H*W).permute(0,2,1)
q = self.mv_attn(q, per_view_feats)      # output = attention over 3 tokens
bottleneck = q.permute(0,2,1).reshape(B, C, H, W)
```

`MultiheadAttention` returns a weighted combination of the **values** — and the values here are 3 pooled per-view vectors. So every one of the 256 spatial positions in the bottleneck is overwritten by a convex combination of 3 global vectors. All spatial structure at the bottleneck is destroyed, and the AdaIN modulation applied on the previous line is thrown away with it. The only information reaching the decoder is via skip connections; the bottleneck has become a 3-token broadcast.

**Fix — residual + norm, as in every standard transformer block:**
```python
attn_out = self.mv_attn(self.norm_q(q), per_view_feats)
q = q + attn_out                               # residual: preserve spatial content
bottleneck = q.permute(0,2,1).reshape(B, C, H, W)
```
Add `self.norm_q = nn.LayerNorm(bottleneck_c)` in `__init__`. This is a two-line change that decides whether Stage 3 can work at all.

### 16. EMA checkpoint drops BatchNorm buffers
**File:** `src/stage3_detail/trainer.py:33-36`

```python
for p_ema, p in zip(ema_model.parameters(), model.parameters()):
```
`parameters()` excludes buffers. The generator is full of `nn.BatchNorm2d`, whose `running_mean` / `running_var` are buffers. So `ema_generator.pt` carries EMA'd weights alongside the `running_stats` frozen at `deepcopy` time — step 0, i.e. `mean=0, var=1`. At inference in `eval()` mode those stats are used. The output will be wrong.

This matters more than usual because `upload_to_kaggle_models.py` *hard-blocks* uploading anything but the EMA checkpoint. The one artefact you are allowed to ship is the broken one.

**Fix:**
```python
def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for pe, p in zip(ema_model.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for be, b in zip(ema_model.buffers(), model.buffers()):
            be.data.copy_(b.data)          # buffers are copied, not averaged
```
**Better:** stop using BatchNorm in a GAN generator entirely — see item 20.

### 17. `build_uv_displacement_dataset.py` is a stub that reports success
**File:** `scripts/build_uv_displacement_dataset.py`

`main()` never opens a scan, never calls `compute_displacement_and_maps`, never writes a single map. It writes `normalization_stats.json` with a **hardcoded** `p99 = 1.85` labelled "Typical" and prints "Dataset build structure prepared." The guide then instructs you to run it, budget "4–8 hours" for it, and verify its output — verification code that will fail on a missing file.

That hardcoded p99 is then the denormalisation constant for the entire displacement pipeline and the thing `upload_to_kaggle_models.py` gates on. It is a made-up number wearing a validation check.

Three deeper problems in the function that *would* run:

- **`scan_vertices - flame_vertices` assumes vertex correspondence that does not exist.** FaceScape scans are not in FLAME topology. You need either a registration pass (fit FLAME to each scan, then compute displacement) or, more robustly, ray-cast each FLAME vertex along its normal to find the nearest point on the scan surface. Currently the subtraction would either throw on shape mismatch or produce meaningless deltas.
- **Point splatting is not rasterisation.** Writing 5023 UV points into a 512² canvas and dilating gives ~2% coverage, not the "~60-70%" the guide's verification asserts. You need barycentric rasterisation over the UV *triangles*. `nvdiffrast` does this (you're already installing it); so does a CPU scanline loop if you want zero GPU quota.
- **FLAME's pkl contains no UV coordinates.** `uv_coords` / `uv_faces` are parameters with no source. They live in the separate `FLAME_texture.npz` / `head_template.obj` from the same download page. The guide never mentions fetching them.
- The per-face Python loop for normals is ~200× slower than `np.add.at`. At 18,760 scans that's the difference between hours and days.

**This script needs to be written, not fixed.** Treat it as its own work item with its own gate — see the revised Guide.

---

## P2 — Wrong but visible

### 18. `insightface.model_zoo.get_model('buffalo_l')` is not a valid call
**File:** `src/utils/validation.py:44-47`, `Guide.md` Step 6

`buffalo_l` is a model *pack*, not a single ONNX file. `get_model` wants a path to a `.onnx`. Use `FaceAnalysis(name='buffalo_l')` and read `face.embedding`, or point at `~/.insightface/models/buffalo_l/w600k_r50.onnx`. Currently this is inside a bare `except Exception: pass`, so the same-person check is **silently skipped** for every photo — and because `embeddings` stays empty, the guard `if len(embeddings) == len(photo_paths)` is False and the check never runs. One of your five advertised validations does nothing.

### 19. The ArcFace model is constructed inside the per-photo loop
**File:** `src/utils/validation.py:42-49`

Model load + `prepare()` runs once per input image, and `validate_inputs()` also constructs its own `FaceDetector()` — so a 5-photo run initialises InsightFace six times. Then `pipeline.run()` constructs a *seventh* via `self.detector`, and re-detects every photo it already detected during validation.

**Fix:** `validate_inputs()` should take the detector as an argument and return its detections; `pipeline.run()` should consume them instead of re-running Stage 0. Currently `val_res.detections` is computed, thrown away, and recomputed 20 lines later.

### 20. BatchNorm in an adversarial generator, at batch size 4, under DDP
**File:** `src/stage3_detail/generator.py:66-71`

Three compounding issues: BN couples samples within a batch (a known source of GAN artefacts); batch 4 per GPU gives very noisy statistics; and under DDP, plain `BatchNorm2d` normalises per-GPU, so your effective batch is 4, not 8.

**Fix:** use `nn.InstanceNorm2d(out_c, affine=True)` or `nn.GroupNorm(8, out_c)`. Both are batch-independent, both are standard in image-to-image GANs, and both make item 16 moot. If you keep BN, wrap with `torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)` before the DDP wrap.

### 21. `blender_export.py` will raise on the first shape key
**File:** `scripts/blender_export.py:41-43`

```python
for idx, d in enumerate(deltas):
    kb.data[idx].co += d          # d is a Python list from JSON -> TypeError
```
`bpy` vector arithmetic needs a `mathutils.Vector`. Also `+=` adds the delta to the *basis* position, which is correct only if `blendshapes.json` stores offsets; if it stores absolute positions you want `=`. The format is undefined because nothing writes that file.

**Fix:**
```python
from mathutils import Vector
kb = mesh_obj.shape_key_add(name=bs_name, from_mix=False)
for idx, d in enumerate(deltas):
    if idx < len(kb.data):
        kb.data[idx].co = kb.data[idx].co + Vector(d)      # deltas, explicit
```
Also: `object_types={'MESH','ARMATURE'}` with no armature in the scene, and `use_selection=True` after operations that may have cleared the selection. Add `bpy.ops.object.select_all(action='DESELECT'); mesh_obj.select_set(True); bpy.context.view_layer.objects.active = mesh_obj` before export.

**And the bigger problem: nothing in the repo generates `blendshapes.json`.** See item 24.

### 22. Deprecated AMP API
**Files:** `trainer.py:3, 100, 122`, `losses.py:22`

`torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` are deprecated in torch ≥2.4 and emit warnings every step, which will bury your actual training logs. Use `torch.amp.autocast('cuda', dtype=torch.float16)` and `torch.amp.GradScaler('cuda')`.

### 23. Train/val split leaks subjects
**File:** `src/stage3_detail/data.py:22`

`self.disp_files[:split_idx]` splits a sorted filename list. FaceScape naming is `{subject}_{expression}`, so a 90/10 cut lands mid-subject: the same person's neutral scan trains while their smile validates. Your validation loss will be optimistic and you will not notice overfitting.

**Fix:** parse the subject ID and split on the *set of subjects*:
```python
subjects = sorted({p.stem.split('_')[0] for p in self.disp_files})
rng = random.Random(1337); rng.shuffle(subjects)
val = set(subjects[:max(1, len(subjects)//10)])
self.files = [p for p in self.disp_files
              if (p.stem.split('_')[0] in val) != is_train]
```

### 24. There is no retopology stage, and no blendshape generation
**Files:** `configs/default.yaml:37-41`, all of Stage 5

The config declares:
```yaml
stage5:
  topology: ict_facekit
  lod_triangle_counts: [24500, 5000, 2000, 500]
  blendshape_standard: arkit52
```
None of this exists. What the pipeline actually exports is FLAME topology: 5023 verts / 9976 triangles. The largest declared LOD (24,500 tris) is 2.5× more triangles than the mesh has. `src/stage5_export/` contains only an empty `__init__.py`.

`Research.md` is clear-eyed about this — it calls Stage 5 "a solved, mostly non-ML geometry-processing problem — doing it right is what actually makes this 'production grade'". The guide then allocates it three weeks and one page, and the page is about installing Blender.

**This is the single largest gap between the plan and the code.** The missing work, concretely:
1. **FLAME → ICT-FaceKit correspondence.** A one-time dense mapping (barycentric coordinates of each ICT vertex on the FLAME surface), computed once by non-rigid ICP between the two templates. Once built, it's a constant matrix: `v_ict = W @ v_flame`, applied in microseconds forever after. This is a genuine multi-day task and it gates everything downstream.
2. **ARKit-52 blendshape generation.** FLAME's expression basis is 100 PCA components; ARKit's is 52 semantic shapes (`jawOpen`, `browInnerUp`, …). These are different bases and the mapping is not built into either. Options: use ICT-FaceKit's own ARKit-compatible blendshape set and transfer it onto your identity via deformation transfer (recommended — ICT ships them and is MIT licensed); or solve a least-squares mapping from FLAME ψ to ARKit weights against a set of posed examples.
3. **Deformation transfer** to move those generic blendshapes onto each subject's identity — standard Sumner & Popović, or simply apply ICT's generic deltas in the subject's local frames.
4. **LOD decimation** with UV/blendshape preservation. Blender's Decimate modifier does not carry shape keys across. This is a known, annoying problem; plan to decimate the neutral, then re-project shape keys onto the decimated topology using the same barycentric machinery from (1).
5. **Skeleton.** `blender_export.py` exports `{'MESH','ARMATURE'}` and passes `use_armature_deform_only=True`, but no armature exists. UE5 Live Link needs a head/jaw/eye joint hierarchy.

Realistically this is **4–6 weeks of geometry work, not 3**, and it is the part that decides whether the output is a game asset or a research mesh.

### 25. Unused / dead configuration
- `configs/default.yaml:10` — `landmark_model: mediapipe`. MediaPipe is in `requirements.txt` and never imported. Either use it (FLAME fitting benefits from dense landmarks; InsightFace's 5-point is thin) or remove both.
- `configs/default.yaml:2` — `pipeline.stages` is read into `self.cfg` and never consulted. `pipeline.run()` hardcodes the sequence. Either drive the run from the config or delete the key.
- `stage0.align_size: 224`, `min_face_confidence: 0.5` — both ignored; crop sizes are hardcoded and `min_face_confidence` is never checked anywhere.
- `stage3.*` in the config duplicates `HYPERPARAMS` in `trainer.py`, and they already disagree (`trainer.py` has `id_loss_lambda`/`photo_loss_lambda` in the guide but not in the code). Single source of truth: load the yaml in the trainer.
- `identity_preservation_loss` is defined in `losses.py`, documented in the guide as a headline loss with `λ=5.0`, and **never imported or called**. The trainer's import list even omits it.

---

## P3 — Environment, process, cleanup

### 26. The Kaggle weekly GPU quota is never mentioned
`Research.md` states it correctly (~30 GPU-hours/week). `Guide.md` only ever discusses the 12-hour session cap. With 2×T4, an hour of wall-clock burns 2 quota-hours, so **a 12-hour session costs ~24 of your 30 weekly hours**. You get roughly one full session plus change per week. The Phase 3 plan of "50k steps" needs to be planned against ~15 wall-clock GPU hours/week, not against the session limit. Put a running quota budget in the guide.

### 27. SSH deploy key will fail on a missing trailing newline
**File:** `Guide.md` Day 0 Step 3 / Phase 1 Cell 1

```python
f.write(deploy_key)
```
Kaggle Secrets strip trailing whitespace. OpenSSH rejects a private key whose final `-----END...-----` line has no newline with `error in libcrypto` / `invalid format`. Write `deploy_key.rstrip() + "\n"`. Also `chmod 700 ~/.ssh`, not just 600 on the key.

Honestly, for a private repo pulled read-only, a fine-grained PAT over HTTPS is less fragile: `git clone https://x-access-token:{token}@github.com/...`.

### 28. `__MACOSX` and `.git` with no commits
The archive carries a full `__MACOSX` resource-fork tree and a `.git` directory with zero commits. Add to `.gitignore`: `__MACOSX/`, `.DS_Store`, `._*`. And commit — there is no history to review, no way to bisect, and no baseline.

### 29. Directory name contains a trailing space
`3d Model /` — a trailing space in a path breaks naive shell scripts and CI. Rename to `face-geo-pipeline`, matching the repo name the guide tells you to create.

### 30. No tests at all
Zero test files. For a pipeline this stateful, the following would have caught items 1, 5, 8, 11, 13, 15 and 16 on day one, and each is under 20 lines:

| Test | Asserts |
|---|---|
| `test_flame_loads` | 5023 verts, no NaN, range in metres |
| `test_flame_neutral_is_pose_free` | `decode(β,0,0) == decode_neutral(β)` |
| `test_distinct_identities` | two subjects → `‖β₁−β₂‖ > 0` |
| `test_disp_roundtrip` | `read(write(x)) ≈ x` within 1/65535 |
| `test_generator_shapes` | `(B,1,512,512)` out, and output std across batch > 0 |
| `test_ema_buffers` | EMA model buffers match source after update |
| `test_identity_score_needs_render` | raises when no preview PNG exists |

Add `pytest` to requirements and a `make test`. Wire it to run before every Kaggle push.

### 31. `setup.py` will install `src` as a top-level package
`find_packages()` picks up `src`, `evaluation`, `scripts` as top-level names. `import src.pipeline` works but `src` is about as generic a namespace as exists, and will collide the moment another dependency does the same. Rename to `face_geo/` or add `package_dir={'': '.'}` with an explicit list. Low urgency, annoying later.

---

## Licensing — the item most likely to kill the project

`Research.md` §7 handles FaceScape correctly. `Guide.md` Phase 4 then narrows the entire licensing track to *one email to Nanjing University*. The actual exposure is the whole stack, and every layer of it is non-commercial:

| Component | Status | Blocks commercial ship? |
|---|---|---|
| **FLAME 2020** (pinned in config) | Non-commercial research | **Yes** |
| **FLAME 2023 Open** | Released 11/2025 under CC-BY-4.0; previous versions remain non-commercial research only | **No — use this** |
| MICA | Non-commercial research | Yes |
| MICA unified dataset (LYHM / FaceWarehouse / Stirling) | Mixed, per-component | Yes, individually |
| SMIRK | Research; its install pulls FLAME, requiring registration, and it uses EMOCA's emotion model, requiring agreement to EMOCA's terms | Yes |
| EMOCA | Non-commercial | Yes |
| FaceScape | Explicit no-commercial-use | Yes |
| nvdiffrast | NVIDIA Source Code License | Check |
| ICT-FaceKit | MIT | No |
| InsightFace / ArcFace models | Model weights are research-licensed | Check |

**The one genuinely good piece of news, which the guide misses entirely:** MPI released FLAME 2023 Open under CC-BY-4.0 in November 2025, along with conversion code to translate expression parameters from FLAME 2023 to FLAME 2023 Open. That removes the foundation-layer blocker. `configs/default.yaml` currently pins `flame_version: FLAME2020`, the non-commercial one, for no stated reason.

**Recommendation — restructure the licensing track as a dependency audit, not one email:**
1. Switch to **FLAME 2023 Open** now, before any weights are trained. Retraining on a different basis later invalidates every checkpoint you've built. Verify the current terms yourself on the MPI site — I'm summarising a changelog entry, not giving legal advice.
2. Treat the "research" models (MICA, SMIRK, EMOCA) as **teachers, not shipped components**: use them to generate supervision for your own regressor trained from scratch on clean data. Whether this launders the licence is a genuinely contested question and depends on the specific terms — get an actual lawyer on it before betting the project.
3. Use FaceScape **only to validate the architecture**, never to train shipped weights, exactly as `Research.md` §7 already says.
4. Start the own-capture track in week 1, not week 8. It's the long pole: consent forms, releases, a rig, and subject recruitment all take calendar time that doesn't compress. Even 30 well-captured subjects with clean releases is worth more than 938 you cannot ship.

I'm not a lawyer and these terms change — verify each one directly with its current licence text before making a commercial commitment.

---

## What I'd do, in order

**This week**
1. Replace `src/utils/flame_model.py` (shipped with this review — tested, fixes items 1, 2, and adds working LBS).
2. Delete every silent fallback (items 5, 6, 7, 8). Let the pipeline fail loudly. Re-run the Phase 1 gate and watch it fail — that failure is your first real measurement.
3. Fix `requirements.txt` (item 3) and get a Kaggle session to actually import everything.
4. Add the seven tests from item 30.

**Next two weeks**
5. Vendor MICA properly, fix the preprocessing (items 10, 11), switch fusion to embedding-space (item 9).
6. Write the mesh renderer so `identity_score` means something.
7. Decide Stage 2: drop `coarse_detail` or bring EMOCA in (item 12).

**Before any Stage 3 training**
8. Actually write `build_uv_displacement_dataset.py` (item 17) — with real registration, real rasterisation, and a real p99.
9. Fix the three training bugs (items 14, 15, 16) and the unit contract (13). Each one alone silently prevents convergence.
10. Prove it on 50 subjects with a fixed seed before spending quota on 938.

**In parallel, starting now**
11. FLAME 2023 Open migration and the full licence audit.
12. Begin the FLAME→ICT-FaceKit correspondence (item 24). It's independent of the ML, it's on the critical path for shipping, and nobody has started it.