# 3D Face & Micro-Detail Dataset Registry (`DATASETS.md`)

This document serves as the master procurement guide, technical specification, and operational registry for all official academic 3D scan databases and high-resolution community datasets used to train the **Humanoid-Face-3D** pipeline to production-grade accuracy.

---

## 🧭 Executive Summary: Where Data Powers the Pipeline

The pipeline requires two fundamentally different types of training data:

1. **Stage 1 (Identity Regression & MICA):** Requires **calibrated 2D multi-view portraits paired with ground-truth 3D laser/photogrammetry scans** to regress 300-D FLAME shape coefficients $\beta$.
2. **Stage 3 (Micro-Detail & Pore GAN):** Requires **ultra-high-resolution heightfields / displacement maps (1024²–4096²)** to synthesize sub-millimeter pores, follicular pits, and dynamic expression wrinkles.
3. **Stages 0, 2, 4, 5:** Use pre-trained vision foundations or pure geometric math; they require zero local scan datasets.

---

## 🏛️ 1. Official Academic 3D Scan Corpora

The following datasets represent the gold standard of ground-truth 3D metric facial scans used in top-tier computer vision research (ECCV, CVPR, SIGGRAPH).

### 1. FaceScape (Nanjing University)
* **Website:** [https://nju-3dv.github.io/projects/FaceScape/](https://nju-3dv.github.io/projects/FaceScape/)
* **Scale:** **847 subjects**, 20 specific facial expressions per subject, 4K multi-view DSLR captures.
* **Assets Provided:**
  - High-precision 3D registered head meshes (OBJ format, ~50,000 to ~2,000,000 vertices).
  - 4K texture maps & displacement maps.
  - Pre-computed bilinear 3D face models (`facescape_bm_v1.6_*.npz`).
* **License:** Free for non-commercial academic research.
* **How to Obtain:**
  1. Download the license agreement from the project website.
  2. Sign and email to `nju3dv@nju.edu.cn` with subject: `[FaceScape Dataset Request]`.
  3. You will receive private Google Drive / Baidu Netdisk download links.
* **Pipeline Integration:**
  Drop downloaded OBJ meshes into `data/external/3d_scans/facescape/` and execute:
  ```bash
  python3 scripts/build_uv_displacement_dataset.py --scans_dir data/external/3d_scans/facescape --resolution 1024
  ```

---

### 2. Stirling / ESRC 3D Face Database (University of Stirling)
* **Website:** [http://pics.stir.ac.uk/ESRC/](http://pics.stir.ac.uk/ESRC/)
* **Scale:** **133 subjects** captured with a 3D Di3D photogrammetry camera system.
* **Assets Provided:**
  - Wavefront OBJ 3D surface meshes (both raw unconformed and standardized conformed).
  - Multi-view 2D portraits and emotional expression sequences.
* **License:** Free for academic research (reciprocal agreement).
* **How to Obtain:**
  1. Download license form from `http://pics.stir.ac.uk/ESRC/`.
  2. Email signed form to `3dfacedb@gmail.com`.
* **Pipeline Integration:**
  Drop conformed meshes into `data/external/3d_scans/stirling/`.

---

### 3. Florence 2D/3D Face Dataset (University of Florence - MICC)
* **Website:** [https://www.micc.unifi.it/resources/datasets/florence-3d-faces/](https://www.micc.unifi.it/resources/datasets/florence-3d-faces/)
* **Scale:** **53 subjects** with high-resolution 3D structured-light scans.
* **Assets Provided:**
  - Controlled structured-light 3D meshes (OBJ / VRML).
  - High-definition video sequences in controlled, PTZ, and outdoor environments.
* **License:** Free for academic research upon form submission.
* **How to Obtain:**
  1. Fill the request form at the MICC resource page.
  2. Direct contact: `micc@unifi.it`.

---

### 4. LYHM / Headspace Dataset (University of York)
* **Website:** [https://www-users.york.ac.uk/~np7/research/Headspace/](https://www-users.york.ac.uk/~np7/research/Headspace/)
* **Scale:** **1,211 subjects** spanning ages 1 to 90 years across diverse demographics.
* **Assets Provided:**
  - High-density 3D surface meshes (~100,000 vertices per head).
  - Full-head cranial geometry including neck and ears.
* **License:** Academic research agreement via University of York Computer Science department.

---

### 5. MICA Unified Training Corpus (Zielonka et al., ECCV 2022)
* **GitHub:** [https://github.com/Zielon/MICA](https://github.com/Zielon/MICA)
* **Scale:** **2,315 unified subjects** across 8 datasets registered directly into **FLAME topology**.
* **Assets Provided:**
  - Pre-registered FLAME shape parameters $\beta$ and ground-truth vertex alignments.
* **How to Obtain:**
  Contact the authors at `mica@tue.mpg.de` or clone `https://github.com/Zielon/MICA` and run `datasets/` scripts.

---

## 🌐 2. Open Community High-Resolution 2D Datasets (Instant Cloud Access)

For weakly supervised training, detail consistency, and multi-view pore learning, open community 2D datasets provide tens of thousands of studio-quality human faces.

| Dataset Name | Source / Kaggle Reference | Image Count | Resolution | Key Use Case |
| :--- | :--- | :---: | :---: | :--- |
| **CelebA-HQ 1024²** | `thang1703/celebahq-1024x1024` | 30,000 | $1024 \times 1024$ | Macro-wrinkle & pore detail training; multi-ethnic diversity. |
| **Flickr-Faces-HQ (FFHQ)** | `tommykamaz/faces-dataset-small` | 70,000 | $1024 \times 1024$ | Unconstrained age variance, eye wrinkles, skin micro-pores. |
| **FLAME 2020 Foundation** | `nightshowdown/flame-model` | N/A | 5,023 verts | Base topology, skin weights, UV coordinates, blendshape bases. |

### How to Attach Instantly in Kaggle Cloud Kernels
No need to download 30 GB locally. Simply add the Kaggle dataset reference to `kernel_configs` in `scripts/kaggle_runner.py`:
```python
"phase3": {
    "slug": "phase-3-deep-detail-gan-train",
    "datasets": [
        "nightshowdown/flame-model",
        "thang1703/celebahq-1024x1024"
    ],
}
```

---

## 🎨 3. Photogrammetry Scans & Micro-Pore Displacement Packs

For micro-pore ground truth:
1. **USC ICT Digital Emily 2 & Digital Ira:**
   - Free open research assets from USC Institute for Creative Technologies.
   - Contains raw 3D head scan meshes (OBJ) + 8K multi-channel displacement maps (pores, secondary wrinkles, tertiary roughness).
   - Available at: [https://vgl.ict.usc.edu/Research/DigitalEmily2/](https://vgl.ict.usc.edu/Research/DigitalEmily2/)
2. **TexturingXYZ Displacement References:**
   - Triple-channel displacement maps (R = Cavity/Pore, G = Micro, B = Secondary wrinkle).
   - Free educational samples available on their store.

---

## 🛠️ 4. Data Procurement & Processing Toolchain

Use the integrated automation script [`scripts/download_community_data.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/download_community_data.py):

* **Check current dataset inventory:**
  ```bash
  python3 scripts/download_community_data.py --check
  ```
* **Generate ready-to-send license application emails:**
  ```bash
  python3 scripts/download_community_data.py --generate_templates
  ```
* **Download open research photogrammetry sample scans:**
  ```bash
  python3 scripts/download_community_data.py --download_sample_scans
  ```
* **Expand synthetic demographic dataset to 100+ subjects (0 GPU quota):**
  ```bash
  python3 scripts/download_community_data.py --scale_synthetic --n_subjects 100
  ```
