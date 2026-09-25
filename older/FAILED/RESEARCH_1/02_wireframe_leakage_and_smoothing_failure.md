# Wireframe Leakage & Conditioning Smoothing Analysis (Research 1)

> **Document:** DOCS/FAILED/RESEARCH_1/02_wireframe_leakage_and_smoothing_failure.md  
> **Topic:** Detailed analysis of polygonal facet artifacts, spatial filtering bugs, and normalization contracts in the Research 1 pipeline.

---

## 1. Problem Statement

In initial production evaluations of Stage 3 (Detail GAN), 3D meshes rendered with synthesized displacement maps exhibited an unphysical geometric artifact: **an overt triangular facet pattern reflecting the FLAME 2020 base mesh topology**.

Instead of pores or micro-wrinkles, the skin appeared as if the underlying polygonal cage was pressed through the surface like chicken wire under rubber.

---

## 2. Root Cause Analysis: The Three-Way Breakdown

The artifact was caused by a compound failure across three separate modules:

```
┌────────────────────────────────────────┐
│  FLAME Mesh (5,023 verts / 9,976 tris) │
└──────────────────┬─────────────────────┘
                   │
                   ▼ Loop Subdiv Level 1 (20k verts)
┌────────────────────────────────────────┐
│  Coarse Ray-Cast Interpolation Error   │  ──> Triangular facets baked into disp maps
└──────────────────┬─────────────────────┘
                   │
                   ▼ Gaussian Blur σ=1.2 (Insufficient)
┌────────────────────────────────────────┐
│  Conditioning Map Edge Leakage         │  ──> Pos & Norm maps leak triangle borders
└──────────────────┬─────────────────────┘
                   │
                   ▼ Normalization Contract Mismatch
┌────────────────────────────────────────┐
│  Inference: Zero-Mean Unit-Variance    │
│  Training:  Clip((pos + 0.20) / 0.40)  │  ──> OOD (Out-of-Distribution) activation
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Discriminator Amplifies Wireframe     │
└────────────────────────────────────────┘
```

---

### Failure 1: The Normalization Domain Mismatch

In `src/stage3_detail/data.py` (the dataset loader used during training):
```python
# Training Contract (data.py)
pos = np.clip((pos_raw + 0.20) / 0.40, 0.0, 1.0)
pos = (pos - 0.5) / 0.5  # Normalized to [-1.0, 1.0]
```
The training data assumed FLAME vertices in meters ($[-0.15\,\text{m}, +0.15\,\text{m}]$), bounded tightly to $[0, 1]$ using a linear shift and scale.

However, in `src/stage3_detail/inference.py`:
```python
# Inference Bug (inference.py)
pos_norm = (pos_map - pos_map.mean()) / (pos_map.std() + 1e-8)
```
Inference applied statistical z-score standardization (zero mean, unit variance). 
* At inference, the input distribution was completely out-of-distribution (OOD) relative to the generator's trained weights.
* The activation values in the first encoder layer (`enc1`) saturated, forcing the residual skip connections (`e1`) to pass raw, un-modulated high frequencies directly to the final layer (`dec1`).

---

### Failure 2: Under-Smoothing of Conditioning Geometry ($\sigma=1.2$)

The FLAME base mesh has coarse planar triangles. When projected into UV space:
* The 3D position map $P(u, v)$ and normal map $N(u, v)$ have discontinuous first and second spatial derivatives ($\nabla^2 P \ne 0$) across triangle seams.
* In Research 1, the smoothing applied was:
  $$\sigma = 1.2 \quad (\text{kernel size} \approx 5 \times 5)$$
* At $1024 \times 1024$ resolution, a $5 \times 5$ kernel only spans $\approx 0.5\%$ of the face. The sharp facet edges between adjacent triangles were completely preserved.

#### The Second-Derivative (Laplacian) Signature:
When measuring the Laplacian of the conditioning maps:
$$\mathcal{L}(u, v) = \sqrt{\left(\nabla^2 P_x\right)^2 + \left(\nabla^2 P_y\right)^2 + \left(\nabla^2 P_z\right)^2}$$
* With $\sigma=1.2$, the interior facial Laplacian was **$0.124$ to $0.187$**—a massive spike along every triangular edge.
* The neural network's convolution kernels detected these high-gradient lines and treated them as primary structural features.

---

### Failure 3: Boundary Smearing Without Normalized Convolution

When standard Gaussian blurring was applied:
$$I_{\text{smooth}} = G_\sigma * I$$
At the outer boundary of the facial mask (where skin meets the unmapped black background), the zero-value background bled into the perimeter of the cheeks, chin, and forehead. This caused severe edge darkening and artificial outward displacement flares at the silhouette boundary.

---

## 3. The Technical Solution & Its Limits

To fix the wireframe defect during Research 1, we implemented:

1. **Subdivision Level 2 Ray-Casting:**  
   Subdivided the FLAME template 2 levels prior to ray-casting ($\approx 80,000$ vertices, $160,000$ triangles). This made the polygon facet width smaller than 2 UV pixels at $1024^2$.
2. **$C^2$ Normalized Convolution ($\sigma=8.0$):**  
   Replaced standard Gaussian blurring with normalized convolution:
   $$I_{\text{norm\_conv}}(u, v) = \frac{(I \cdot M) * G_\sigma}{M * G_\sigma + \epsilon}$$
   Where $M$ is the binary facial mask. This eliminated mask edge bleeding and pushed the interior Laplacian down to **$0.029347$** (clean $C^2$ continuity).
3. **Harmonized Normalization:**  
   Aligned `inference.py` strictly with `data.py`:
   $$\hat{P} = \text{clip}\left(\frac{P + 0.20}{0.40}, 0.0, 1.0\right)$$
4. **Depthwise-Separable Residual Smoothing Block in Generator:**  
   Added `e1_smooth` to the $e_1$ skip connection in `DetailGenerator`:
   $$e_{1,\text{clean}} = e_1 + \text{Smooth}(e_1)$$

---

## 4. Why This Still Didn't Yield Film-Grade Detail

While the above mathematical fix completely removed the wireframe grid lines, **it exposed the second, deeper failure**:

Once the wireframe noise was gone, the resulting displacement maps were revealed to be **almost completely flat**. The Meta Multiface scans did not have skin pores; they only had general cranial shape.

Removing the wireframe artifact did not produce pore detail—it simply produced smooth, featureless plastic. This proved definitively that the detail generation strategy had to be rebuilt from the ground up.
