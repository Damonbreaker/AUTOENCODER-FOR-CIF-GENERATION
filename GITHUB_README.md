# AUTOENCODER-FOR-CIF-GENERATION

ML framework to reconstruct MOF crystal structures from CIF inputs using a **UMA → attention → VAE → MLP heads** pipeline.

This repository implements an inspectable, step-by-step path:

**CIF → structure code M → lattice normalization → UMA node embeddings → attention compression → VAE latent → predict N and L**

---

## Pipeline overview

```
CIF file
  │
  ▼  Step 1 — Read CIF (GEMMI)
gemmi_cif_utils.py, cif_read_all.py
  │
  ▼  Step 2 — Structure code M = (N, A, F, L)
structure_M_utils.py, structure_M_all.py, structure_M_batch_by_N.py
  │
  ▼  Step 3 — Lattice normalization (V, N, Std, V+Std, N+Std)
lattice_norm_utils.py, normalize_lattice_npz.py,
lattice_std_scale_utils.py, standard_scale_lattice_npz.py,
plot_lattice_norm_histograms.py
  │
  ▼  Step 4 — UMA encode (frozen message-passing backbone)
uma_structure.py, encoder_uma_crystal.py
  │
  ▼  Step 5 — Attention compression (N atoms → 1 crystal vector)
atom_attention_compressor.py, uma_attn_compress_N_0000_0050.py, uma_split_utils.py
  │
  ▼  Step 6 — VAE reparameterization + MLP heads (N and L)
vae_reparameterize.py, vae_atom_compressor.py, mlp_head_n.py, mlp_head_l.py
  │
  ▼  Step 7 — End-to-end training (100 epochs)
uma_train_dataset.py, train_pipeline.py, train_cdvae_style.py
```

---

## Structure representation

Each crystal is stored as **M = (N, A, F, L)**:

| Symbol | Meaning |
|--------|---------|
| **N** | Number of atoms |
| **A** | Atomic numbers, shape `(N,)` |
| **F** | Fractional coordinates, shape `(N, 3)` |
| **L** | Lattice parameters `[a, b, c, α, β, γ]` (angles in degrees) |

Saved as `.npz` files (one per structure).

---

## Step 1 — Read CIF (GEMMI)

### `gemmi_cif_utils.py`
Helper library for CIF I/O with GEMMI: find `.cif` files, fix quoted values, read structures, extract composition/summary metadata.

### `cif_read_all.py`
Batch runner over a CIF folder. Writes `cif_read_all_manifest.json`, `cif_read_all_records.jsonl`, and an error log.

---

## Step 2 — Extract structure code M = (N, A, F, L)

### `structure_M_utils.py`
Core conversion: GEMMI structure → `CrystalM`. Defines save/load/check for `.npz` format.

### `structure_M_all.py`
Batch runner: processes all CIFs and writes one `.npz` per structure into `all_npz/`, plus manifest and JSONL summary.

### `structure_M_batch_by_N.py`
Groups structure `.npz` files into N-bins (width 50, e.g. `N_0000_0050`) before UMA encoding. Writes batch manifests.

---

## Step 3 — Lattice normalization

Angles **α, β, γ are never scaled** — only lengths `a, b, c` are transformed.

### Volume (triclinic cell)

\[
V = a\,b\,c \sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha\cos\beta\cos\gamma}
\]

### Method 1 — Scale by N (CDVAE `scale_length`)

\[
a' = \frac{a}{N^{1/3}}, \quad b' = \frac{b}{N^{1/3}}, \quad c' = \frac{c}{N^{1/3}}
\]

### Method 2 — Scale by volume

\[
a' = \frac{a}{V^{1/3}}, \quad b' = \frac{b}{V^{1/3}}, \quad c' = \frac{c}{V^{1/3}}
\]

### Method 3 — Standard scaling (z-score on lengths)

Fit on a cohort (default: structures with **N ≤ 50**):

\[
a'' = \frac{a' - \mu_a}{\sigma_a}, \quad b'' = \frac{b' - \mu_b}{\sigma_b}, \quad c'' = \frac{c' - \mu_c}{\sigma_c}
\]

### Five lattice variants (N ≤ 50 cohort)

| Variant | Description |
|---------|-------------|
| `L_var_V` | Volume scaling only |
| `L_var_N` | N^(1/3) scaling only |
| `L_var_std` | StandardScaler on raw a,b,c |
| `L_var_V_std` | Volume scaling, then StandardScaler |
| `L_var_N_std` | N^(1/3) scaling, then StandardScaler |

### Scripts

| File | Role |
|------|------|
| `lattice_norm_utils.py` | Volume formula, `scale_N`, `scale_V` utilities |
| `normalize_lattice_npz.py` | Batch: writes `L_scaled_N` and `L_scaled_V` per structure |
| `lattice_std_scale_utils.py` | `ABCScaler` — fit/transform μ, σ on [a,b,c] |
| `standard_scale_lattice_npz.py` | Batch: produces all 5 variants above |
| `plot_lattice_norm_histograms.py` | Histograms / boxplots / violins for comparing variants |

---

## Step 4 — UMA encode (frozen backbone)

UMA (Universal Machine-learning potential for Atoms) runs **frozen** message passing and outputs per-atom embeddings. No energy/force/stress heads are used.

### `uma_structure.py`
Converts Task 02 `.npz` → **ASE Atoms** with full 3D periodic boundary conditions (physical cell from L, Cartesian positions from F).

### `encoder_uma_crystal.py`
Loads pretrained UMA backbone (`uma-m-1p1` by default), runs `backbone.forward()`, and returns per-atom **`node_emb_l0`** with shape **(N, 128)**. Saved as `*_uma_node.pt`.

**Requires:** `fairchem-core`, `ase`, HuggingFace login for UMA weights.

---

## Step 5 — Attention compression

Compress variable-length atom embeddings **(N, 128)** → single crystal vector **(1, 128)**.

### `atom_attention_compressor.py`
`AtomAttentionCompressor`: single-head attention over atoms — mean-pool query, K/V from all atoms, output is the attention message **m = a V** . Can run offline batch with 70/15/15 split.

### `uma_attn_compress_N_0000_0050.py`
Entry point for structures with **0 ≤ N < 50**.

### `uma_split_utils.py`
Lists `*_uma_node.pt` files and creates/loads **70/15/15** train/val/test split manifest.

---

## Step 6 — VAE + MLP heads

### `vae_reparameterize.py`
Gaussian VAE head: compressed vector → `μ`, `logvar`, latent `z` via reparameterization trick. Includes KL divergence.

\[
z = \mu + \epsilon \exp\!\left(\tfrac{1}{2}\log\sigma^2\right), \quad \epsilon \sim \mathcal{N}(0, I)
\]

\[
\mathrm{KL} = -\tfrac{1}{2}\sum_i \left(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2\right)
\]

### `vae_atom_compressor.py`
Chains `AtomAttentionCompressor` + `VAEReparameterizeHead` into one trainable encoder block.

### `mlp_head_n.py`
MLP: `z (64)` → positive atom-count rate (Softplus). Loss = **MSE + Poisson NLL** on N.

### `mlp_head_l.py`
MLP: `z (64)` → `L_norm (6)`. Loss = **MSE** on normalized lattice.

---

## Step 7 — End-to-end training

### `uma_train_dataset.py`
Loads `*_uma_node.pt` embeddings and Task 02 lattice labels (`N`, `L_norm`). Builds train/val/test splits.

### `train_pipeline.py` (primary)
Jointly trains attention compressor, VAE head, N head, and L head for **100 epochs** (Adam, lr = 1e-3).

**Total loss:**

\[
\mathcal{L} = 0.01\,\mathrm{KL} + 1.0\,(\mathrm{MSE}_N + \mathrm{Poisson}_N) + 10.0\,\mathrm{MSE}_L
\]

### `train_cdvae_style.py` (alternate)
Same attention + VAE backbone, but **N** is predicted with **CrossEntropy** (51 classes) instead of MSE + Poisson — closer to CDVAE's `num_atom_loss`.

---

## Data flow summary

| Step | Input | Output |
|------|-------|--------|
| 1 | `.cif` files | CIF manifest + records |
| 2 | CIF records | `{stem}.npz` with N, A, F, L |
| 2b | `.npz` all | N-batched folders `batch_N_*` |
| 3 | structure `.npz` | `L_scaled_N`, `L_scaled_V`, then 5 std variants |
| 4 | structure `.npz` | `{stem}_uma_node.pt` — `node_emb_l0` (N×128) |
| 5 | `*_uma_node.pt` | compressed crystal vector (1×128) |
| 6–7 | UMA + lattice labels | trained VAE + N/L decoders |

---

## Files currently in this repository

| File | Step |
|------|------|
| `gemmi_cif_utils.py` | 1 |
| `cif_read_all.py` | 1 |
| `structure_M_utils.py` | 2 |
| `structure_M_all.py` | 2 |
| `structure_M_batch_by_N.py` | 2 |
| `lattice_norm_utils.py` | 3 |
| `normalize_lattice_npz.py` | 3 |
| `lattice_std_scale_utils.py` | 3 |
| `standard_scale_lattice_npz.py` | 3 |
| `plot_lattice_norm_histograms.py` | 3 |
| `uma_structure.py` | 4 *(to upload)* |
| `encoder_uma_crystal.py` | 4 *(to upload)* |
| `atom_attention_compressor.py` | 5 |
| `uma_attn_compress_N_0000_0050.py` | 5 |
| `uma_split_utils.py` | 5 |
| `vae_reparameterize.py` | 6 |
| `vae_atom_compressor.py` | 6 |
| `mlp_head_n.py` | 6 |
| `mlp_head_l.py` | 6 |
| `uma_train_dataset.py` | 7 |
| `train_pipeline.py` | 7 |
| `train_cdvae_style.py` | 7 |

---

## Still to upload

| File | Purpose |
|------|---------|
| `uma_structure.py` | Task 02 `.npz` → ASE Atoms (PBC cell) |
| `encoder_uma_crystal.py` | Frozen UMA backbone → `node_emb_l0` (N×128) |

Optional later: batch runners (`uma_by_batch.py`, `uma_N_*.py`), PBS scripts, folder layout (`pipelines/`, `tasks/`).

---

## Dependencies

| Stage | Packages |
|-------|----------|
| Steps 1–2 | `gemmi`, `numpy` |
| Step 3 | `numpy`, `scikit-learn` (implicit in ABCScaler), `plotly` + `kaleido` (plots) |
| Step 4 | `fairchem-core`, `ase`, `torch`, HuggingFace UMA weights |
| Steps 5–7 | `torch` |

---

## Quick run order (local)

```bash
# 1. Read CIFs
py -3 cif_read_all.py path/to/cifs

# 2. Extract M
py -3 structure_M_all.py path/to/cifs
py -3 structure_M_batch_by_N.py

# 3. Normalize lattice
py -3 normalize_lattice_npz.py --task2
py -3 standard_scale_lattice_npz.py --task2 --n-max 50

# 4. UMA encode (after uploading Step 4 scripts)
# py -3 encoder driver using uma_structure.py + encoder_uma_crystal.py

# 5. Attention compress
py -3 uma_attn_compress_N_0000_0050.py

# 7. Train
py -3 train_pipeline.py
# or CDVAE-style N head:
py -3 train_cdvae_style.py
```

---

## License / attribution

Built as a stepwise reconstruction of CDVAE-style crystal autoencoding, using Meta FAIR's UMA backbone for node embeddings. See original CDVAE paper and fairchem documentation for model citations.
