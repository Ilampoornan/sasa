# Session Notes — Block C Execution (2026-06-28)

## Context

Working on a RunPod GPU instance (SSH). The 50 GB network volume is mounted at `/workspace`
(334 TB shared NFS). The container root disk is only 20 GB, so all caches and checkpoints
must live under `/workspace/sasa/`. A `.venv` (Python 3.11) with all dependencies was
already present at `/workspace/sasa/.venv`.

Goal: execute Block C of plan1.md — reproduce one real GPT-2 result from the pre-trained
SASA checkpoint without any new training, then run the stretch goals (cluster decoder
analysis + temporal subspace figure).

---

## Environment facts established at session start

| Item | Value |
|---|---|
| Volume mount | `/workspace` (NFS, 334 TB, caches go here) |
| Container root | 20 GB overlay, 13 GB free — do not fill |
| venv | `/workspace/sasa/.venv` (Python 3.11, all deps installed) |
| CUDA | Available (`torch.cuda.is_available() == True`) |
| Checkpoint | `checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10/` (cfg.json + sae_weights.safetensors) |
| HF cache (eval scripts) | `/workspace/sasa/.cache/hf*` (set automatically by `eval/_common.py`) |
| HF cache (training) | Requires `SASA_CACHE_ROOT=/workspace/sasa/.cache` env var (otherwise defaults to `~/.cache/sasa` on the container root) |

---

## Code changes made

### 1. `eval/_common.py` — fixed layer-norm denormalization bug in `SAEBenchAdapter.decode()`

**Problem:** When SAEBench called `sae.encode(x)` with a 3D tensor `(B, T, D)`, the
`TopKSASAInference.encode()` stored `ln_mu` and `ln_std` as shape `(B, T, 1)` (e.g.
`(16, 128, 1)`). The adapter's `decode()` method then reshaped the 3D output to 2D
`(B*T, D)` before calling `run_time_activation_norm_fn_out`. PyTorch tried to broadcast
`(2048, 768)` against `(16, 128, 1)`, which fails because 2048 ≠ 128 at dim -2.

**Error message:** `The size of tensor a (2048) must match the size of tensor b (128)
at non-singleton dimension 1`

**Fix (line ~399):** Remove the reshape-to-2D workaround and call `run_time_norm_out`
directly on the output tensor. The `ln_mu`/`ln_std` already match the shape of `out`
(both are 3D when encode was called with a 3D input), so broadcasting works correctly.

```python
# Before (broken):
if out.dim() > 2:
    out = run_time_norm_out(out.reshape(-1, out.shape[-1])).view(*original_shape)
else:
    out = run_time_norm_out(out)

# After (fixed):
out = run_time_norm_out(out)
```

### 2. `analysis/cluster_decoders.py` — added `--sae-dir` flag

**Problem:** The README documents `python -m analysis.cluster_decoders --sae-dir <path>`
but the script only had `--pt-path` (raw `.pt` state dict) and `--sae-release` (SAE Lens
pretrained). The SASA checkpoint is stored as `cfg.json + sae_weights.safetensors` and
must be loaded via `SAE.load_from_disk()`, which neither existing flag supports.

**Fix:** Added `--sae-dir` as a third option in the mutually-exclusive source group
(argparse, ~line 243), and added the corresponding `elif args.sae_dir is not None:` branch
in the dispatch block (~line 425) that:
1. Imports and registers the `topk_sasa` architecture class
2. Calls `SAE.load_from_disk(args.sae_dir, device=load_device)`
3. Extracts `W_dec` and sets `cache_tag` from the folder name

---

## Steps executed

### Step 1 — SAEBench core eval

**Command:**
```bash
source .venv/bin/activate
python -m eval.run_core \
  --topk-sasa-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --hook blocks.7.hook_resid_pre \
  --force-rerun
```

First run failed on SASA due to the layer-norm bug above. Fixed and reran. Both SAEs
evaluated successfully in ~33 s total. Results written to
`saebench_results_gpt2_topk_sasa/core/`.

**Results (headline table for the meeting):**

| Metric | Standard SAE | SASA |
|---|---|---|
| KL div score ↑ | 0.978 | 0.967 |
| CE loss score ↑ | 0.979 | 0.966 |
| Explained variance ↑ | 0.991 | 0.988 |
| L0 atoms | 59.2 | 60.0 |
| Cosine sim ↑ | 0.962 | 0.946 |
| **Avg max decoder cosim ↓** | **0.557** | **0.258** |
| Frac features alive | 0.990 | 0.900 |

SASA matches the standard SAE to within ~1.3% on all reconstruction metrics at the same
sparsity, while its decoder directions are **54% less similar** to each other — directly
demonstrating the reduction in feature redundancy that the paper claims.

### Step 2 — Checkpoint inspection

```python
from sae_lens import SAE, register_sae_class
from sasa import TopKSASAInference, TopKSASAInferenceConfig
import torch
register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)
sae = SAE.load_from_disk("checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10", device="cuda")
print(sae.W_dec.shape)   # torch.Size([12288, 768]) = 2048*6 x d_model ✓
W = sae.W_dec.view(2048, 6, 768)
print(torch.linalg.matrix_norm(W.float(), ord='nuc').mean())  # 5.9216 ± 0.044
```

Confirmed architecture: 2048 groups × rank 6 × d_model 768. Mean nuclear norm 5.92 with
std 0.044 — very tight across all groups, showing the nuclear-norm regularizer worked.

### Step 3 — Training smoke test

```bash
export SASA_CACHE_ROOT=/workspace/sasa/.cache
python -m scripts.train_sasa \
  --model-name gpt2 --hook-name blocks.7.hook_resid_pre --d-in 768 \
  --n-groups 2048 --group-rank 6 --k-groups 10 \
  --dataset apollo-research/Skylion007-openwebtext-tokenizer-gpt2 --tokenized \
  --training-tokens 2000000 --context-size 128 \
  --no-wandb --save-dir checkpoints
```

Completed 2 M tokens in ~100 s. Loss trajectory:
- Step 100: mse_loss = 62,784 | nuclear_loss = 599.5
- Step 200: mse_loss = 55,465
- Step 300: mse_loss = 50,642
- Step 400: mse_loss = 32,668 | nuclear_loss = 599.2

`avg_active_groups` was exactly 10.0 throughout (top-k selection working correctly).
`aux_loss` = 0 (no dead groups early in training). Model saved to
`checkpoints/topk_sasa_gpt2_blocks_7_hook_resid_pre_n2048_r6_k10/`.

**Important:** Always set `SASA_CACHE_ROOT=/workspace/sasa/.cache` before training.
Without it the script defaults to `~/.cache/sasa` on the container root (20 GB disk).

### Step 4 — `--sae-dir` flag added to cluster_decoders (code change, see above)

Verified with:
```bash
python -m analysis.cluster_decoders --sae-dir checkpoints/... --help
# shows: --pt-path PT_PATH | --sae-release SAE_RELEASE | --sae-dir SAE_DIR
```

### Step 5 — Cluster decoder redundancy analysis

**Standard SAE:**
```bash
python -m analysis.cluster_decoders \
  --sae-release gpt2-small-res-jb --hook blocks.7.hook_resid_pre \
  --knn-backend torch_exact --tune
```

**SASA:**
```bash
python -m analysis.cluster_decoders \
  --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --knn-backend torch_exact --tune
```

Note: both runs save to `manifold-sae/cluster_results/leiden_wdec_knn100/` (same path),
so the second run overwrites the first. Extract stats from each before running the next.

**Results:**

| Metric | Standard SAE | SASA |
|---|---|---|
| Total decoder atoms | 24,576 | 12,288 |
| Leiden clusters | 1,228 | 744 |
| Median cluster size | 16 atoms/concept | 13 atoms/concept |
| Median effective rank | 15.9 | 12.9 |
| Redundancy (size/pca_dim) | 1.12 | 1.11 |
| % clusters in [8,16] | 21% | 52% |

The standard SAE fragments each concept into ~16 independently-directed decoder atoms.
SASA represents the same concepts with ~13 atoms organized as rank-6 subspaces (groups),
using half the parameter budget. The % of clusters landing in the target [8,16] range
is much higher for SASA (52% vs 21%), meaning its clusters are more uniformly sized.

### Step 6 — Temporal subspace figure

**Finding the temporal group:** The paper uses group 1473 for its GPT-2 run, but
that is checkpoint-specific. A scan of all 2048 groups was run on controlled prompts
(days of week, months, seasons):

```python
# Top results:
#  #1  group 1751  mean_norm=11.37  ← used for analysis
#  #2  group 1473  mean_norm=8.11   ← paper's group, #2 in this checkpoint
```

**Running the analysis:**
```bash
python -m analysis.temporal_subspace \
  --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --group-id 1751 \
  --output-dir analysis_output/temporal_group1751 \
  --mode prompt --device cuda
```

Note: `--mdf-style` was intentionally omitted. It triggers
`collect_active_group_vectors_from_corpus` which tries to load the `openwebtext` dataset
using a bare name (no HF namespace), causing a `HfUriError` with the current
`huggingface_hub` version. Use `--corpus-dataset Skylion007/openwebtext` if corpus
scan is needed in future.

**Results (group 1751, from report):**

| Concept | Mean group norm | Cyclic order alignment |
|---|---|---|
| Days of week | 11.78 | **0.841** |
| Months | 11.43 | **0.700** |
| Years | 11.76 | monotonic PC trend |

Group 1751 fires consistently on all three temporal scales with nearly identical norms,
and the PCA ring structure shows clear cyclic ordering for days and months. This mirrors
§7.3 of the paper (Group 1473) — a single rank-6 subspace capturing a multi-dimensional
temporal concept that a standard SAE would split across many atoms.

**Output files** (`analysis_output/temporal_group1751/`):
- `group_1916_temporal_paper.png/.pdf` — main paper-style figure
- `group_1916_temporal_paper_3d.png/.pdf` — 3D embedding view
- `group_1916_temporal_paper_embeddings.png/.pdf` — embedding method comparison
- `group_1916_temporal_pca_scatter.png/.pdf` — raw PCA scatter
- `group_1916_temporal_paper_report.md` — full diagnostic report
- `group_1916_temporal_paper_summary.json` — machine-readable metrics

(The filename prefix says `group_1916` due to a quirk in the script's naming logic, but
the analysis is confirmed to be on group 1751 — the report header reads
`Group: 1751 | Group rank: 6`.)

---

## Files changed

| File | Change |
|---|---|
| `eval/_common.py` | Fixed layer-norm denorm broadcast bug in `SAEBenchAdapter.decode()` |
| `analysis/cluster_decoders.py` | Added `--sae-dir` flag and `SAE.load_from_disk` branch |

## Files / directories created (outputs)

| Path | Contents |
|---|---|
| `saebench_results_gpt2_topk_sasa/core/` | SAEBench eval JSONs + `core_summary.csv` |
| `checkpoints/topk_sasa_gpt2_blocks_7_hook_resid_pre_n2048_r6_k10/` | Smoke-test trained SAE (2 M tokens) |
| `manifold-sae/cluster_results/leiden_wdec_knn100/` | Cluster membership + stats (last run = SASA) |
| `analysis_output/temporal_group1751/` | Temporal PCA figures and report for group 1751 |
| `/workspace/sasa/.cache/` | HF model/dataset cache (on volume disk) |
