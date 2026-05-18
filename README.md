# SASA: Subspace-Aware Sparse Autoencoders

Code to reproduce the GPT-2 results of the SASA paper. SASA replaces each one-dimensional SAE decoder atom with a learned decoder subspace, enforces Top-`s` group sparsity, and adds a nuclear-norm regularizer.

## Layout

```
sasa/                  TopK-SASA model + BatchTopK-Manifold baseline architecture
scripts/train_sasa.py  GPT-2 training
eval/                  SAEBench wrappers: core metrics + interpretability
theory/                Line-covering verification
analysis/              Decoder-cluster redundancy, temporal + geographical subspaces
checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10/   Pre-trained SASA (GPT-2 layer 7, K=2048, r=6, s=10)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `SASA_CACHE_ROOT` to control HuggingFace / W&B / tmp cache root (default: `~/.cache/sasa`).

## Train SASA on GPT-2 (150M OpenWebText tokens, ~hrs on 1x A100)

```bash
python -m scripts.train_sasa \
  --model-name gpt2 \
  --hook-name blocks.7.hook_resid_pre \
  --d-in 768 \
  --n-groups 2048 --group-rank 6 --k-groups 10 \
  --dataset apollo-research/Skylion007-openwebtext-tokenizer-gpt2 --tokenized \
  --training-tokens 150000000 \
  --context-size 128 \
  --save-dir checkpoints
```

## Evaluation

KL / CE / explained variance / L0, SASA vs. SAELens control SAE:

```bash
python -m eval.run_core \
  --topk-sasa-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --hook blocks.7.hook_resid_pre
```

Sparse-probing AutoInterp + absorption, cross-architecture:

```bash
python -m eval.run_interpretability \
  --manifold-sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --baseline-sae-release gpt2-small-res-jb \
  --hook blocks.7.hook_resid_pre \
  --model-name gpt2
```

## Theory verification

Synthetic + GPT-2 line-covering curves:

```bash
python -m theory.verify_covering
```

## Interpretability figures

Decoder-cluster redundancy ratios on a trained SAE:

```bash
python -m analysis.cluster_decoders --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10
```

Temporal subspace:

```bash
python -m analysis.temporal_subspace --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10
```

Geographical subspace; needs RAVEL city/country/continent JSONs under `data/ravel/`:

```bash
python -m analysis.geographical_subspace --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10
```
