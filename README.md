# SASA: Subspace-Aware Sparse Autoencoders

Code to reproduce the GPT-2 results from the SASA paper. A standard sparse
autoencoder gives every feature a single decoder direction. SASA instead learns
a small decoder *subspace* per feature, keeps only the top groups active for any
given token, and regularizes those subspaces with a nuclear-norm penalty.

## Layout

```
sasa/                  SASA model
scripts/train_sasa.py  Training script for GPT-2
eval/                  SAEBench evaluation wrappers
analysis/              Decoder-cluster redundancy analysis
checkpoints/           Pre-trained SASA checkpoint (GPT-2 layer 7; 2048 groups, rank 6, 10 active per token)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`SASA_CACHE_ROOT` sets where HuggingFace, W&B, and temporary files are cached
(default: `~/.cache/sasa`).

## Training

The command below trains a SASA autoencoder on layer 7 of GPT-2 small over 150M
tokens of OpenWebText. A single GPU is enough.

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

Both evaluation scripts compare a trained SASA autoencoder against a baseline:
the standard pretrained GPT-2 SAE released as `gpt2-small-res-jb` through SAE
Lens. This is the closest off-the-shelf reference point — a conventional SAE
trained on the same layer of the same model.

Reconstruction and faithfulness metrics (KL divergence, cross-entropy,
explained variance, L0):

```bash
python -m eval.run_core \
  --topk-sasa-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --hook blocks.7.hook_resid_pre
```

Sparse-probing accuracy:

```bash
python -m eval.run_interpretability \
  --manifold-sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --baseline-sae-release gpt2-small-res-jb \
  --model-name gpt2
```

## Decoder-cluster analysis

Reports how strongly decoder directions cluster together (redundancy ratios) for
a trained autoencoder:

```bash
python -m analysis.cluster_decoders --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10
```
