import argparse
import os
from pathlib import Path

import torch

cache_root = Path(os.environ.get("SASA_CACHE_ROOT", Path.home() / ".cache" / "sasa"))
cache_root.mkdir(parents=True, exist_ok=True)
wandb_root = cache_root / "wandb"
tmp_root = cache_root / "tmp"
wandb_root.mkdir(parents=True, exist_ok=True)
tmp_root.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(cache_root / "hf")
os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hf_transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf_hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DIR"] = str(wandb_root)
os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
os.environ["WANDB_CONFIG_DIR"] = str(wandb_root / "config")
os.environ["WANDB_DATA_DIR"] = str(wandb_root / "data")
os.environ["TMPDIR"] = str(tmp_root)
os.environ["TEMP"] = str(tmp_root)
os.environ["TMP"] = str(tmp_root)
for d in ["hf", "hf_datasets", "hf_transformers", "hf_hub"]:
    (cache_root / d).mkdir(parents=True, exist_ok=True)
for d in ["cache", "config", "data"]:
    (wandb_root / d).mkdir(parents=True, exist_ok=True)

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    register_sae_class,
    register_sae_training_class,
)

from sasa import (
    TopKSASA,
    TopKSASAConfig,
    TopKSASAInference,
    TopKSASAInferenceConfig,
)

_original_log = LoggingConfig.log


def _patched_log(
    self, trainer, weights_path, cfg_path, sparsity_path=None, wandb_aliases=None
):
    try:
        _original_log(
            self, trainer, weights_path, cfg_path, sparsity_path, wandb_aliases
        )
    except OSError as e:
        if e.errno == 122:
            print(
                f"WARNING: Disk quota exceeded when logging checkpoint to wandb. Error: {e}"
            )
        else:
            raise


LoggingConfig.log = _patched_log

register_sae_training_class("topk_sasa", TopKSASA, TopKSASAConfig)
register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)


def train_topk_sasa(
    model_name: str,
    hook_name: str,
    d_in: int,
    n_groups: int,
    group_rank: int,
    k_groups: int,
    lr: float,
    weight_decay: float,
    train_batch_size_tokens: int,
    context_size: int,
    training_tokens: int,
    dataset_path: str,
    streaming: bool,
    is_dataset_tokenized: bool,
    log_to_wandb: bool,
    wandb_project: str,
    device: str | None,
    seed: int,
    n_checkpoints: int,
    checkpoint_path: str,
    save_dir: str,
    dtype: str,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d_sae = n_groups * group_rank
    total_training_steps = training_tokens // train_batch_size_tokens
    lr_warm_up_steps = 1000
    lr_decay_steps = total_training_steps // 5

    cfg = LanguageModelSAERunnerConfig(
        model_name=model_name,
        hook_name=hook_name,
        dataset_path=dataset_path,
        streaming=streaming,
        is_dataset_tokenized=is_dataset_tokenized,
        sae=TopKSASAConfig(
            d_in=d_in,
            d_sae=d_sae,
            n_groups=n_groups,
            group_rank=group_rank,
            k_groups=k_groups,
            k_aux=512,
            aux_coefficient=1.0,
            encoder_norm_renorm=True,
            normalize_activations="layer_norm",
            apply_b_dec_to_input=True,
        ),
        lr=lr,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        train_batch_size_tokens=train_batch_size_tokens,
        context_size=context_size,
        n_batches_in_buffer=128,
        training_tokens=training_tokens,
        store_batch_size_prompts=16,
        eval_batch_size_prompts=8,
        n_eval_batches=10,
        logger=LoggingConfig(
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=20,
        ),
        device=device,
        seed=seed,
        n_checkpoints=n_checkpoints,
        checkpoint_path=checkpoint_path,
        dtype=dtype,
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / (
        f"topk_sasa_"
        f"{model_name.replace('/', '_')}_"
        f"{hook_name.replace('.', '_')}_"
        f"n{n_groups}_r{group_rank}_k{k_groups}"
    )
    sae.save_inference_model(str(save_path))
    print(f"Saved SAE to: {save_path}")
    return sae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_save_dir = str(Path(__file__).resolve().parent / "trained_saes")
    parser.add_argument("--model-name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--hook-name", type=str, default="blocks.12.hook_resid_post")
    parser.add_argument("--d-in", type=int, default=2304)
    parser.add_argument("--n-groups", type=int, default=1536)
    parser.add_argument("--group-rank", type=int, default=16)
    parser.add_argument("--k-groups", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--training-tokens", type=int, default=500_000_000)
    parser.add_argument("--batch-size-tokens", type=int, default=4096)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="topk-sasa")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-checkpoints", type=int, default=10)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/topk_sasa")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=default_save_dir,
        help="Directory to save the trained SAE (inference model).",
    )
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--tokenized", action="store_true")
    args = parser.parse_args()

    train_topk_sasa(
        model_name=args.model_name,
        hook_name=args.hook_name,
        d_in=args.d_in,
        n_groups=args.n_groups,
        group_rank=args.group_rank,
        k_groups=args.k_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_batch_size_tokens=args.batch_size_tokens,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        dataset_path=args.dataset,
        streaming=not args.no_streaming,
        is_dataset_tokenized=args.tokenized,
        log_to_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        device=args.device,
        seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir,
        dtype=args.dtype,
    )
