"""
Run SAEBench sparse probing (sae-probes) comparing two SAEs at the same hook.

Compares:
  - local BatchTopK Manifold SAE loaded from disk
  - a baseline SAE (either a SAE-Lens release via `SAE.from_pretrained(...)` or a local Mistral wrapper)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE, register_sae_class

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sasa.batchtopk_baseline import (  # noqa: E402
    BatchTopKManifoldSAEInference,
    BatchTopKManifoldSAEInferenceConfig,
)
from eval._common import _set_hf_cache_defaults, SAEBenchAdapter  # noqa: E402


class _SaeProbesEncodeFloat32Adapter:
    def __init__(self, sae: Any):
        self._sae = sae

    @property
    def cfg(self):
        return self._sae.cfg

    @property
    def W_enc(self):
        return getattr(self._sae, "W_enc", None)

    @property
    def W_dec(self):
        return getattr(self._sae, "W_dec", None)

    @property
    def b_dec(self):
        return getattr(self._sae, "b_dec", None)

    @property
    def device(self):
        return getattr(self._sae, "device", None)

    @property
    def dtype(self):
        return getattr(self._sae, "dtype", None)

    def to(self, *args, **kwargs):
        if hasattr(self._sae, "to"):
            self._sae = self._sae.to(*args, **kwargs)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self._sae.encode(x)
        return z.to(dtype=torch.float32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if hasattr(self._sae, "decode"):
            return self._sae.decode(z)
        raise AttributeError("Wrapped SAE has no decode()")


def _infer_hook_layer_from_hook_name(hook_name: str | None) -> int | None:
    if not hook_name or not isinstance(hook_name, str):
        return None
    # e.g. "blocks.7.hook_resid_pre"
    parts = hook_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except Exception:
            return None
    return None


def _canonicalize_model_name_for_sae_probes(model_name: str | None) -> str | None:
    if not model_name or not isinstance(model_name, str):
        return None
    if model_name in {"gpt2-small", "openai/gpt2", "gpt2"} or model_name.startswith("gpt2"):
        return "gpt2"
    return model_name


def _resolve_model_name_from_cfgs(*cfgs: Any) -> str | None:
    for cfg in cfgs:
        try:
            canon = _canonicalize_model_name_for_sae_probes(getattr(cfg, "model_name", None))
            if canon:
                return canon
        except Exception:
            pass
        try:
            meta = getattr(cfg, "metadata", None)
            if isinstance(meta, dict):
                canon = _canonicalize_model_name_for_sae_probes(meta.get("model_name"))
                if canon:
                    return canon
            if meta is not None and hasattr(meta, "model_name"):
                canon = _canonicalize_model_name_for_sae_probes(getattr(meta, "model_name", None))
                if canon:
                    return canon
        except Exception:
            pass
    return None


def _group_atoms_sorted_group_indices(
    X_train_atoms: torch.Tensor, y_train: torch.Tensor, *, n_groups: int, group_rank: int
) -> torch.Tensor:
    X = X_train_atoms.view(-1, n_groups, group_rank)
    pos = X[y_train == 1].mean(dim=0)
    neg = X[y_train == 0].mean(dim=0)
    diff = pos - neg
    scores = diff.norm(dim=-1)
    return torch.argsort(scores.abs(), descending=True)


def _group_atoms_expand_atom_indices(
    group_ids: torch.Tensor, *, group_rank: int
) -> torch.Tensor:
    r = torch.arange(group_rank, device=group_ids.device)
    return (group_ids[:, None] * group_rank + r[None, :]).reshape(-1)


def _run_group_atoms_sae_probes_jsons(
    *,
    sae: Any,
    sae_results_path: str,
    model_name: str,
    hook_name: str,
    datasets: list[str],
    ks_groups: list[int],
    reg_type: str,
    setting: str,
    binarize: bool,
    model_cache_path: str,
    device: str,
    n_groups: int,
    group_rank: int,
    batch_size: int = 128,
) -> None:
    from sae_probes.generate_model_activations import ensure_dataset_activations
    from sae_probes.generate_sae_activations import generate_sae_activations
    from sae_probes.run_sae_evals import get_save_metrics_path, mean_act_normalization
    from sae_probes.utils_training import find_best_reg

    ensure_dataset_activations(
        model_name=model_name,
        dataset_short_names=datasets,
        hook_names=[hook_name],
        model_cache_path=model_cache_path,
        device=device,
    )

    for dataset in datasets:
        save_path = get_save_metrics_path(
            dataset=dataset,
            hook_name=hook_name,
            reg_type=reg_type,  # type: ignore[arg-type]
            model_name=model_name,
            sae_results_path=sae_results_path,
            binarize=binarize,
            setting=setting,  # type: ignore[arg-type]
        )
        if save_path.exists():
            continue

        acts = generate_sae_activations(
            sae=sae,
            setting=setting,  # type: ignore[arg-type]
            dataset=dataset,
            hook_name=hook_name,
            model_name=model_name,
            device=device,
            num_train=None,
            frac=None,
            model_cache_path=model_cache_path,
            batch_size=batch_size,
        )

        X_train = acts.X_train.float()
        X_test = acts.X_test.float()
        y_train = acts.y_train.long()
        y_test = acts.y_test.long()

        X_train_norm = mean_act_normalization(X_train)
        sorted_groups = _group_atoms_sorted_group_indices(
            X_train_norm, y_train, n_groups=n_groups, group_rank=group_rank
        )

        all_metrics: list[dict[str, Any]] = []
        for k_groups in ks_groups:
            top_groups = sorted_groups[: int(k_groups)]
            atom_indices = _group_atoms_expand_atom_indices(
                top_groups, group_rank=group_rank
            )

            X_train_filtered = X_train[:, atom_indices]
            X_test_filtered = X_test[:, atom_indices]

            if binarize and setting == "normal":
                X_train_filtered = X_train_filtered > 1
                X_test_filtered = X_test_filtered > 1

            results = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train.numpy(),
                X_test=X_test_filtered,
                y_test=y_test.numpy(),
                n_jobs=-1,
                parallel=False,
                penalty=reg_type,
            )
            metrics = dict(results.metrics.__dict__)
            metrics.update(
                {
                    "k": int(k_groups),  # number of groups (not atoms)
                    "k_atoms": int(atom_indices.numel()),
                    "dataset": dataset,
                    "hook_name": hook_name,
                    "reg_type": reg_type,
                    "binarize": binarize,
                    "group_indices": top_groups.cpu().tolist(),
                    "indices": atom_indices.cpu().tolist(),  # length = k_groups*group_rank
                }
            )
            all_metrics.append(metrics)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)


def load_saes(
    *,
    device: str,
    manifold_sae_dir: str,
    baseline_sae_release: str | None,
    downloaded_sae_path: str,
    downloaded_sae_layer: int | None,
    dtype: str,
    manifold_feature_mode: str,
) -> tuple[list[tuple[str, Any]], int | None, int | None]:
    manifold_sae_raw = SAE.load_from_disk(manifold_sae_dir, device=device)
    if manifold_feature_mode not in {"atoms", "group_norm", "group_atoms"}:
        raise ValueError(
            "--manifold-feature-mode must be 'atoms', 'group_norm', or 'group_atoms'"
        )
    manifold_adapter = SAEBenchAdapter(
        manifold_sae_raw,
        match_token_norm=False,
        n_groups=int(getattr(manifold_sae_raw, "n_groups", 4096)),
        group_rank=int(getattr(manifold_sae_raw, "group_rank", 8)),
        use_group_norm=(manifold_feature_mode == "group_norm"),
    )

    hook_name = getattr(manifold_adapter.cfg, "hook_name", None)
    manifold_hook_layer = _infer_hook_layer_from_hook_name(hook_name)

    baseline_layer: int | None = None
    if baseline_sae_release:
        baseline_raw = SAE.from_pretrained(baseline_sae_release, hook_name, device=device)
        baseline_adapter = SAEBenchAdapter(baseline_raw, match_token_norm=False)
        baseline_layer = _infer_hook_layer_from_hook_name(getattr(baseline_adapter.cfg, "hook_name", None))
        baseline_name = f"baseline_{baseline_sae_release}"
    else:
        # Backwards-compatible Mistral baseline loader.
        from code_stuff.run_saebench_compare_mistral import (  # noqa: E402
            _resolve_downloaded_sae_path,
            load_downloaded_mistral_sae,
        )

        downloaded_hook_layer = (
            downloaded_sae_layer if downloaded_sae_layer is not None else manifold_hook_layer
        )
        resolved_downloaded_path, resolved_downloaded_layer = _resolve_downloaded_sae_path(
            downloaded_sae_path, downloaded_hook_layer
        )
        downloaded_sae_wrapper = load_downloaded_mistral_sae(
            str(resolved_downloaded_path),
            device,
            dtype,
            hook_layer=resolved_downloaded_layer,
        )
        baseline_adapter = SAEBenchAdapter(downloaded_sae_wrapper, match_token_norm=False)
        baseline_layer = resolved_downloaded_layer
        baseline_name = "baseline_mistral"

    model_tag = _resolve_model_name_from_cfgs(manifold_adapter.cfg, baseline_adapter.cfg) or "model"
    manifold_label = f"layer{manifold_hook_layer}" if manifold_hook_layer is not None else "layer_unknown"
    baseline_label = f"layer{baseline_layer}" if baseline_layer is not None else "layer_unknown"
    manifold_tag = (
        "group_norm"
        if manifold_feature_mode == "group_norm"
        else ("group_atoms" if manifold_feature_mode == "group_atoms" else "atoms")
    )
    selected_saes = [
        (
            f"manifold_{model_tag}_{manifold_label}_{manifold_tag}",
            _SaeProbesEncodeFloat32Adapter(manifold_adapter),
        ),
        (
            f"{baseline_name}_{model_tag}_{baseline_label}",
            _SaeProbesEncodeFloat32Adapter(baseline_adapter),
        ),
    ]
    return selected_saes, manifold_hook_layer, baseline_layer


def run_sparse_probing_sae_probes_eval(
    *,
    selected_saes: list[tuple[str, Any]],
    device: str,
    output_dir: Path,
    model_name: str,
    dataset_names: list[str] | None,
    ks: list[int],
    reg_type: str,
    setting: str,
    binarize: bool,
    include_llm_baseline: bool,
    baseline_method: str,
    results_path: str,
    model_cache_path: str | None,
    force_rerun: bool,
) -> dict[str, Any]:
    from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
        SparseProbingSaeProbesEvalConfig,
    )
    import sae_bench.evals.sparse_probing_sae_probes.main as sp_main

    output_path = output_dir / "sparse_probing_sae_probes"
    output_path.mkdir(parents=True, exist_ok=True)

    config = SparseProbingSaeProbesEvalConfig(
        model_name=model_name,
        reg_type=reg_type,
        setting=setting,
        ks=ks,
        binarize=binarize,
        results_path=results_path,
        model_cache_path=model_cache_path,
        include_llm_baseline=include_llm_baseline,
        baseline_method=baseline_method,
    )
    if dataset_names is not None:
        config.dataset_names = dataset_names

    sp_main.run_eval(
        config=config,
        selected_saes=selected_saes,
        device=device,
        output_path=str(output_path),
        force_rerun=force_rerun,
    )

    results: dict[str, Any] = {}
    for sae_name, _ in selected_saes:
        p = output_path / f"{sae_name}_custom_sae_eval_results.json"
        if p.exists():
            results[sae_name] = json.loads(p.read_text())
    return results


def _print_sparse_probing_sae_probes_summary(results: dict[str, Any], ks: list[int]) -> None:
    print("\n### Sparse probing (sae-probes) (higher is better)")
    for sae_name, result in results.items():
        metrics = result.get("eval_result_metrics", {}) or {}
        sae_metrics = metrics.get("sae", {}) if isinstance(metrics, dict) else {}
        llm_metrics = metrics.get("llm", {}) if isinstance(metrics, dict) else {}
        print(f"  {sae_name}:")
        if isinstance(llm_metrics, dict) and llm_metrics:
            print(f"    llm_test_accuracy: {llm_metrics.get('llm_test_accuracy', 'N/A')}")
        for k in ks:
            print(
                f"    sae_top_{k}_test_accuracy: "
                f"{sae_metrics.get(f'sae_top_{k}_test_accuracy', 'N/A')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifold-sae-dir",
        type=str,
        default=str(REPO_ROOT / "trained_saes/mistral_layer8_batchtopk_manifold_n4096_r8_k10"),
    )
    parser.add_argument(
        "--baseline-sae-release",
        type=str,
        default=None,
        help="Optional SAE-Lens release string (e.g. 'gpt2-small-res-jb'). If set, uses SAE.from_pretrained at the manifold SAE's hook.",
    )
    parser.add_argument(
        "--downloaded-sae-path",
        type=str,
        default=str(REPO_ROOT / "downloaded_saes/mistral_sae"),
    )
    parser.add_argument("--downloaded-sae-layer", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "saebench_results_mistral_comparison"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--manifold-feature-mode",
        type=str,
        default="atoms",
        choices=["atoms", "group_norm", "group_atoms"],
        help="How to present Manifold SAE features to k-sparse probing.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="sae-probes model name (e.g. 'gpt2' or 'mistral-7b'). If empty, inferred from SAE cfg.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=None,
        help="Optional sae-probes dataset short names; if omitted uses full DATASETS list.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--reg-type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument(
        "--setting",
        type=str,
        default="normal",
        choices=["normal", "scarcity", "imbalance"],
    )
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--include-llm-baseline", action="store_true", default=False)
    parser.add_argument("--baseline-method", type=str, default="logreg")
    parser.add_argument(
        "--results-path",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "sparse_probing_sae_probes"),
    )
    parser.add_argument(
        "--model-cache-path",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "sparse_probing_sae_probes--model_acts_cache"),
    )

    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    workspace_root = REPO_ROOT
    _set_hf_cache_defaults(workspace_root)

    try:
        register_sae_class(
            "batchtopk_manifold_sae",
            BatchTopKManifoldSAEInference,
            BatchTopKManifoldSAEInferenceConfig,
        )
    except ValueError:
        pass
    try:
        from sasa import TopKSASAInference, TopKSASAInferenceConfig

        register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)
    except (ImportError, ValueError):
        pass

    device = args.device
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_saes, manifold_layer, baseline_layer = load_saes(
        device=device,
        manifold_sae_dir=args.manifold_sae_dir,
        baseline_sae_release=args.baseline_sae_release,
        downloaded_sae_path=args.downloaded_sae_path,
        downloaded_sae_layer=args.downloaded_sae_layer,
        dtype=args.dtype,
        manifold_feature_mode=args.manifold_feature_mode,
    )

    # Prefer CLI model_name; fall back to SAE metadata if needed.
    model_name = args.model_name.strip() if isinstance(args.model_name, str) else ""
    if not model_name:
        inferred = _resolve_model_name_from_cfgs(selected_saes[0][1].cfg, selected_saes[1][1].cfg)  # type: ignore
        if inferred:
            model_name = inferred
    if not model_name:
        raise ValueError("--model-name is required (could not infer from SAE cfgs)")

    dataset_names = args.dataset if args.dataset is not None and len(args.dataset) > 0 else None

    if args.manifold_feature_mode == "group_atoms":
        if dataset_names is None:
            from sae_probes import DATASETS

            dataset_names = list(DATASETS)

        # Re-load manifold SAE to get n_groups / group_rank (SAE-Lens stores these on the module)
        manifold_raw = SAE.load_from_disk(args.manifold_sae_dir, device=device)
        n_groups = int(getattr(manifold_raw, "n_groups", 4096))
        group_rank = int(getattr(manifold_raw, "group_rank", 8))

        # Write manifold JSONs using group-wise selection (k groups -> k*rank atom dims)
        manifold_name, manifold_sae = selected_saes[0]
        hook_name = manifold_sae.cfg.hook_name
        manifold_results_root = str(Path(args.results_path) / f"{manifold_name}_custom_sae")
        _run_group_atoms_sae_probes_jsons(
            sae=manifold_sae,
            sae_results_path=manifold_results_root,
            model_name=model_name,
            hook_name=hook_name,
            datasets=dataset_names,
            ks_groups=args.ks,
            reg_type=args.reg_type,
            setting=args.setting,
            binarize=args.binarize,
            model_cache_path=args.model_cache_path,
            device=device,
            n_groups=n_groups,
            group_rank=group_rank,
        )

        # Run baseline normally (and manifold will be skipped since JSONs now exist)
        results = run_sparse_probing_sae_probes_eval(
            selected_saes=selected_saes,
            device=device,
            output_dir=out_dir,
            model_name=model_name,
            dataset_names=dataset_names,
            ks=args.ks,
            reg_type=args.reg_type,
            setting=args.setting,
            binarize=args.binarize,
            include_llm_baseline=args.include_llm_baseline,
            baseline_method=args.baseline_method,
            results_path=args.results_path,
            model_cache_path=args.model_cache_path,
            force_rerun=args.force_rerun,
        )
    else:
        results = run_sparse_probing_sae_probes_eval(
            selected_saes=selected_saes,
            device=device,
            output_dir=out_dir,
            model_name=model_name,
            dataset_names=dataset_names,
            ks=args.ks,
            reg_type=args.reg_type,
            setting=args.setting,
            binarize=args.binarize,
            include_llm_baseline=args.include_llm_baseline,
            baseline_method=args.baseline_method,
            results_path=args.results_path,
            model_cache_path=args.model_cache_path,
            force_rerun=args.force_rerun,
        )

    summary_path = out_dir / "sparse_probing_sae_probes" / "sparse_probing_sae_probes_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))

    _print_sparse_probing_sae_probes_summary(results, ks=args.ks)
    print(f"\nWrote results to: {out_dir / 'sparse_probing_sae_probes'}")
    if manifold_layer is not None and baseline_layer is not None and manifold_layer != baseline_layer:
        print(f"Warning: manifold layer {manifold_layer} != baseline layer {baseline_layer}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
