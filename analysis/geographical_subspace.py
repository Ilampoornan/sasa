#!/usr/bin/env python3
"""
Geography analysis for BatchTopK ManifoldSAE groups using RAVEL data.

This script:
1) Extracts group vectors at city tokens from RAVEL prompts.
2) Computes city->country and city/country->continent hierarchy metrics.
3) Renders paper-style figures (PCA scatter + hierarchy plots + token profiles).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from sae_lens import SAE, register_sae_class


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sasa.batchtopk_baseline import (
    BatchTopKManifoldSAEInference,
    BatchTopKManifoldSAEInferenceConfig,
)


@dataclass(frozen=True)
class CitySample:
    city: str
    country: str
    continent: str
    prompt: str


DEFAULT_PROFILE_PROMPTS = [
    "A storm delayed my flight to Chicago in the US.",
    "The Toronto skyline in Canada is impressive.",
    "Our team met in Paris in France before heading back from Europe.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cache_env(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    wandb_root = cache_root / "wandb"
    tmp_root = cache_root / "tmp"
    wandb_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    hf_home = cache_root / "hf"
    transformers_cache = cache_root / "hf_transformers"
    hub_cache = transformers_cache
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DIR"] = str(wandb_root)
    os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_root / "config")
    os.environ["WANDB_DATA_DIR"] = str(wandb_root / "data")
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    for d in ["hf", "hf_datasets", "hf_transformers"]:
        (cache_root / d).mkdir(parents=True, exist_ok=True)
    for d in ["cache", "config", "data"]:
        (wandb_root / d).mkdir(parents=True, exist_ok=True)


def get_hook_name(sae: SAE) -> str:
    if hasattr(sae.cfg, "hook_name") and sae.cfg.hook_name:
        return sae.cfg.hook_name
    if hasattr(sae.cfg, "metadata") and sae.cfg.metadata:
        if isinstance(sae.cfg.metadata, dict) and "hook_name" in sae.cfg.metadata:
            return sae.cfg.metadata["hook_name"]
        if hasattr(sae.cfg.metadata, "hook_name"):
            return getattr(sae.cfg.metadata, "hook_name")
    return "blocks.7.hook_resid_pre"


def normalize_token(token: str) -> str:
    token = token.replace("\u0120", " ").replace("\u2581", " ")
    token = token.strip().lower()
    token = token.strip(".,;:!?\"'()[]{}")
    return token


def is_single_token(tokenizer: AutoTokenizer, text: str) -> bool:
    if not text:
        return False
    leading_ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    if len(leading_ids) == 1:
        return True
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids) == 1


def build_geo_profile_prompts(
    samples: List[CitySample],
    tokenizer: AutoTokenizer,
    seed: int,
    n_prompts: int = 3,
) -> List[str]:
    if not samples:
        return []

    rng = random.Random(seed)
    templates = [
        "I visited {city}, {country} last summer; it's a favorite stop in {continent}.",
        "The {continent} tour started in {city}, and the local food in {country} was unforgettable.",
        "A storm delayed my flight to {city} in {country}, but the trip across {continent} was worth it.",
        "Our team met in {city} before crossing {continent}; the host country was {country}.",
    ]
    rng.shuffle(templates)

    shuffled = samples[:]
    rng.shuffle(shuffled)

    target_specs = [
        ("country", "United States"),
        ("country", "Canada"),
        ("continent", "Europe"),
    ]
    chosen: List[CitySample] = []
    seen = set()
    for field, value in target_specs[:n_prompts]:
        candidates = [
            s
            for s in shuffled
            if getattr(s, field) == value and s.city not in seen
        ]
        if candidates:
            sample = rng.choice(candidates)
            chosen.append(sample)
            seen.add(sample.city)

    if len(chosen) < n_prompts:
        strict = [
            s
            for s in shuffled
            if is_single_token(tokenizer, s.city)
            and is_single_token(tokenizer, s.country)
            and is_single_token(tokenizer, s.continent)
            and s.city not in seen
        ]
        relaxed = [
            s
            for s in shuffled
            if is_single_token(tokenizer, s.city)
            and is_single_token(tokenizer, s.country)
            and s.city not in seen
        ]
        pool = strict if len(strict) >= n_prompts else (relaxed if relaxed else shuffled)
        for sample in pool:
            if sample.city in seen:
                continue
            chosen.append(sample)
            seen.add(sample.city)
            if len(chosen) >= n_prompts:
                break

    prompts = []
    for idx, sample in enumerate(chosen[:n_prompts]):
        template = templates[idx % len(templates)]
        prompts.append(
            template.format(
                city=sample.city,
                country=sample.country,
                continent=sample.continent,
            )
        )
    return prompts


def load_model(
    model_name: str,
    device: str,
    model_from_pretrained_kwargs: dict | None,
    local_files_only: bool,
    tokenizer: AutoTokenizer | None,
) -> HookedTransformer:
    model_kwargs = dict(model_from_pretrained_kwargs or {})
    model_kwargs.setdefault("local_files_only", local_files_only)
    if tokenizer is not None:
        model_kwargs.setdefault("tokenizer", tokenizer)
    try:
        from sae_lens import HookedSAETransformer  # noqa: WPS433

        return HookedSAETransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            **model_kwargs,
        )
    except Exception:
        return HookedTransformer.from_pretrained(
            model_name,
            device=device,
            **model_kwargs,
        )


def resolve_cached_model_path(model_name: str, cache_root: Path) -> str | None:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)
    if "/" not in model_name:
        return None
    org, repo = model_name.split("/", 1)
    snapshots_dir = cache_root / "hf_transformers" / f"models--{org}--{repo}" / "snapshots"
    if not snapshots_dir.exists():
        return None
    candidates = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        if (cand / "config.json").exists():
            return str(cand)
    return None


def parse_city(prompt: str, mode: str) -> str:
    key = "country" if mode == "country" else "continent"
    marker = f"is a city in the {key} of"
    idx = prompt.rfind(marker)
    if idx < 0:
        raise ValueError(f"Failed to parse city in prompt: {prompt}")
    prefix = prompt[:idx].strip()
    city = prefix.split()[-1].strip(".,;:!?\"'()[]{}")
    if not city:
        raise ValueError(f"Empty city parsed from prompt: {prompt}")
    return city


def load_ravel_city_samples(
    country_path: Path,
    continent_path: Path,
    seed: int,
    max_cities: int | None,
) -> List[CitySample]:
    country_data = json.loads(country_path.read_text())
    continent_data = json.loads(continent_path.read_text())

    city_to_country: Dict[str, Tuple[str, str]] = {}
    for prompt, label in country_data:
        city = parse_city(prompt, "country")
        city_to_country[city] = (label, prompt)

    city_to_continent: Dict[str, str] = {}
    for prompt, label in continent_data:
        city = parse_city(prompt, "continent")
        city_to_continent[city] = label

    cities = sorted(set(city_to_country) & set(city_to_continent))
    samples = [
        CitySample(
            city=city,
            country=city_to_country[city][0],
            continent=city_to_continent[city],
            prompt=city_to_country[city][1],
        )
        for city in cities
    ]

    if max_cities is not None and max_cities > 0 and len(samples) > max_cities:
        rng = random.Random(seed)
        rng.shuffle(samples)
        samples = samples[:max_cities]

    return samples


def find_subsequence_positions(sequence: Sequence[int], subseq: Sequence[int]) -> List[int]:
    if not subseq:
        return []
    positions = []
    max_start = len(sequence) - len(subseq)
    for i in range(max_start + 1):
        if sequence[i : i + len(subseq)] == list(subseq):
            positions.append(i)
    return positions


def collect_city_vectors(
    samples: List[CitySample],
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    representation: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[str]]:
    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    city_vectors: Dict[str, np.ndarray] = {}
    city_norms: Dict[str, float] = {}
    missing_cities: List[str] = []

    city_token_ids: Dict[str, List[int]] = {}
    for sample in samples:
        tok_ids = tokenizer.encode(" " + sample.city, add_special_tokens=False)
        if not tok_ids:
            tok_ids = tokenizer.encode(sample.city, add_special_tokens=False)
        city_token_ids[sample.city] = tok_ids

    start = group_id * group_rank
    end = start + group_rank
    w_dec_group = sae.W_dec[start:end]

    for start_idx in range(0, len(samples), batch_size):
        batch_samples = samples[start_idx : start_idx + batch_size]
        prompts = [s.prompt for s in batch_samples]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
            resid = cache[hook_name]
            sae_acts = sae.encode(resid)

        batch, seq, _ = sae_acts.shape
        sae_acts = sae_acts.view(batch, seq, n_groups, group_rank)
        group_acts = sae_acts[:, :, group_id, :]
        if representation == "recon":
            group_vecs = torch.einsum("bsg,gd->bsd", group_acts, w_dec_group)
        else:
            group_vecs = group_acts
        group_norms = group_acts.norm(dim=-1)

        input_ids_cpu = input_ids.detach().cpu().tolist()
        attention_cpu = attention.detach().cpu().tolist()
        group_vecs_cpu = group_vecs.detach().cpu()
        group_norms_cpu = group_norms.detach().cpu()

        for i, sample in enumerate(batch_samples):
            tok_ids = city_token_ids.get(sample.city, [])
            if not tok_ids:
                missing_cities.append(sample.city)
                continue
            positions = find_subsequence_positions(input_ids_cpu[i], tok_ids)
            if not positions:
                # fallback to no-leading-space tokenization
                tok_ids = tokenizer.encode(sample.city, add_special_tokens=False)
                positions = find_subsequence_positions(input_ids_cpu[i], tok_ids)
            if not positions:
                missing_cities.append(sample.city)
                continue
            pos = positions[-1]
            span = range(pos, pos + len(tok_ids))
            if any(attention_cpu[i][j] == 0 for j in span):
                missing_cities.append(sample.city)
                continue
            vec = group_vecs_cpu[i, pos : pos + len(tok_ids), :].mean(dim=0)
            norm = float(group_norms_cpu[i, pos : pos + len(tok_ids)].mean().item())
            city_vectors[sample.city] = vec.numpy()
            city_norms[sample.city] = norm

    return city_vectors, city_norms, missing_cities


def pick_position_after(positions: List[int], start_after: int) -> int | None:
    for pos in positions:
        if pos > start_after:
            return pos
    return None


def collect_category_norms(
    samples: List[CitySample],
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, List[float]], Dict[str, int], Dict[str, Dict[str, float]]]:
    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    prompts = [
        f"City: {s.city}. Country: {s.country}. Continent: {s.continent}."
        for s in samples
    ]
    token_ids: Dict[str, Dict[str, List[int]]] = {}
    for sample in samples:
        token_ids[sample.city] = {
            "city": tokenizer.encode(f" {sample.city}", add_special_tokens=False)
            or tokenizer.encode(sample.city, add_special_tokens=False),
            "country": tokenizer.encode(f" {sample.country}", add_special_tokens=False)
            or tokenizer.encode(sample.country, add_special_tokens=False),
            "continent": tokenizer.encode(f" {sample.continent}", add_special_tokens=False)
            or tokenizer.encode(sample.continent, add_special_tokens=False),
        }

    norms_by_type: Dict[str, List[float]] = {"city": [], "country": [], "continent": []}
    city_norm_by_city: Dict[str, float] = {}
    country_norms_by_country: Dict[str, List[float]] = {}
    continent_norms_by_continent: Dict[str, List[float]] = {}
    missing_counts = {"city": 0, "country": 0, "continent": 0}

    for start_idx in range(0, len(samples), batch_size):
        batch_samples = samples[start_idx : start_idx + batch_size]
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
            resid = cache[hook_name]
            acts = sae.encode(resid)

        batch, seq, _ = acts.shape
        acts = acts.view(batch, seq, n_groups, group_rank)
        group_acts = acts[:, :, group_id, :]
        group_norms = group_acts.norm(dim=-1)

        input_ids_cpu = input_ids.detach().cpu().tolist()
        attention_cpu = attention.detach().cpu().tolist()
        group_norms_cpu = group_norms.detach().cpu()

        for i, sample in enumerate(batch_samples):
            ids = token_ids.get(sample.city)
            if not ids:
                continue
            city_ids = ids["city"]
            country_ids = ids["country"]
            continent_ids = ids["continent"]

            city_positions = find_subsequence_positions(input_ids_cpu[i], city_ids)
            if not city_positions:
                missing_counts["city"] += 1
                continue
            city_pos = pick_position_after(city_positions, -1)
            if city_pos is None:
                missing_counts["city"] += 1
                continue

            country_positions = find_subsequence_positions(input_ids_cpu[i], country_ids)
            if not country_positions:
                missing_counts["country"] += 1
                continue
            country_pos = pick_position_after(country_positions, city_pos + len(city_ids) - 1)
            if country_pos is None:
                missing_counts["country"] += 1
                continue

            continent_positions = find_subsequence_positions(input_ids_cpu[i], continent_ids)
            if not continent_positions:
                missing_counts["continent"] += 1
                continue
            continent_pos = pick_position_after(continent_positions, country_pos + len(country_ids) - 1)
            if continent_pos is None:
                missing_counts["continent"] += 1
                continue

            spans = {
                "city": (city_pos, len(city_ids)),
                "country": (country_pos, len(country_ids)),
                "continent": (continent_pos, len(continent_ids)),
            }
            for key, (pos, length) in spans.items():
                span = range(pos, pos + length)
                if any(attention_cpu[i][j] == 0 for j in span):
                    missing_counts[key] += 1
                    continue
                norm = float(group_norms_cpu[i, pos : pos + length].mean().item())
                norms_by_type[key].append(norm)
                if key == "city":
                    city_norm_by_city[sample.city] = norm
                elif key == "country":
                    country_norms_by_country.setdefault(sample.country, []).append(norm)
                elif key == "continent":
                    continent_norms_by_continent.setdefault(sample.continent, []).append(norm)

    country_norm_by_country = {
        country: float(np.mean(vals)) for country, vals in country_norms_by_country.items() if vals
    }
    continent_norm_by_continent = {
        cont: float(np.mean(vals)) for cont, vals in continent_norms_by_continent.items() if vals
    }
    norm_maps = {
        "city": city_norm_by_city,
        "country": country_norm_by_country,
        "continent": continent_norm_by_continent,
    }

    return norms_by_type, missing_counts, norm_maps


def collect_category_vectors(
    samples: List[CitySample],
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    representation: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    prompts = [
        f"City: {s.city}. Country: {s.country}. Continent: {s.continent}."
        for s in samples
    ]
    token_ids: Dict[str, Dict[str, List[int]]] = {}
    for sample in samples:
        token_ids[sample.city] = {
            "city": tokenizer.encode(f" {sample.city}", add_special_tokens=False)
            or tokenizer.encode(sample.city, add_special_tokens=False),
            "country": tokenizer.encode(f" {sample.country}", add_special_tokens=False)
            or tokenizer.encode(sample.country, add_special_tokens=False),
            "continent": tokenizer.encode(f" {sample.continent}", add_special_tokens=False)
            or tokenizer.encode(sample.continent, add_special_tokens=False),
        }

    start = group_id * group_rank
    end = start + group_rank
    w_dec_group = sae.W_dec[start:end]

    city_vecs_acc: Dict[str, List[np.ndarray]] = {}
    country_vecs_acc: Dict[str, List[np.ndarray]] = {}
    continent_vecs_acc: Dict[str, List[np.ndarray]] = {}
    missing_counts = {"city": 0, "country": 0, "continent": 0}

    for start_idx in range(0, len(samples), batch_size):
        batch_samples = samples[start_idx : start_idx + batch_size]
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
            resid = cache[hook_name]
            acts = sae.encode(resid)

        batch, seq, _ = acts.shape
        acts = acts.view(batch, seq, n_groups, group_rank)
        group_acts = acts[:, :, group_id, :]
        if representation == "recon":
            group_vecs = torch.einsum("bsg,gd->bsd", group_acts, w_dec_group)
        else:
            group_vecs = group_acts

        input_ids_cpu = input_ids.detach().cpu().tolist()
        attention_cpu = attention.detach().cpu().tolist()
        group_vecs_cpu = group_vecs.detach().cpu()

        for i, sample in enumerate(batch_samples):
            ids = token_ids.get(sample.city)
            if not ids:
                continue
            city_ids = ids["city"]
            country_ids = ids["country"]
            continent_ids = ids["continent"]

            city_positions = find_subsequence_positions(input_ids_cpu[i], city_ids)
            if not city_positions:
                missing_counts["city"] += 1
                continue
            city_pos = pick_position_after(city_positions, -1)
            if city_pos is None:
                missing_counts["city"] += 1
                continue

            country_positions = find_subsequence_positions(input_ids_cpu[i], country_ids)
            if not country_positions:
                missing_counts["country"] += 1
                continue
            country_pos = pick_position_after(country_positions, city_pos + len(city_ids) - 1)
            if country_pos is None:
                missing_counts["country"] += 1
                continue

            continent_positions = find_subsequence_positions(input_ids_cpu[i], continent_ids)
            if not continent_positions:
                missing_counts["continent"] += 1
                continue
            continent_pos = pick_position_after(continent_positions, country_pos + len(country_ids) - 1)
            if continent_pos is None:
                missing_counts["continent"] += 1
                continue

            spans = {
                "city": (city_pos, len(city_ids)),
                "country": (country_pos, len(country_ids)),
                "continent": (continent_pos, len(continent_ids)),
            }
            for key, (pos, length) in spans.items():
                span = range(pos, pos + length)
                if any(attention_cpu[i][j] == 0 for j in span):
                    missing_counts[key] += 1
                    continue
                vec = group_vecs_cpu[i, pos : pos + length, :].mean(dim=0).numpy()
                if key == "city":
                    city_vecs_acc.setdefault(sample.city, []).append(vec)
                elif key == "country":
                    country_vecs_acc.setdefault(sample.country, []).append(vec)
                elif key == "continent":
                    continent_vecs_acc.setdefault(sample.continent, []).append(vec)

    city_vecs = {
        city: np.mean(np.stack(vecs, axis=0), axis=0) for city, vecs in city_vecs_acc.items() if vecs
    }
    country_vecs = {
        country: np.mean(np.stack(vecs, axis=0), axis=0)
        for country, vecs in country_vecs_acc.items()
        if vecs
    }
    continent_vecs = {
        cont: np.mean(np.stack(vecs, axis=0), axis=0)
        for cont, vecs in continent_vecs_acc.items()
        if vecs
    }
    return city_vecs, country_vecs, continent_vecs, missing_counts


def compute_norm_discrimination(norms_by_type: Dict[str, List[float]]) -> Dict[str, object]:
    stats = {}
    for key, values in norms_by_type.items():
        arr = np.array(values, dtype=float)
        stats[key] = {
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "std": float(np.std(arr)) if arr.size else 0.0,
            "n": int(arr.size),
        }

    def pairwise_auc(a: List[float], b: List[float]) -> float | None:
        if len(a) < 2 or len(b) < 2:
            return None
        labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))], axis=0)
        scores = np.concatenate([np.array(a), np.array(b)], axis=0)
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return None

    aucs = {
        "city_vs_country": pairwise_auc(norms_by_type["city"], norms_by_type["country"]),
        "city_vs_continent": pairwise_auc(norms_by_type["city"], norms_by_type["continent"]),
        "country_vs_continent": pairwise_auc(norms_by_type["country"], norms_by_type["continent"]),
    }

    return {"stats": stats, "pairwise_auc": aucs}


def plot_norm_discrimination(
    output_dir: Path,
    group_id: int,
    norms_by_type: Dict[str, List[float]],
) -> Dict[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    labels = ["City", "Country", "Continent"]
    data = [norms_by_type["city"], norms_by_type["country"], norms_by_type["continent"]]
    colors = ["#8FA3B8", "#F28E2B", "#59A14F"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    ax.set_title(f"Group {group_id} norm discrimination")
    ax.set_ylabel("Group norm")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    png_path = output_dir / f"group_{group_id}_geo_norm_discrimination.png"
    pdf_path = output_dir / f"group_{group_id}_geo_norm_discrimination.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}
def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def hierarchy_metrics(
    city_vecs: Dict[str, np.ndarray],
    country_vecs: Dict[str, np.ndarray],
    city_to_country: Dict[str, str],
) -> Dict[str, float]:
    cities = [c for c in city_vecs if city_to_country.get(c) in country_vecs]
    countries = list(country_vecs.keys())
    if not cities or not countries:
        return {
            "city_country_top1": 0.0,
            "city_country_top5": 0.0,
            "city_country_mrr": 0.0,
            "mean_city_country_delta": 0.0,
        }

    city_mat = np.stack([city_vecs[c] for c in cities], axis=0)
    country_mat = np.stack([country_vecs[c] for c in countries], axis=0)
    sims = cosine_sim_matrix(city_mat, country_mat)

    top1 = 0
    top5 = 0
    mrr_sum = 0.0
    deltas = []

    for i, city in enumerate(cities):
        correct_country = city_to_country[city]
        if correct_country not in country_vecs:
            continue
        correct_idx = countries.index(correct_country)
        ranking = np.argsort(sims[i])[::-1]
        rank = int(np.where(ranking == correct_idx)[0][0]) + 1
        if rank == 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
        mrr_sum += 1.0 / rank

        correct_sim = sims[i, correct_idx]
        other_sim = np.delete(sims[i], correct_idx)
        deltas.append(float(correct_sim - other_sim.mean()))

    n = len(cities)
    return {
        "city_country_top1": float(top1 / max(n, 1)),
        "city_country_top5": float(top5 / max(n, 1)),
        "city_country_mrr": float(mrr_sum / max(n, 1)),
        "mean_city_country_delta": float(np.mean(deltas) if deltas else 0.0),
    }


def continent_metrics(
    country_vecs: Dict[str, np.ndarray],
    city_vecs: Dict[str, np.ndarray],
    country_to_continent: Dict[str, str],
    city_to_country: Dict[str, str],
) -> Dict[str, float]:
    continent_to_vecs: Dict[str, List[np.ndarray]] = {}
    for country, vec in country_vecs.items():
        cont = country_to_continent.get(country)
        if cont is None:
            continue
        continent_to_vecs.setdefault(cont, []).append(vec)

    continent_vecs = {
        cont: np.mean(np.stack(vecs, axis=0), axis=0)
        for cont, vecs in continent_to_vecs.items()
        if vecs
    }

    continents = list(continent_vecs.keys())
    if not continents:
        return {
            "country_continent_top1": 0.0,
            "city_continent_top1": 0.0,
            "continent_within_minus_between": 0.0,
        }

    countries = [c for c in country_vecs if c in country_to_continent]
    country_mat = np.stack([country_vecs[c] for c in countries], axis=0)
    cont_mat = np.stack([continent_vecs[c] for c in continents], axis=0)
    sims_country = cosine_sim_matrix(country_mat, cont_mat)

    top1_country = 0
    for i, country in enumerate(countries):
        cont = country_to_continent[country]
        correct_idx = continents.index(cont)
        rank = int(np.where(np.argsort(sims_country[i])[::-1] == correct_idx)[0][0]) + 1
        if rank == 1:
            top1_country += 1

    cities = [c for c in city_vecs if city_to_country.get(c) in country_to_continent]
    city_mat = np.stack([city_vecs[c] for c in cities], axis=0)
    sims_city = cosine_sim_matrix(city_mat, cont_mat)

    top1_city = 0
    for i, city in enumerate(cities):
        country = city_to_country[city]
        cont = country_to_continent[country]
        correct_idx = continents.index(cont)
        rank = int(np.where(np.argsort(sims_city[i])[::-1] == correct_idx)[0][0]) + 1
        if rank == 1:
            top1_city += 1

    sims_cc = cosine_sim_matrix(country_mat, country_mat)
    within: List[float] = []
    between: List[float] = []
    for i in range(len(countries)):
        for j in range(i + 1, len(countries)):
            ci = country_to_continent[countries[i]]
            cj = country_to_continent[countries[j]]
            if ci == cj:
                within.append(float(sims_cc[i, j]))
            else:
                between.append(float(sims_cc[i, j]))

    within_mean = float(np.mean(within) if within else 0.0)
    between_mean = float(np.mean(between) if between else 0.0)
    return {
        "country_continent_top1": float(top1_country / max(len(countries), 1)),
        "city_continent_top1": float(top1_city / max(len(cities), 1)),
        "continent_within_mean": within_mean,
        "continent_between_mean": between_mean,
        "continent_within_minus_between": float(within_mean - between_mean),
    }


def neighbor_metrics(
    city_vecs: Dict[str, np.ndarray],
    city_to_country: Dict[str, str],
    city_to_continent: Dict[str, str],
    ks: Iterable[int] = (1, 3, 5),
) -> Dict[str, object]:
    cities = list(city_vecs.keys())
    if not cities:
        return {
            "same_country_at_k": {},
            "same_continent_at_k": {},
            "n_cities": 0,
        }
    city_mat = np.stack([city_vecs[c] for c in cities], axis=0)
    normed = city_mat / (np.linalg.norm(city_mat, axis=1, keepdims=True) + 1e-8)
    sims = normed @ normed.T
    np.fill_diagonal(sims, -np.inf)

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    same_country = {str(k): 0 for k in ks}
    same_continent = {str(k): 0 for k in ks}

    for i, city in enumerate(cities):
        country = city_to_country.get(city)
        continent = city_to_continent.get(city)
        if country is None or continent is None:
            continue
        row = sims[i]
        order = np.argpartition(-row, ks[-1])[: ks[-1]]
        order = order[np.argsort(-row[order])]
        for k in ks:
            neighbors = order[:k]
            same_country[str(k)] += sum(
                1 for idx in neighbors if city_to_country.get(cities[idx]) == country
            )
            same_continent[str(k)] += sum(
                1 for idx in neighbors if city_to_continent.get(cities[idx]) == continent
            )

    n = len(cities)
    return {
        "same_country_at_k": {k: float(v) / max(n * int(k), 1) for k, v in same_country.items()},
        "same_continent_at_k": {k: float(v) / max(n * int(k), 1) for k, v in same_continent.items()},
        "n_cities": n,
    }


def compute_group_norms_for_prompts(
    prompts: List[str],
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    max_length: int,
) -> Tuple[List[List[str]], List[np.ndarray]]:
    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        resid = cache[hook_name]
        use_pre_acts = False
        if hasattr(sae, "W_enc") and hasattr(sae, "b_enc"):
            try:
                sae_in = sae.process_sae_in(resid)
                pre_acts = sae_in @ sae.W_enc + sae.b_enc  # type: ignore[operator]
                acts = pre_acts
                use_pre_acts = True
            except Exception:
                acts = sae.encode(resid)
        else:
            acts = sae.encode(resid)

    batch, seq, _ = acts.shape
    if use_pre_acts and acts.shape[-1] != n_groups * group_rank:
        acts = sae.encode(resid)
    acts = acts.view(batch, seq, n_groups, group_rank)
    group_acts = acts[:, :, group_id, :]
    group_norms = group_acts.norm(dim=-1).detach().cpu().numpy()

    input_ids_cpu = input_ids.detach().cpu().tolist()
    attention_cpu = attention.detach().cpu().tolist()

    token_lists: List[List[str]] = []
    norm_lists: List[np.ndarray] = []
    for b in range(batch):
        tokens = []
        norms = []
        for t in range(seq):
            if attention_cpu[b][t] == 0:
                continue
            tok = tokenizer.decode([int(input_ids_cpu[b][t])])
            tokens.append(tok)
            norms.append(group_norms[b, t])
        token_lists.append(tokens)
        norm_lists.append(np.array(norms))
    return token_lists, norm_lists


def plot_geo_activation_profiles(
    output_dir: Path,
    group_id: int,
    prompts: List[str],
    token_lists: List[List[str]],
    norm_lists: List[np.ndarray],
    city_set: set[str],
    country_set: set[str],
    continent_set: set[str],
    token_category_overrides: Dict[str, str] | None = None,
) -> Dict[str, str]:
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.2,
        }
    )

    colors = {
        "city": "#1f77b4",
        "country": "#ff7f0e",
        "continent": "#2ca02c",
    }
    line_color = "#1f1f1f"
    text_effects = [pe.withStroke(linewidth=2.0, foreground="white")]

    fig, axes = plt.subplots(len(prompts), 1, figsize=(13.0, 3.2 * len(prompts)))
    if len(prompts) == 1:
        axes = [axes]

    overrides = {k.lower(): v for k, v in (token_category_overrides or {}).items()}

    for idx, ax in enumerate(axes):
        tokens = token_lists[idx]
        norms = norm_lists[idx]
        ax.plot(range(len(norms)), norms, color=line_color, linewidth=2.2, alpha=0.85)
        for pos, tok in enumerate(tokens):
            token_norm = normalize_token(tok)
            category = overrides.get(token_norm)
            if category not in colors:
                category = None
                if token_norm in city_set:
                    category = "city"
                elif token_norm in country_set:
                    category = "country"
                elif token_norm in continent_set:
                    category = "continent"
            if category is None:
                continue
            ax.scatter(
                pos,
                norms[pos],
                color=colors[category],
                s=70,
                zorder=3,
                edgecolors="white",
                linewidths=0.9,
            )
            ax.annotate(
                token_norm if len(token_norm) <= 10 else token_norm[:10],
                xy=(pos, norms[pos]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=colors[category],
                path_effects=text_effects,
                zorder=4,
            )
        ax.set_title(prompts[idx], fontweight="semibold")
        ax.set_ylabel("Group norm", fontweight="semibold")
        ax.set_xlabel("Token position", fontweight="semibold")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=1.1)
        ax.margins(y=0.25)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label.title(),
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
        )
        for label, color in colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(colors),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.95),
    )

    fig.suptitle(
        f"Geo Token Activation Profiles (Group {group_id})",
        y=0.995,
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    png_path = output_dir / f"group_{group_id}_geo_token_profiles.png"
    pdf_path = output_dir / f"group_{group_id}_geo_token_profiles.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def plot_geo_structure(
    output_dir: Path,
    group_id: int,
    city_vecs: Dict[str, np.ndarray],
    city_to_continent: Dict[str, str],
    country_vecs: Dict[str, np.ndarray],
    country_to_continent: Dict[str, str],
    city_to_country: Dict[str, str],
    continent_vecs: Dict[str, np.ndarray] | None,
    city_norms: Dict[str, float] | None,
    country_norms: Dict[str, float] | None,
    metrics: Dict[str, float],
    seed: int,
    include_continents: bool = False,
    output_suffix: str = "",
) -> Dict[str, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    if not city_vecs or not country_vecs:
        return {}

    cities = list(city_vecs.keys())
    countries = list(country_vecs.keys())
    city_mat = np.stack([city_vecs[c] for c in cities], axis=0)
    country_mat = np.stack([country_vecs[c] for c in countries], axis=0)

    fit_mat = np.concatenate([city_mat, country_mat], axis=0)
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_mat)
    pca = PCA(n_components=2, random_state=seed)
    fit_proj = pca.fit_transform(fit_scaled)
    explained = pca.explained_variance_ratio_

    city_proj = fit_proj[: len(cities)]
    country_proj = fit_proj[len(cities) :]
    cont_proj = None
    cont_names: List[str] = []
    if include_continents and continent_vecs:
        cont_names = list(continent_vecs.keys())
        cont_mat = np.stack([continent_vecs[c] for c in cont_names], axis=0)
        cont_proj = pca.transform(scaler.transform(cont_mat))

    max_city_points = 2000
    city_plot = city_proj
    if city_proj.shape[0] > max_city_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(city_proj.shape[0], size=max_city_points, replace=False)
        city_plot = city_proj[idx]

    fig = plt.figure(figsize=(7.8, 5.6))
    ax0 = fig.add_subplot(1, 1, 1)

    ax0.scatter(
        city_plot[:, 0],
        city_plot[:, 1],
        s=8,
        alpha=0.25,
        color="#4C78A8",
        label="City",
        zorder=1,
    )
    ax0.scatter(
        country_proj[:, 0],
        country_proj[:, 1],
        s=28,
        alpha=0.7,
        color="#F28E2B",
        marker="o",
        edgecolors="black",
        linewidths=0.4,
        label="Country",
        zorder=2,
    )
    if cont_proj is not None:
        ax0.scatter(
            cont_proj[:, 0],
            cont_proj[:, 1],
            s=40,
            alpha=0.8,
            color="#59A14F",
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            label="Continent",
            zorder=3,
        )

    ax0.set_title("Geographical catergories")
    ax0.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax0.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax0.grid(True, alpha=0.2, linewidth=0.5)
    ax0.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    png_path = output_dir / f"group_{group_id}_geo_structure{output_suffix}.png"
    pdf_path = output_dir / f"group_{group_id}_geo_structure{output_suffix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "pca_explained_var_1": float(explained[0]),
        "pca_explained_var_2": float(explained[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geo analysis for a group using RAVEL city prompts."
    )
    parser.add_argument("--sae-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--group-id", type=int, default=1570)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-cities", type=int, default=None)
    parser.add_argument("--representation", type=str, choices=["latent", "recon"], default="latent")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(Path(os.environ.get("SASA_CACHE_ROOT", Path.home() / ".cache" / "sasa"))),
        help="HF/transformers cache root for model files.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        default=True,
        help="Allow downloads if model files are missing locally.",
    )
    parser.add_argument(
        "--ravel-data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "ravel"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    set_cache_env(Path(args.cache_root))
    if args.allow_downloads:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    register_sae_class(
        "batchtopk_manifold_sae",
        BatchTopKManifoldSAEInference,
        BatchTopKManifoldSAEInferenceConfig,
    )

    sae = SAE.load_from_disk(args.sae_dir, device=device)
    sae.eval()

    model_name = "gpt2"
    model_kwargs = {}
    if hasattr(sae.cfg, "metadata") and sae.cfg.metadata:
        if isinstance(sae.cfg.metadata, dict):
            model_name = sae.cfg.metadata.get("model_name", model_name)
        else:
            model_name = getattr(sae.cfg.metadata, "model_name", model_name)
    if hasattr(sae.cfg, "model_from_pretrained_kwargs") and sae.cfg.model_from_pretrained_kwargs:
        model_kwargs = dict(sae.cfg.model_from_pretrained_kwargs)
    model_kwargs.setdefault("cache_dir", str(Path(args.cache_root) / "hf_transformers"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_kwargs.get("cache_dir"),
        local_files_only=not args.allow_downloads,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(
        model_name,
        device=device,
        model_from_pretrained_kwargs=model_kwargs,
        local_files_only=not args.allow_downloads,
        tokenizer=tokenizer,
    )
    model.eval()
    hook_name = get_hook_name(sae)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1]
        / "output"
        / f"group_{args.group_id}_geo_ravel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ravel_dir = Path(args.ravel_data_dir)
    samples = load_ravel_city_samples(
        ravel_dir / "country_data.json",
        ravel_dir / "continent_data.json",
        seed=args.seed,
        max_cities=args.max_cities,
    )

    city_vecs, city_norms, missing = collect_city_vectors(
        samples=samples,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        group_id=args.group_id,
        hook_name=hook_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        representation=args.representation,
    )

    city_to_country = {s.city: s.country for s in samples if s.city in city_vecs}
    city_to_continent = {s.city: s.continent for s in samples if s.city in city_vecs}

    country_to_cities: Dict[str, List[str]] = {}
    for city, country in city_to_country.items():
        country_to_cities.setdefault(country, []).append(city)

    country_vecs = {
        country: np.mean(np.stack([city_vecs[c] for c in cities], axis=0), axis=0)
        for country, cities in country_to_cities.items()
        if cities
    }
    country_to_continent: Dict[str, str] = {}
    for city, country in city_to_country.items():
        if country not in country_to_continent:
            country_to_continent[country] = city_to_continent.get(city, "Unknown")

    metrics = {}
    metrics.update(hierarchy_metrics(city_vecs, country_vecs, city_to_country))
    metrics.update(continent_metrics(country_vecs, city_vecs, country_to_continent, city_to_country))
    neighbor = neighbor_metrics(city_vecs, city_to_country, city_to_continent)

    norms_by_type, norm_missing, norm_maps = collect_category_norms(
        samples=samples,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        group_id=args.group_id,
        hook_name=hook_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    norm_disc = compute_norm_discrimination(norms_by_type)
    norm_plot = plot_norm_discrimination(
        output_dir=output_dir,
        group_id=args.group_id,
        norms_by_type=norms_by_type,
    )

    prompt_city_vecs, prompt_country_vecs, prompt_continent_vecs, prompt_missing = collect_category_vectors(
        samples=samples,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        group_id=args.group_id,
        hook_name=hook_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        representation=args.representation,
    )

    fig_info = plot_geo_structure(
        output_dir=output_dir,
        group_id=args.group_id,
        city_vecs=prompt_city_vecs,
        city_to_continent=city_to_continent,
        country_vecs=prompt_country_vecs,
        country_to_continent=country_to_continent,
        city_to_country=city_to_country,
        continent_vecs=None,
        city_norms=norm_maps.get("city", {}),
        country_norms=norm_maps.get("country", {}),
        metrics=metrics,
        seed=args.seed,
        include_continents=False,
    )
    fig_info_all = plot_geo_structure(
        output_dir=output_dir,
        group_id=args.group_id,
        city_vecs=prompt_city_vecs,
        city_to_continent=city_to_continent,
        country_vecs=prompt_country_vecs,
        country_to_continent=country_to_continent,
        city_to_country=city_to_country,
        continent_vecs=prompt_continent_vecs,
        city_norms=norm_maps.get("city", {}),
        country_norms=norm_maps.get("country", {}),
        metrics=metrics,
        seed=args.seed,
        include_continents=True,
        output_suffix="_all",
    )

    # Token activation profiles using fixed examples.
    qual_prompts = list(DEFAULT_PROFILE_PROMPTS)
    if not qual_prompts:
        qual_prompts = build_geo_profile_prompts(
            samples=samples,
            tokenizer=tokenizer,
            seed=args.seed,
            n_prompts=3,
        )
    if not qual_prompts and samples:
        sample = samples[0]
        qual_prompts = [
            f"{sample.city} is in {sample.country} in {sample.continent}."
        ]

    token_lists, norm_lists = compute_group_norms_for_prompts(
        prompts=qual_prompts,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        group_id=args.group_id,
        hook_name=hook_name,
        device=device,
        max_length=args.max_length,
    )

    city_set = {normalize_token(c) for c in city_to_country}
    country_set = {normalize_token(c) for c in country_to_cities}
    continent_set = {normalize_token(c) for c in set(city_to_continent.values())}

    profile_info = plot_geo_activation_profiles(
        output_dir=output_dir,
        group_id=args.group_id,
        prompts=qual_prompts,
        token_lists=token_lists,
        norm_lists=norm_lists,
        city_set=city_set,
        country_set=country_set,
        continent_set=continent_set,
        token_category_overrides={"us": "continent"},
    )

    report = {
        "group_id": args.group_id,
        "sae_dir": args.sae_dir,
        "model_name": model_name,
        "hook_name": hook_name,
        "device": device,
        "representation": args.representation,
        "counts": {
            "cities_total": len(samples),
            "cities_used": len(city_vecs),
            "countries": len(country_vecs),
            "continents": len(set(city_to_continent.values())),
            "missing_cities": missing,
        },
        "metrics": metrics,
        "neighbors": neighbor,
        "norm_discrimination": {
            "stats": norm_disc["stats"],
            "pairwise_auc": norm_disc["pairwise_auc"],
            "missing_counts": norm_missing,
            "prompt_vector_missing": prompt_missing,
            "figure": norm_plot,
        },
        "profile_prompts": qual_prompts,
        "figures": {
            "structure": fig_info,
            "structure_all": fig_info_all,
            "token_profiles": profile_info,
        },
    }

    summary_path = output_dir / f"group_{args.group_id}_geo_ravel_summary.json"
    summary_path.write_text(json.dumps(report, indent=2))

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
