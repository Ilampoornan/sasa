#!/usr/bin/env python3
"""
Paper-style temporal concept analysis for a single TopKSASA group.

This script follows the PCA-style visualization approach in 2405.14860v3:
- Build controlled prompts for days, months, and years.
- Extract group activations (latent) or decoder reconstructions at target tokens.
- Run PCA and visualize ring-like structure for cyclic concepts.
- Summarize trends for years (linear or circular alignment).

Outputs a paper-ready multi-panel figure and a concise report.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import pairwise_distances, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.manifold import Isomap, MDS, SpectralEmbedding
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from sae_lens import SAE, register_sae_class


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sasa import TopKSASAInference, TopKSASAInferenceConfig



DAYS = [
    ("Monday", 0),
    ("Tuesday", 1),
    ("Wednesday", 2),
    ("Thursday", 3),
    ("Friday", 4),
    ("Saturday", 5),
    ("Sunday", 6),
]
MONTHS = [
    ("January", 1),
    ("February", 2),
    ("March", 3),
    ("April", 4),
    ("May", 5),
    ("June", 6),
    ("July", 7),
    ("August", 8),
    ("September", 9),
    ("October", 10),
    ("November", 11),
    ("December", 12),
]
DAY_ALIASES: dict[int, list[str]] = {
    0: ["Mon", "Mondays"],
    1: ["Tue", "Tues", "Tuesdays"],
    2: ["Wed", "Weds", "Wednesdays"],
    3: ["Thu", "Thur", "Thurs", "Thursdays"],
    4: ["Fri", "Fridays"],
    5: ["Sat", "Saturdays"],
    6: ["Sun", "Sundays"],
}
MONTH_ALIASES: dict[int, list[str]] = {
    1: ["Jan"],
    2: ["Feb"],
    3: ["Mar"],
    4: ["Apr"],
    5: ["May"],
    6: ["Jun"],
    7: ["Jul"],
    8: ["Aug"],
    9: ["Sep", "Sept"],
    10: ["Oct"],
    11: ["Nov"],
    12: ["Dec"],
}

YEAR_RE = re.compile(r"^(18|19|20)\d{2}$")

SEASONS = [
    ("Winter", 0),
    ("Spring", 1),
    ("Summer", 2),
    ("Autumn", 3),
]

DAY_TEMPLATES = [
    "The meeting is on {term}.",
    "We met on {term}.",
    "The deadline is {term}.",
    "She called on {term}.",
    "They arrived on {term}.",
    "The appointment is {term}.",
    "It happened on {term}.",
    "The class meets every {term}.",
]
MONTH_TEMPLATES = [
    "The event is in {term}.",
    "She was born in {term}.",
    "The report is due in {term}.",
    "They travel in {term}.",
    "The festival is held in {term}.",
    "We leave in {term}.",
    "The schedule changed in {term}.",
    "It happened in {term}.",
]
YEAR_TEMPLATES = [
    "The event happened in {term}.",
    "She was born in {term}.",
    "He graduated in {term}.",
    "The company was founded in {term}.",
    "The treaty was signed in {term}.",
]

QUAL_PROMPTS = [
    "On Monday, March 3, 1997, the committee met in private session.",
    "On Friday, September 21, 2001, the city was marked by heavy rain.",
    "In the summer of 2012, the team traveled to Europe for training.",
    "By January 2015, the project was considered complete.",
    "On Tuesday evening in April, the event happened without warning.",
]


@dataclass
class Sample:
    concept: str
    term: str
    value: int
    prompt: str
    target_pos: int
    source: str = "prompt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_hook_name(sae: SAE) -> str:
    if hasattr(sae.cfg, "hook_name") and sae.cfg.hook_name:
        return sae.cfg.hook_name
    if hasattr(sae.cfg, "metadata") and sae.cfg.metadata:
        if isinstance(sae.cfg.metadata, dict) and "hook_name" in sae.cfg.metadata:
            return sae.cfg.metadata["hook_name"]
        if hasattr(sae.cfg.metadata, "hook_name"):
            return getattr(sae.cfg.metadata, "hook_name")
    return "blocks.7.hook_resid_pre"


def find_subsequence(sequence: list[int], subseq: list[int]) -> int | None:
    if not subseq or len(subseq) > len(sequence):
        return None
    for idx in range(len(sequence) - len(subseq), -1, -1):
        if sequence[idx : idx + len(subseq)] == subseq:
            return idx
    return None


def extract_snippet(token_ids: list[int], pos: int, tokenizer: AutoTokenizer, window: int = 6) -> str:
    start = max(0, pos - window)
    end = min(len(token_ids), pos + window + 1)
    return tokenizer.decode(token_ids[start:end]).strip()


def normalize_token(token: str) -> str:
    token = token.replace("\u0120", " ")
    token = token.strip().lower()
    token = token.strip(".,;:!?\"'()[]{}")
    return token


def build_token_label_maps() -> tuple[dict[str, int], dict[str, int]]:
    day_map: dict[str, int] = {}
    month_map: dict[str, int] = {}
    for name, idx in DAYS:
        day_map[normalize_token(name)] = idx
    for idx, aliases in DAY_ALIASES.items():
        for alias in aliases:
            day_map[normalize_token(alias)] = idx
    for name, idx in MONTHS:
        month_map[normalize_token(name)] = idx
    for idx, aliases in MONTH_ALIASES.items():
        for alias in aliases:
            month_map[normalize_token(alias)] = idx
    return day_map, month_map


def build_expanded_temporal_terms(include_numbers: bool) -> set[str]:
    terms = {
        "spring",
        "summer",
        "autumn",
        "fall",
        "winter",
        "morning",
        "afternoon",
        "evening",
        "night",
        "midnight",
        "noon",
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "week",
        "weekend",
        "month",
        "year",
        "decade",
        "century",
        "hour",
        "minute",
        "second",
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "annual",
        "annually",
    }
    if include_numbers:
        for i in range(1, 32):
            terms.add(str(i))
        for i in range(1, 32):
            if i % 10 == 1 and i != 11:
                suffix = "st"
            elif i % 10 == 2 and i != 12:
                suffix = "nd"
            elif i % 10 == 3 and i != 13:
                suffix = "rd"
            else:
                suffix = "th"
            terms.add(f"{i}{suffix}")
    return terms


def season_for_month(month_idx: int) -> str:
    if month_idx in {12, 1, 2}:
        return "Winter"
    if month_idx in {3, 4, 5}:
        return "Spring"
    if month_idx in {6, 7, 8}:
        return "Summer"
    return "Autumn"


def build_hierarchy_token_id_map(
    tokenizer: AutoTokenizer,
    year_min: int,
    year_max: int,
    include_aliases: bool,
) -> tuple[dict[int, tuple[str, str, int]], dict[str, str]]:
    mapping: dict[int, tuple[str, str, int]] = {}
    month_to_season: dict[str, str] = {}

    day_name_by_idx = {idx: name for name, idx in DAYS}
    month_name_by_idx = {idx: name for name, idx in MONTHS}

    def add_term(term: str, category: str, label: str, label_idx: int) -> None:
        tok_ids = tokenizer.encode(" " + term, add_special_tokens=False)
        if len(tok_ids) == 1 and tok_ids[0] not in mapping:
            mapping[tok_ids[0]] = (category, label, label_idx)

    for name, idx in DAYS:
        add_term(name, "day_of_week", name, idx)
        add_term(name.lower(), "day_of_week", name, idx)
    if include_aliases:
        for idx, aliases in DAY_ALIASES.items():
            label_name = day_name_by_idx[idx]
            for alias in aliases:
                add_term(alias, "day_of_week", label_name, idx)
                add_term(alias.lower(), "day_of_week", label_name, idx)

    for name, idx in MONTHS:
        add_term(name, "month", name, idx)
        add_term(name.lower(), "month", name, idx)
        month_to_season[name] = season_for_month(idx)
    if include_aliases:
        for idx, aliases in MONTH_ALIASES.items():
            label_name = month_name_by_idx[idx]
            for alias in aliases:
                add_term(alias, "month", label_name, idx)
                add_term(alias.lower(), "month", label_name, idx)
        for name, idx in MONTHS:
            month_to_season[name] = season_for_month(idx)

    for season_name, idx in SEASONS:
        add_term(season_name, "season", season_name, idx)
        add_term(season_name.lower(), "season", season_name, idx)
    add_term("Fall", "season", "Autumn", 3)
    add_term("fall", "season", "Autumn", 3)

    for year in range(year_min, year_max + 1):
        tok_ids = tokenizer.encode(" " + str(year), add_special_tokens=False)
        if len(tok_ids) == 1 and tok_ids[0] not in mapping:
            mapping[tok_ids[0]] = ("year", str(year), year)

    return mapping, month_to_season


def build_samples(
    tokenizer: AutoTokenizer,
    year_min: int,
    year_max: int,
    max_prompts_per_term: int | None,
    years_single_token_only: bool,
) -> list[Sample]:
    samples: list[Sample] = []

    def add_samples(concept: str, items: Iterable[tuple[str, int]], templates: list[str]) -> None:
        for term, value in items:
            term_token_ids = tokenizer.encode(" " + term, add_special_tokens=False)
            if not term_token_ids:
                continue
            prompts = [tmpl.format(term=term) for tmpl in templates]
            if max_prompts_per_term is not None:
                prompts = prompts[:max_prompts_per_term]
            for prompt in prompts:
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                start = find_subsequence(prompt_ids, term_token_ids)
                if start is None:
                    continue
                target_pos = start + len(term_token_ids) - 1
                samples.append(
                    Sample(
                        concept=concept,
                        term=term,
                        value=value,
                        prompt=prompt,
                        target_pos=target_pos,
                        source="prompt",
                    )
                )

    add_samples("days", DAYS, DAY_TEMPLATES)
    add_samples("months", MONTHS, MONTH_TEMPLATES)

    years = []
    for y in range(year_min, year_max + 1):
        token_ids = tokenizer.encode(" " + str(y), add_special_tokens=False)
        if years_single_token_only and len(token_ids) != 1:
            continue
        years.append((str(y), y))
    add_samples("years", years, YEAR_TEMPLATES)
    return samples


def build_token_id_map(
    tokenizer: AutoTokenizer,
    year_min: int,
    year_max: int,
    years_single_token_only: bool,
    include_aliases: bool = True,
) -> tuple[dict[int, tuple[str, int]], dict[int, tuple[str, int]], dict[int, tuple[str, int]]]:
    day_ids: dict[int, tuple[str, int]] = {}
    month_ids: dict[int, tuple[str, int]] = {}
    year_ids: dict[int, tuple[str, int]] = {}
    day_name_by_idx = {idx: name for name, idx in DAYS}
    month_name_by_idx = {idx: name for name, idx in MONTHS}

    def add_single_token_term(
        term: str,
        label_name: str,
        label_idx: int,
        mapping: dict[int, tuple[str, int]],
    ) -> None:
        tok_ids = tokenizer.encode(" " + term, add_special_tokens=False)
        if len(tok_ids) == 1:
            mapping.setdefault(tok_ids[0], (label_name, label_idx))

    for name, idx in DAYS:
        add_single_token_term(name, name, idx, day_ids)
    if include_aliases:
        for idx, aliases in DAY_ALIASES.items():
            label_name = day_name_by_idx[idx]
            for alias in aliases:
                add_single_token_term(alias, label_name, idx, day_ids)
    for name, idx in MONTHS:
        add_single_token_term(name, name, idx, month_ids)
    if include_aliases:
        for idx, aliases in MONTH_ALIASES.items():
            label_name = month_name_by_idx[idx]
            for alias in aliases:
                add_single_token_term(alias, label_name, idx, month_ids)
    for year in range(year_min, year_max + 1):
        tok_ids = tokenizer.encode(" " + str(year), add_special_tokens=False)
        if years_single_token_only and len(tok_ids) != 1:
            continue
        if len(tok_ids) == 1:
            year_ids[tok_ids[0]] = (str(year), year)
    return day_ids, month_ids, year_ids


CORPUS_TEXT_FIELDS = ("text", "content", "document", "article", "body")


def collect_group_vectors_from_corpus(
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    dataset_name: str,
    dataset_split: str,
    max_docs: int,
    max_tokens: int,
    batch_size: int,
    max_samples_per_label: int,
    year_min: int,
    year_max: int,
    years_single_token_only: bool,
    representation: str,
    min_group_norm: float,
    include_aliases: bool,
) -> tuple[np.ndarray, np.ndarray, list[Sample]]:
    day_ids, month_ids, year_ids = build_token_id_map(
        tokenizer,
        year_min,
        year_max,
        years_single_token_only,
        include_aliases=include_aliases,
    )

    if not day_ids and not month_ids and not year_ids:
        raise RuntimeError("No single-token temporal terms found for corpus scan.")

    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    start = group_id * group_rank
    end = start + group_rank
    w_dec_group = sae.W_dec[start:end]

    samples: list[Sample] = []
    vectors: list[np.ndarray] = []
    norms: list[float] = []

    counts = {
        "days": {idx: 0 for _, idx in DAYS},
        "months": {idx: 0 for _, idx in MONTHS},
        "years": {year: 0 for year in range(year_min, year_max + 1)},
    }

    def label_done() -> bool:
        days_done = all(v >= max_samples_per_label for v in counts["days"].values())
        months_done = all(v >= max_samples_per_label for v in counts["months"].values())
        years_done = all(v >= max_samples_per_label for v in counts["years"].values())
        return days_done and months_done and years_done

    stream = load_dataset(dataset_name, split=dataset_split, streaming=True)
    buffer_texts: list[str] = []
    docs_seen = 0

    def process_batch(texts: list[str]) -> None:
        if not texts:
            return
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
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
        if representation == "recon":
            group_vecs = torch.einsum("bsg,gd->bsd", group_acts, w_dec_group)
        else:
            group_vecs = group_acts

        input_ids_cpu = input_ids.detach().cpu().tolist()
        attention_cpu = attention.detach().cpu().tolist()

        for b in range(batch):
            token_ids = input_ids_cpu[b]
            for t in range(seq):
                if attention_cpu[b][t] == 0:
                    continue
                tok_id = token_ids[t]
                if tok_id in day_ids:
                    name, idx = day_ids[tok_id]
                    if counts["days"][idx] >= max_samples_per_label:
                        continue
                    group_norm_val = float(group_norms[b, t].item())
                    if group_norm_val <= min_group_norm:
                        continue
                    vec = group_vecs[b, t].detach().cpu().numpy()
                    samples.append(
                        Sample(
                            concept="days",
                            term=name,
                            value=idx,
                            prompt=extract_snippet(token_ids, t, tokenizer),
                            target_pos=t,
                            source="corpus",
                        )
                    )
                    vectors.append(vec)
                    norms.append(group_norm_val)
                    counts["days"][idx] += 1
                elif tok_id in month_ids:
                    name, idx = month_ids[tok_id]
                    if counts["months"][idx] >= max_samples_per_label:
                        continue
                    group_norm_val = float(group_norms[b, t].item())
                    if group_norm_val <= min_group_norm:
                        continue
                    vec = group_vecs[b, t].detach().cpu().numpy()
                    samples.append(
                        Sample(
                            concept="months",
                            term=name,
                            value=idx,
                            prompt=extract_snippet(token_ids, t, tokenizer),
                            target_pos=t,
                            source="corpus",
                        )
                    )
                    vectors.append(vec)
                    norms.append(group_norm_val)
                    counts["months"][idx] += 1
                elif tok_id in year_ids:
                    name, idx = year_ids[tok_id]
                    if counts["years"][idx] >= max_samples_per_label:
                        continue
                    group_norm_val = float(group_norms[b, t].item())
                    if group_norm_val <= min_group_norm:
                        continue
                    vec = group_vecs[b, t].detach().cpu().numpy()
                    samples.append(
                        Sample(
                            concept="years",
                            term=name,
                            value=idx,
                            prompt=extract_snippet(token_ids, t, tokenizer),
                            target_pos=t,
                            source="corpus",
                        )
                    )
                    vectors.append(vec)
                    norms.append(group_norm_val)
                    counts["years"][idx] += 1

    for sample in stream:
        text = None
        for field in CORPUS_TEXT_FIELDS:
            if field in sample and isinstance(sample[field], str):
                text = sample[field]
                break
        if not text:
            continue
        buffer_texts.append(text)
        docs_seen += 1
        if len(buffer_texts) >= batch_size:
            process_batch(buffer_texts)
            buffer_texts = []
        if label_done() or docs_seen >= max_docs:
            break

    if buffer_texts and not label_done():
        process_batch(buffer_texts)

    if not vectors:
        raise RuntimeError("No corpus vectors collected. Increase corpus docs or max tokens.")

    return np.stack(vectors, axis=0), np.array(norms), samples


def collect_active_group_vectors_from_corpus(
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    dataset_name: str,
    dataset_split: str,
    max_docs: int,
    max_tokens: int,
    batch_size: int,
    max_samples: int,
    year_min: int,
    year_max: int,
    representation: str,
    min_group_norm: float,
) -> tuple[np.ndarray, list[str]]:
    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    start = group_id * group_rank
    end = start + group_rank
    w_dec_group = sae.W_dec[start:end]

    vectors: list[np.ndarray] = []
    tokens: list[str] = []

    stream = load_dataset(dataset_name, split=dataset_split, streaming=True)
    buffer_texts: list[str] = []
    docs_seen = 0

    def process_batch(texts: list[str]) -> None:
        if not texts:
            return
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
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
        if representation == "recon":
            group_vecs = torch.einsum("bsg,gd->bsd", group_acts, w_dec_group)
        else:
            group_vecs = group_acts

        input_ids_cpu = input_ids.detach().cpu().tolist()
        attention_cpu = attention.detach().cpu().tolist()

        for b in range(batch):
            token_ids = input_ids_cpu[b]
            for t in range(seq):
                if attention_cpu[b][t] == 0:
                    continue
                if len(vectors) >= max_samples:
                    return
                group_norm_val = float(group_norms[b, t].item())
                if group_norm_val <= min_group_norm:
                    continue
                tok_id = token_ids[t]
                token_str = tokenizer.decode([int(tok_id)])
                vec = group_vecs[b, t].detach().cpu().numpy()
                vectors.append(vec)
                tokens.append(token_str)

    for sample in stream:
        text = None
        for field in CORPUS_TEXT_FIELDS:
            if field in sample and isinstance(sample[field], str):
                text = sample[field]
                break
        if not text:
            continue
        buffer_texts.append(text)
        docs_seen += 1
        if len(buffer_texts) >= batch_size:
            process_batch(buffer_texts)
            buffer_texts = []
            if len(vectors) >= max_samples:
                break
        if docs_seen >= max_docs:
            break

    if buffer_texts and len(vectors) < max_samples:
        process_batch(buffer_texts)

    if not vectors:
        raise RuntimeError("No active group vectors collected from corpus.")

    return np.stack(vectors, axis=0), tokens


def collect_group_specificity_from_corpus(
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    dataset_name: str,
    dataset_split: str,
    max_docs: int,
    max_tokens: int,
    batch_size: int,
    min_group_norm: float,
    top_k: int,
    quantile: float,
    label_set: str,
) -> dict[str, object]:
    day_map, month_map = build_token_label_maps()
    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    if label_set not in {"base", "expanded", "expanded-no-numbers"}:
        raise ValueError(f"Unknown specificity label set: {label_set}")
    include_numbers = label_set == "expanded"
    expanded_terms = (
        build_expanded_temporal_terms(include_numbers=include_numbers)
        if label_set != "base"
        else set()
    )

    norms: list[float] = []
    labels: list[bool] = []
    temporal_counts = {"day": 0, "month": 0, "year": 0, "other": 0}

    top_temporal: list[tuple[float, str, str]] = []
    top_nontemporal: list[tuple[float, str, str]] = []

    def push_top(heap: list[tuple[float, str, str]], item: tuple[float, str, str]) -> None:
        heap.append(item)
        heap.sort(key=lambda x: x[0])
        if len(heap) > top_k:
            heap.pop(0)

    stream = load_dataset(dataset_name, split=dataset_split, streaming=True)
    buffer_texts: list[str] = []
    docs_seen = 0

    def process_batch(texts: list[str]) -> None:
        if not texts:
            return
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
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

        for b in range(batch):
            token_ids = input_ids_cpu[b]
            for t in range(seq):
                if attention_cpu[b][t] == 0:
                    continue
                norm_val = float(group_norms[b, t].item())
                if norm_val <= min_group_norm:
                    continue
                token_str = tokenizer.decode([int(token_ids[t])])
                token_norm = normalize_token(token_str)
                if not token_norm:
                    continue
                is_day = token_norm in day_map
                is_month = token_norm in month_map
                is_year = bool(YEAR_RE.match(token_norm))
                is_extra = token_norm in expanded_terms
                is_temporal = is_day or is_month or is_year or is_extra
                labels.append(is_temporal)
                norms.append(norm_val)
                if is_day:
                    temporal_counts["day"] += 1
                elif is_month:
                    temporal_counts["month"] += 1
                elif is_year:
                    temporal_counts["year"] += 1
                elif is_extra:
                    temporal_counts["other"] += 1
                snippet = extract_snippet(token_ids, t, tokenizer)
                if is_temporal:
                    push_top(top_temporal, (norm_val, token_str, snippet))
                else:
                    push_top(top_nontemporal, (norm_val, token_str, snippet))

    for sample in stream:
        text = None
        for field in CORPUS_TEXT_FIELDS:
            if field in sample and isinstance(sample[field], str):
                text = sample[field]
                break
        if not text:
            continue
        buffer_texts.append(text)
        docs_seen += 1
        if len(buffer_texts) >= batch_size:
            process_batch(buffer_texts)
            buffer_texts = []
        if docs_seen >= max_docs:
            break

    if buffer_texts:
        process_batch(buffer_texts)

    if not norms:
        raise RuntimeError("No tokens collected for specificity analysis.")

    norms_arr = np.array(norms)
    labels_arr = np.array(labels)
    pos_mask = labels_arr
    neg_mask = ~labels_arr
    pos_mean = float(norms_arr[pos_mask].mean()) if pos_mask.any() else float("nan")
    neg_mean = float(norms_arr[neg_mask].mean()) if neg_mask.any() else float("nan")
    pos_std = float(norms_arr[pos_mask].std()) if pos_mask.any() else float("nan")
    neg_std = float(norms_arr[neg_mask].std()) if neg_mask.any() else float("nan")
    pooled = math.sqrt(0.5 * (pos_std**2 + neg_std**2)) + 1e-8
    effect = (pos_mean - neg_mean) / pooled if np.isfinite(pooled) else float("nan")

    try:
        auc = float(roc_auc_score(labels_arr.astype(int), norms_arr))
    except ValueError:
        auc = float("nan")

    threshold = float(np.quantile(norms_arr, quantile))
    top_mask = norms_arr >= threshold
    precision = float(labels_arr[top_mask].mean()) if top_mask.any() else float("nan")

    top_temporal.sort(key=lambda x: x[0], reverse=True)
    top_nontemporal.sort(key=lambda x: x[0], reverse=True)

    return {
        "n_tokens": int(len(norms_arr)),
        "label_set": label_set,
        "temporal_rate": float(labels_arr.mean()),
        "day_count": int(temporal_counts["day"]),
        "month_count": int(temporal_counts["month"]),
        "year_count": int(temporal_counts["year"]),
        "other_temporal_count": int(temporal_counts["other"]),
        "mean_norm_temporal": pos_mean,
        "mean_norm_non_temporal": neg_mean,
        "std_norm_temporal": pos_std,
        "std_norm_non_temporal": neg_std,
        "effect_size": float(effect),
        "auc": auc,
        "precision_at_99pct": precision,
        "threshold_99pct": threshold,
        "top_temporal": [
            {"norm": float(n), "token": tok, "snippet": snip} for n, tok, snip in top_temporal
        ],
        "top_non_temporal": [
            {"norm": float(n), "token": tok, "snippet": snip} for n, tok, snip in top_nontemporal
        ],
    }


def collect_hierarchy_vectors_from_corpus(
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    dataset_name: str,
    dataset_split: str,
    max_docs: int,
    max_tokens: int,
    batch_size: int,
    min_group_norm: float,
    max_samples_per_label: int,
    year_min: int,
    year_max: int,
    include_aliases: bool,
    representation: str,
) -> dict[str, object]:
    token_map, month_to_season = build_hierarchy_token_id_map(
        tokenizer,
        year_min=year_min,
        year_max=year_max,
        include_aliases=include_aliases,
    )
    if not token_map:
        raise RuntimeError("No single-token temporal terms found for hierarchy analysis.")

    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    start = group_id * group_rank
    end = start + group_rank
    w_dec_group = sae.W_dec[start:end]

    counts: dict[str, int] = {}
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    categories: list[str] = []

    stream = load_dataset(dataset_name, split=dataset_split, streaming=True)
    buffer_texts: list[str] = []
    docs_seen = 0

    def process_batch(texts: list[str]) -> None:
        if not texts:
            return
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
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
        if representation == "recon":
            group_vecs = torch.einsum("bsg,gd->bsd", group_acts, w_dec_group)
        else:
            group_vecs = group_acts

        input_ids_cpu = input_ids.detach().cpu().tolist()
        attention_cpu = attention.detach().cpu().tolist()

        for b in range(batch):
            token_ids = input_ids_cpu[b]
            for t in range(seq):
                if attention_cpu[b][t] == 0:
                    continue
                norm_val = float(group_norms[b, t].item())
                if norm_val <= min_group_norm:
                    continue
                tok_id = token_ids[t]
                if tok_id not in token_map:
                    continue
                category, label, label_idx = token_map[tok_id]
                key = f"{category}:{label}"
                if counts.get(key, 0) >= max_samples_per_label:
                    continue
                vec = group_vecs[b, t].detach().cpu().numpy()
                vectors.append(vec)
                labels.append(label)
                categories.append(category)
                counts[key] = counts.get(key, 0) + 1

    for sample in stream:
        text = None
        for field in CORPUS_TEXT_FIELDS:
            if field in sample and isinstance(sample[field], str):
                text = sample[field]
                break
        if not text:
            continue
        buffer_texts.append(text)
        docs_seen += 1
        if len(buffer_texts) >= batch_size:
            process_batch(buffer_texts)
            buffer_texts = []
        if docs_seen >= max_docs:
            break

    if buffer_texts:
        process_batch(buffer_texts)

    if not vectors:
        raise RuntimeError("No hierarchy vectors collected.")

    return {
        "vectors": np.stack(vectors, axis=0),
        "labels": labels,
        "categories": categories,
        "month_to_season": month_to_season,
    }


def compute_hierarchy_metrics(
    vectors: np.ndarray,
    labels: list[str],
    categories: list[str],
    month_to_season: dict[str, str],
    min_label_samples: int,
    seed: int,
    n_perm: int,
) -> dict[str, object]:
    labels_arr = np.array(labels)
    categories_arr = np.array(categories)
    label_counts = {label: int((labels_arr == label).sum()) for label in np.unique(labels_arr)}
    keep_labels = {label for label, count in label_counts.items() if count >= min_label_samples}
    keep_mask = np.array([label in keep_labels for label in labels_arr])

    vectors = vectors[keep_mask]
    labels_arr = labels_arr[keep_mask]
    categories_arr = categories_arr[keep_mask]

    n_total = vectors.shape[0]
    mean_all = vectors.mean(axis=0)
    total_ss = float(((vectors - mean_all) ** 2).sum())

    between_cat = 0.0
    for category in np.unique(categories_arr):
        mask = categories_arr == category
        n_c = int(mask.sum())
        mean_c = vectors[mask].mean(axis=0)
        between_cat += n_c * float(((mean_c - mean_all) ** 2).sum())

    between_label = 0.0
    for label in np.unique(labels_arr):
        mask = labels_arr == label
        n_l = int(mask.sum())
        mean_l = vectors[mask].mean(axis=0)
        category = categories_arr[mask][0]
        mean_c = vectors[categories_arr == category].mean(axis=0)
        between_label += n_l * float(((mean_l - mean_c) ** 2).sum())

    eta2_cat = between_cat / total_ss if total_ss > 0 else float("nan")
    eta2_label = between_label / total_ss if total_ss > 0 else float("nan")

    label_list = sorted(np.unique(labels_arr))
    label_centroids = {label: vectors[labels_arr == label].mean(axis=0) for label in label_list}
    label_categories = {label: categories_arr[labels_arr == label][0] for label in label_list}

    n_labels = len(label_list)
    dists = []
    hdist = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            li = label_list[i]
            lj = label_list[j]
            dist = float(np.linalg.norm(label_centroids[li] - label_centroids[lj]))
            dists.append(dist)
            if label_categories[li] == label_categories[lj]:
                hdist.append(1.0)
            else:
                hdist.append(2.0)

    rsa_corr = spearman_corr(np.array(dists), np.array(hdist))
    rng = np.random.default_rng(seed)
    perm_corr = []
    categories_list = [label_categories[label] for label in label_list]
    for _ in range(n_perm):
        rng.shuffle(categories_list)
        perm_hdist = []
        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                same = categories_list[i] == categories_list[j]
                perm_hdist.append(1.0 if same else 2.0)
        perm_corr.append(spearman_corr(np.array(dists), np.array(perm_hdist)))
    perm_arr = np.array(perm_corr)
    rsa_p = float((np.sum(perm_arr >= rsa_corr) + 1) / (n_perm + 1))

    month_labels = [label for label in label_list if label_categories[label] == "month"]
    season_labels = [label for label in label_list if label_categories[label] == "season"]
    season_centroids = {}
    for label in season_labels:
        season_centroids[label] = label_centroids[label]
    if not season_centroids:
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            season_months = [m for m in month_labels if month_to_season.get(m) == season]
            if season_months:
                season_centroids[season] = np.stack(
                    [label_centroids[m] for m in season_months], axis=0
                ).mean(axis=0)

    month_season_accuracy = float("nan")
    month_season_p = float("nan")
    if season_centroids and month_labels:
        def cos_sim(a, b):
            return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

        season_names = list(season_centroids.keys())
        correct = 0
        for month in month_labels:
            sims = [cos_sim(label_centroids[month], season_centroids[s]) for s in season_names]
            best = season_names[int(np.argmax(sims))]
            if month_to_season.get(month) == best:
                correct += 1
        month_season_accuracy = correct / max(len(month_labels), 1)

        perm_acc = []
        for _ in range(n_perm):
            shuffled = season_names.copy()
            rng.shuffle(shuffled)
            mapping = {season_names[i]: shuffled[i] for i in range(len(season_names))}
            correct_perm = 0
            for month in month_labels:
                sims = [cos_sim(label_centroids[month], season_centroids[s]) for s in season_names]
                best = season_names[int(np.argmax(sims))]
                if month_to_season.get(month) == mapping.get(best, ""):
                    correct_perm += 1
            perm_acc.append(correct_perm / max(len(month_labels), 1))
        perm_acc_arr = np.array(perm_acc)
        month_season_p = float((np.sum(perm_acc_arr >= month_season_accuracy) + 1) / (n_perm + 1))

    label_counts_filtered = {label: label_counts[label] for label in label_list}

    return {
        "n_samples": int(n_total),
        "n_labels": int(len(label_list)),
        "categories": sorted(set(categories_arr)),
        "label_counts": label_counts_filtered,
        "eta2_category": float(eta2_cat),
        "eta2_label_given_category": float(eta2_label),
        "rsa_spearman": float(rsa_corr),
        "rsa_p_value": float(rsa_p),
        "month_to_season_accuracy": float(month_season_accuracy),
        "month_to_season_p_value": float(month_season_p),
        "n_perm": int(n_perm),
    }


def compute_group_norms_for_prompts(
    prompts: list[str],
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    max_length: int,
) -> tuple[list[list[str]], list[np.ndarray]]:
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
        acts = sae.encode(resid)

    batch, seq, _ = acts.shape
    acts = acts.view(batch, seq, n_groups, group_rank)
    group_acts = acts[:, :, group_id, :]
    group_norms = group_acts.norm(dim=-1).detach().cpu().numpy()

    input_ids_cpu = input_ids.detach().cpu().tolist()
    attention_cpu = attention.detach().cpu().tolist()

    token_lists: list[list[str]] = []
    norm_lists: list[np.ndarray] = []
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


def classify_temporal_token(
    token_norm: str,
    day_map: dict[str, int],
    month_map: dict[str, int],
) -> str | None:
    if token_norm in day_map:
        return "day"
    if token_norm in month_map:
        return "month"
    if token_norm in {"winter", "spring", "summer", "autumn", "fall"}:
        return "season"
    if YEAR_RE.match(token_norm):
        return "year"
    if token_norm.isdigit():
        return "number"
    return None


def plot_temporal_activation_profiles(
    output_dir: Path,
    group_id: int,
    prompts: list[str],
    token_lists: list[list[str]],
    norm_lists: list[np.ndarray],
) -> dict[str, str]:
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

    day_map, month_map = build_token_label_maps()
    colors = {
        "day": "#4C78A8",
        "month": "#F28E2B",
        "season": "#59A14F",
        "year": "#9C755F",
        "number": "#B07AA1",
    }

    fig, axes = plt.subplots(len(prompts), 1, figsize=(13.0, 3.2 * len(prompts)))
    if len(prompts) == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        tokens = token_lists[idx]
        norms = norm_lists[idx]
        ax.plot(range(len(norms)), norms, color="#2f2f2f", linewidth=1.4)
        for pos, tok in enumerate(tokens):
            token_norm = normalize_token(tok)
            category = classify_temporal_token(token_norm, day_map, month_map)
            if category is None:
                continue
            ax.scatter(
                pos,
                norms[pos],
                color=colors[category],
                s=40,
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )
            ax.text(
                pos,
                norms[pos] + 0.25,
                token_norm if len(token_norm) <= 8 else token_norm[:8],
                fontsize=8,
                ha="center",
                va="bottom",
                color=colors[category],
            )
        ax.set_title(prompts[idx])
        ax.set_ylabel("Group norm")
        ax.set_xlabel("Token position")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label.title(),
                   markerfacecolor=color, markersize=6)
        for label, color in colors.items()
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
    )

    fig.suptitle("Temporal Token Activation Profiles", y=1.02, fontsize=14)
    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_token_profiles.png"
    pdf_path = output_dir / "group_1916_temporal_token_profiles.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def plot_year_axis_summary(
    output_dir: Path,
    group_id: int,
    years: np.ndarray,
    axis_values: np.ndarray,
    year_label: str,
) -> dict[str, str]:
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

    unique_years = np.array(sorted(set(years.tolist())))
    means = []
    stds = []
    counts = []
    for y in unique_years:
        vals = axis_values[years == y]
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))
        counts.append(int(vals.shape[0]))

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    counts_arr = np.array(counts)
    sem = stds_arr / np.sqrt(np.maximum(counts_arr, 1))

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.8))
    ax.plot(unique_years, means_arr, color="#2f2f2f", linewidth=1.8)
    ax.fill_between(
        unique_years,
        means_arr - sem,
        means_arr + sem,
        color="#4C78A8",
        alpha=0.25,
        linewidth=0,
    )
    ax.scatter(unique_years, means_arr, s=24, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel(year_label)
    ax.set_title("Year Axis Summary (mean ± SEM)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"TopKSASA Group {group_id}: Year Axis Structure", y=1.02, fontsize=13)
    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_year_axis.png"
    pdf_path = output_dir / "group_1916_temporal_year_axis.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def plot_temporal_category_scatter(
    output_dir: Path,
    group_id: int,
    vectors: np.ndarray,
    labels: list[str],
    categories: list[str],
    month_to_season: dict[str, str],
) -> dict[str, str]:
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

    labels_arr = np.array(labels)
    categories_arr = np.array(categories)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(vectors - vectors.mean(axis=0, keepdims=True))

    fig, ax_all = plt.subplots(1, 1, figsize=(7.2, 5.2))
    ax_month = None

    cat_colors = {
        "day_of_week": "#4C78A8",
        "month": "#F28E2B",
        "season": "#59A14F",
        "year": "#9C755F",
    }
    for cat, color in cat_colors.items():
        mask = categories_arr == cat
        if not mask.any():
            continue
        ax_all.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=16,
            color=color,
            alpha=0.35,
            label=cat.replace("_", " ").title(),
            edgecolors="none",
        )
        mean = coords[mask].mean(axis=0)
        ax_all.scatter(
            [mean[0]],
            [mean[1]],
            s=120,
            color=color,
            edgecolors="white",
            linewidths=0.8,
        )
    ax_all.set_title("Temporal categories")
    ax_all.set_xlabel("PC1")
    ax_all.set_ylabel("PC2")
    ax_all.legend(frameon=False, fontsize=9)
    ax_all.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    ax_all.spines["top"].set_visible(False)
    ax_all.spines["right"].set_visible(False)

    month_mask = categories_arr == "month"
    month_plane = None
    month_order_alignment = float("nan")
    month_ring_cv = float("nan")
    month_order_corr = float("nan")
    season_axis_eta2 = float("nan")
    month_silhouette = float("nan")
    season_silhouette = float("nan")
    if month_mask.any():
        month_labels = labels_arr[month_mask]
        month_vectors = vectors[month_mask]
        month_index = {name: idx for name, idx in MONTHS}
        month_idx = np.array([month_index.get(label, -1) for label in month_labels])
        valid_mask = month_idx > 0
        month_labels = month_labels[valid_mask]
        month_vectors = month_vectors[valid_mask]
        month_idx = month_idx[valid_mask]

        season_labels = np.array([month_to_season.get(label, "Unknown") for label in month_labels])
        mean_month = month_vectors.mean(axis=0, keepdims=True)
        centered_month = month_vectors - mean_month

        coords_month = None
        try:
            unique_months = np.unique(month_idx)
            if unique_months.size >= 3 and month_vectors.shape[0] > unique_months.size:
                lda_month = LinearDiscriminantAnalysis(n_components=2, solver="svd")
                coords_month = lda_month.fit_transform(centered_month, month_idx)
                month_plane = ("LDA-months", "LD1/LD2")
        except Exception:
            coords_month = None

        if coords_month is None:
            try:
                lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
                lda.fit(centered_month, season_labels)
                axis1 = lda.scalings_[:, 0]
                axis1 = axis1 / (np.linalg.norm(axis1) + 1e-8)
                proj1 = centered_month @ axis1
                residual = centered_month - np.outer(proj1, axis1)
                pca_resid = PCA(n_components=min(5, residual.shape[1]), random_state=0)
                proj_resid = pca_resid.fit_transform(residual)
                best_idx, best_corr = best_pc_by_spearman(proj_resid, month_idx.astype(float))
                axis2 = pca_resid.components_[best_idx]
                axis2 = axis2 / (np.linalg.norm(axis2) + 1e-8)
                coords_month = np.stack([proj1, centered_month @ axis2], axis=1)
                month_plane = ("LDA-seasons", f"PC{best_idx + 1}")
                month_order_corr = float(best_corr)

                # eta^2 for season separation along axis1
                total_ss = float(((proj1 - proj1.mean()) ** 2).sum())
                between = 0.0
                for season in np.unique(season_labels):
                    mask = season_labels == season
                    if not mask.any():
                        continue
                    mean_s = proj1[mask].mean()
                    between += mask.sum() * float((mean_s - proj1.mean()) ** 2)
                season_axis_eta2 = between / total_ss if total_ss > 0 else float("nan")
            except Exception:
                coords_month = None

        if coords_month is None:
            pca_month = PCA(n_components=min(5, month_vectors.shape[1]), random_state=0)
            proj_month = pca_month.fit_transform(centered_month)
            labels0 = month_idx - 1
            pc_i, pc_j, _ = best_plane_by_order(proj_month, labels0, 12)
            month_plane = (pc_i + 1, pc_j + 1)
            coords_month = proj_month[:, [pc_i, pc_j]]

        means = []
        for name, idx in MONTHS:
            mask = month_idx == idx
            if not mask.any():
                continue
            means.append((idx, name, coords_month[mask].mean(axis=0)))
        means.sort(key=lambda x: x[0])
        if means:
            mean_coords = np.stack([m[2] for m in means], axis=0)
            mean_labels = np.array([m[0] for m in means])
            month_order_alignment = circular_alignment_score(mean_coords, mean_labels - 1, 12)
            month_ring_cv = ring_radius_cv(mean_coords)

        season_colors = {
            "Winter": "#4C78A8",
            "Spring": "#59A14F",
            "Summer": "#F28E2B",
            "Autumn": "#9C755F",
        }

        if coords_month is not None:
            unique_months = np.unique(month_idx)
            if unique_months.size > 1:
                month_silhouette = float(silhouette_score(coords_month, month_idx))
            unique_seasons = np.unique(season_labels)
            if unique_seasons.size > 1:
                season_silhouette = float(silhouette_score(coords_month, season_labels))

    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_category_scatter.png"
    pdf_path = output_dir / "group_1916_temporal_category_scatter.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "month_plane": month_plane,
        "month_order_alignment": month_order_alignment,
        "month_ring_cv": month_ring_cv,
        "month_order_corr": month_order_corr,
        "season_axis_eta2": season_axis_eta2,
        "month_silhouette": month_silhouette,
        "season_silhouette": season_silhouette,
    }


def plot_season_order_from_months(
    output_dir: Path,
    group_id: int,
    vectors: np.ndarray,
    labels: list[str],
    categories: list[str],
    month_to_season: dict[str, str],
) -> dict[str, str]:
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

    labels_arr = np.array(labels)
    categories_arr = np.array(categories)
    month_mask = categories_arr == "month"
    if not month_mask.any():
        return {}

    month_labels = labels_arr[month_mask]
    month_vectors = vectors[month_mask]
    month_index = {name: idx for name, idx in MONTHS}
    month_idx = np.array([month_index.get(label, -1) for label in month_labels])
    valid_mask = month_idx > 0
    month_labels = month_labels[valid_mask]
    month_vectors = month_vectors[valid_mask]
    month_idx = month_idx[valid_mask]

    month_centroids = []
    month_names = []
    month_ids = []
    for name, idx in MONTHS:
        mask = month_idx == idx
        if not mask.any():
            continue
        month_centroids.append(month_vectors[mask].mean(axis=0))
        month_names.append(name)
        month_ids.append(idx)
    if len(month_centroids) < 4:
        return {}

    month_centroids = np.stack(month_centroids, axis=0)
    month_ids_arr = np.array(month_ids)
    coords, month_order_alignment, month_ring_cv = fit_order_projection(
        month_centroids, month_ids_arr, n_labels=12, label_offset=1
    )

    season_order = [("Winter", 0), ("Spring", 1), ("Summer", 2), ("Autumn", 3)]
    season_coords = []
    season_labels = []
    for season, idx in season_order:
        season_mask = np.array(
            [month_to_season.get(name) == season for name in month_names]
        )
        if not season_mask.any():
            continue
        season_coords.append(coords[season_mask].mean(axis=0))
        season_labels.append(idx)
    if len(season_coords) < 3:
        return {}

    season_coords = np.stack(season_coords, axis=0)
    season_labels_arr = np.array(season_labels)
    season_alignment = circular_alignment_score(season_coords, season_labels_arr, 4)
    season_ring_cv = ring_radius_cv(season_coords)

    season_colors = {
        "Winter": "#4C78A8",
        "Spring": "#59A14F",
        "Summer": "#F28E2B",
        "Autumn": "#9C755F",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2))
    for name, coord in zip(month_names, coords):
        season = month_to_season.get(name)
        color = season_colors.get(season, "#7f7f7f")
        ax.scatter(
            coord[0],
            coord[1],
            s=40,
            color=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.6,
        )

    for (season, _), coord in zip(season_order, season_coords):
        color = season_colors.get(season, "#7f7f7f")
        ax.scatter(
            coord[0],
            coord[1],
            s=180,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )

    for i in range(len(season_coords)):
        j = (i + 1) % len(season_coords)
        ax.plot(
            [season_coords[i, 0], season_coords[j, 0]],
            [season_coords[i, 1], season_coords[j, 1]],
            color="#1f1f1f",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )

    center = season_coords.mean(axis=0)
    max_span = np.max(np.linalg.norm(season_coords - center, axis=1))
    pad = max(0.6, 0.6 * max_span)
    ax.set_xlim(center[0] - max_span - pad, center[0] + max_span + pad)
    ax.set_ylim(center[1] - max_span - pad, center[1] + max_span + pad)
    ax.set_title("Season ordering from month clusters")
    ax.set_xlabel("Order axis 1")
    ax.set_ylabel("Order axis 2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for season, color in season_colors.items():
        ax.scatter([], [], color=color, label=season, s=60, edgecolors="black", linewidths=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_season_order.png"
    pdf_path = output_dir / "group_1916_temporal_season_order.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "season_order_alignment": float(season_alignment),
        "season_ring_cv": float(season_ring_cv),
        "month_order_alignment": float(month_order_alignment),
        "month_ring_cv": float(month_ring_cv),
    }


def collect_group_vectors(
    samples: list[Sample],
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    hook_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    representation: str,
    min_group_norm: float,
) -> tuple[np.ndarray, np.ndarray, list[Sample]]:
    group_vectors: list[np.ndarray] = []
    group_norms: list[float] = []
    kept_samples: list[Sample] = []

    n_groups = int(sae.cfg.n_groups)
    group_rank = int(sae.cfg.group_rank)
    if group_id < 0 or group_id >= n_groups:
        raise ValueError(f"group_id {group_id} out of range (0..{n_groups - 1})")
    start_idx = group_id * group_rank
    end_idx = start_idx + group_rank
    w_dec_group = sae.W_dec[start_idx:end_idx]

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        prompts = [s.prompt for s in batch_samples]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
            resid = cache[hook_name]
            acts = sae.encode(resid)

        batch, seq, _ = acts.shape
        acts = acts.view(batch, seq, n_groups, group_rank)
        group_acts = acts[:, :, group_id, :]
        group_norms_batch = group_acts.norm(dim=-1)
        if representation == "recon":
            group_vecs = torch.einsum("bsg,gd->bsd", group_acts, w_dec_group)
        else:
            group_vecs = group_acts

        for i, sample in enumerate(batch_samples):
            if sample.target_pos >= group_acts.shape[1]:
                continue
            group_norm_val = float(group_norms_batch[i, sample.target_pos].item())
            if group_norm_val <= min_group_norm:
                continue
            vec = group_vecs[i, sample.target_pos].detach().cpu().numpy()
            group_vectors.append(vec)
            group_norms.append(group_norm_val)
            kept_samples.append(sample)

    if not group_vectors:
        raise RuntimeError("No vectors collected. Check prompt construction or tokenization.")

    return np.stack(group_vectors, axis=0), np.array(group_norms), kept_samples


def pca_projection(
    vectors: np.ndarray,
    n_components: int = 5,
    standardize: bool = False,
) -> tuple[PCA, np.ndarray]:
    if standardize:
        mean = vectors.mean(axis=0, keepdims=True)
        std = vectors.std(axis=0, keepdims=True) + 1e-8
        data = (vectors - mean) / std
    else:
        data = vectors - vectors.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=0)
    projected = pca.fit_transform(data)
    return pca, projected


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_rank = a.argsort().argsort()
    b_rank = b.argsort().argsort()
    return pearson_corr(a_rank.astype(float), b_rank.astype(float))


def circular_alignment_score(coords: np.ndarray, labels: np.ndarray, n_labels: int) -> float:
    if coords.size == 0:
        return float("nan")
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    expected = 2.0 * math.pi * (labels.astype(float) / n_labels)
    delta = np.exp(1j * (theta - expected))
    return float(np.abs(delta.mean()))


def best_plane_by_order(
    proj: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    max_components: int = 5,
) -> tuple[int, int, float]:
    best = (-1, -1, -1.0)
    k = min(max_components, proj.shape[1])
    for i in range(k):
        for j in range(i + 1, k):
            score = circular_alignment_score(proj[:, [i, j]], labels, n_labels)
            if score > best[2]:
                best = (i, j, score)
    return best


def ring_radius_cv(coords: np.ndarray) -> float:
    r = np.linalg.norm(coords, axis=1)
    return float(r.std() / (r.mean() + 1e-8))


def pca_plane_metrics(
    proj: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    max_components: int = 5,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    k = min(max_components, proj.shape[1])
    for i in range(k - 1):
        j = i + 1
        coords = proj[:, [i, j]]
        metrics[f"PC{i + 1}-PC{j + 1}"] = {
            "order_alignment": float(circular_alignment_score(coords, labels, n_labels)),
            "ring_radius_cv": ring_radius_cv(coords),
        }
    return metrics


def pc1_radius_correlation(
    proj: np.ndarray,
    plane: tuple[int, int] = (1, 2),
) -> float:
    if proj.shape[1] <= max(plane):
        return float("nan")
    coords = proj[:, [plane[0], plane[1]]]
    radius = np.linalg.norm(coords, axis=1)
    return pearson_corr(proj[:, 0], radius)


def best_pc_by_spearman(proj: np.ndarray, labels: np.ndarray) -> tuple[int, float]:
    best_idx = 0
    best_corr = -1.0
    for i in range(proj.shape[1]):
        corr = abs(spearman_corr(labels, proj[:, i]))
        if corr > best_corr:
            best_corr = corr
            best_idx = i
    return best_idx, best_corr


def fit_order_projection(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    label_offset: int = 0,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    labels0 = labels - label_offset
    angles = 2.0 * math.pi * (labels0.astype(float) / n_labels)
    Y = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X = vectors
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    XtX = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    W = np.linalg.solve(XtX, Xc.T @ Y)
    coords = Xc @ W
    mean_coords = []
    mean_labels = []
    for idx in range(n_labels):
        mask = labels0 == idx
        if not mask.any():
            continue
        mean_coords.append(coords[mask].mean(axis=0))
        mean_labels.append(idx)
    mean_coords = np.stack(mean_coords, axis=0)
    mean_labels = np.array(mean_labels)
    alignment = circular_alignment_score(mean_coords, mean_labels, n_labels)
    ring_cv = ring_radius_cv(mean_coords)
    return coords, alignment, ring_cv


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    if points.shape[0] < 3:
        center = points.mean(axis=0)
        radius = np.linalg.norm(points - center, axis=1).mean()
        return center, float(radius)
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bcoef, c = params
    center = np.array([a / 2.0, bcoef / 2.0])
    radius_sq = c + center.dot(center)
    radius = math.sqrt(max(radius_sq, 1e-8))
    return center, float(radius)


def project_to_circle(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    vec = points - center
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    unit = vec / norms
    return center + radius * unit


def compute_embedding_methods(
    vectors: np.ndarray,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    methods: dict[str, np.ndarray] = {}
    X = vectors - vectors.mean(axis=0, keepdims=True)
    if X.shape[0] < 3:
        return methods
    methods["PCA"] = PCA(n_components=2, random_state=random_state).fit_transform(X)
    n_neighbors = min(10, max(2, X.shape[0] - 1))
    try:
        methods["Isomap"] = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X)
    except Exception:
        pass
    try:
        methods["Spectral"] = SpectralEmbedding(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state,
        ).fit_transform(X)
    except Exception:
        pass
    try:
        methods["MDS"] = MDS(
            n_components=2,
            random_state=random_state,
            n_init=4,
            max_iter=300,
        ).fit_transform(X)
    except Exception:
        pass
    return methods


def select_best_embedding(
    methods: dict[str, np.ndarray],
    labels: np.ndarray,
    n_labels: int,
) -> tuple[str, np.ndarray, dict[str, float]]:
    best_name = ""
    best_coords = None
    best_score = -1.0
    best_ring = float("nan")
    for name, coords in methods.items():
        score = circular_alignment_score(coords, labels, n_labels)
        ring = ring_radius_cv(coords)
        if score > best_score:
            best_score = score
            best_ring = ring
            best_name = name
            best_coords = coords
    if best_coords is None:
        raise RuntimeError("No embedding method succeeded for plotting.")
    return best_name, best_coords, {"order_alignment": float(best_score), "ring_cv": float(best_ring)}


def build_circular_targets(
    labels: np.ndarray,
    n_labels: int,
    label_offset: int,
) -> np.ndarray:
    labels0 = labels.astype(float) - label_offset
    angles = 2.0 * math.pi * (labels0 / n_labels)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def circular_regression_cv(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    label_offset: int,
    n_splits: int,
    ridge: float,
    standardize: bool,
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float, list[float]]:
    if splits is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = [(train, test) for train, test in kf.split(vectors)]
    scores = []
    for train_idx, test_idx in splits:
        X_train = vectors[train_idx]
        X_test = vectors[test_idx]
        if standardize:
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
        Y_train = build_circular_targets(labels[train_idx], n_labels, label_offset)
        Y_test = build_circular_targets(labels[test_idx], n_labels, label_offset)
        model = Ridge(alpha=ridge, fit_intercept=True)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        score = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
        scores.append(float(score))
    return float(np.mean(scores)), float(np.std(scores)), scores


def permutation_test_circular_regression(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    label_offset: int,
    n_splits: int,
    ridge: float,
    standardize: bool,
    seed: int,
    n_perm: int,
) -> dict[str, object]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = [(train, test) for train, test in kf.split(vectors)]
    obs_mean, obs_std, obs_scores = circular_regression_cv(
        vectors=vectors,
        labels=labels,
        n_labels=n_labels,
        label_offset=label_offset,
        n_splits=n_splits,
        ridge=ridge,
        standardize=standardize,
        seed=seed,
        splits=splits,
    )
    rng = np.random.default_rng(seed)
    perm_means = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_mean, _, _ = circular_regression_cv(
            vectors=vectors,
            labels=perm_labels,
            n_labels=n_labels,
            label_offset=label_offset,
            n_splits=n_splits,
            ridge=ridge,
            standardize=standardize,
            seed=seed,
            splits=splits,
        )
        perm_means.append(perm_mean)
    perm_means_arr = np.array(perm_means)
    p_value = float((np.sum(perm_means_arr >= obs_mean) + 1) / (n_perm + 1))
    return {
        "observed_mean_r2": float(obs_mean),
        "observed_std_r2": float(obs_std),
        "observed_fold_r2": obs_scores,
        "null_mean_r2": float(perm_means_arr.mean()),
        "null_std_r2": float(perm_means_arr.std()),
        "null_r2_samples": perm_means,
        "p_value": p_value,
        "n_perm": int(n_perm),
        "n_splits": int(n_splits),
    }


def subsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def compute_vr_persistence_h1(
    points: np.ndarray,
    max_eps_quantile: float,
    max_triangles: int,
    seed: int,
) -> dict[str, object]:
    n_points = points.shape[0]
    if n_points < 6:
        raise RuntimeError("Need at least 6 points for persistent homology.")

    dist = pairwise_distances(points, metric="euclidean")
    iu = np.triu_indices(n_points, k=1)
    dists = dist[iu]
    max_eps = float(np.quantile(dists, max_eps_quantile))

    adj = [set() for _ in range(n_points)]
    edges: list[tuple[int, int, float]] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = dist[i, j]
            if d <= max_eps:
                edges.append((i, j, float(d)))
                adj[i].add(j)
                adj[j].add(i)

    triangles: list[tuple[int, int, int, float]] = []
    tri_count = 0
    for i in range(n_points):
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i].intersection(adj[j])
            for k in common:
                if k <= j:
                    continue
                filt = max(dist[i, j], dist[i, k], dist[j, k])
                if filt <= max_eps:
                    triangles.append((i, j, k, float(filt)))
                    tri_count += 1
                    if tri_count > max_triangles:
                        raise RuntimeError(
                            "Too many triangles for VR complex. Reduce ph_max_points or ph_max_eps_quantile."
                        )

    simplices: list[dict[str, object]] = []
    for i in range(n_points):
        simplices.append(
            {"key": (i,), "dim": 0, "filtration": 0.0, "verts": (i,)}
        )
    for i, j, d in edges:
        simplices.append(
            {"key": (i, j), "dim": 1, "filtration": d, "verts": (i, j)}
        )
    for i, j, k, f in triangles:
        simplices.append(
            {"key": (i, j, k), "dim": 2, "filtration": f, "verts": (i, j, k)}
        )

    simplices.sort(key=lambda s: (s["filtration"], s["dim"]))
    index_by_key = {s["key"]: idx for idx, s in enumerate(simplices)}

    boundaries: dict[int, list[int]] = {}
    for idx, simplex in enumerate(simplices):
        dim = simplex["dim"]
        verts = simplex["verts"]
        if dim == 1:
            boundaries[idx] = [index_by_key[(verts[0],)], index_by_key[(verts[1],)]]
        elif dim == 2:
            i, j, k = verts
            boundaries[idx] = [
                index_by_key[tuple(sorted((i, j)))],
                index_by_key[tuple(sorted((i, k)))],
                index_by_key[tuple(sorted((j, k)))],
            ]

    low: dict[int, int] = {}
    columns: dict[int, set[int]] = {}
    birth_edges: dict[int, float] = {}
    pairs: list[tuple[int, int, int]] = []

    for idx, simplex in enumerate(simplices):
        dim = simplex["dim"]
        if dim == 0:
            continue
        col = set(boundaries.get(idx, []))
        while col:
            pivot = max(col)
            if pivot in low:
                col ^= columns[low[pivot]]
            else:
                break
        if not col:
            if dim == 1:
                birth_edges[idx] = float(simplex["filtration"])
        else:
            pivot = max(col)
            low[pivot] = idx
            pairs.append((pivot, idx, dim - 1))
        columns[idx] = col

    h1_intervals: list[tuple[float, float]] = []
    paired_births: set[int] = set()
    for birth_idx, death_idx, dim in pairs:
        if dim == 1:
            birth = float(simplices[birth_idx]["filtration"])
            death = float(simplices[death_idx]["filtration"])
            h1_intervals.append((birth, death))
            paired_births.add(birth_idx)

    for birth_idx, birth in birth_edges.items():
        if birth_idx not in paired_births:
            h1_intervals.append((birth, max_eps))

    persistences = [death - birth for birth, death in h1_intervals]
    max_pers = float(np.max(persistences)) if persistences else 0.0
    mean_pers = float(np.mean(persistences)) if persistences else 0.0
    total_pers = float(np.sum(persistences)) if persistences else 0.0

    return {
        "n_points": int(n_points),
        "n_edges": int(len(edges)),
        "n_triangles": int(len(triangles)),
        "max_eps": float(max_eps),
        "h1_intervals": h1_intervals,
        "max_persistence": max_pers,
        "mean_persistence": mean_pers,
        "total_persistence": total_pers,
    }


def persistent_homology_test(
    vectors: np.ndarray,
    seed: int,
    max_points: int,
    pca_dim: int,
    max_eps_quantile: float,
    max_triangles: int,
    null_samples: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sample = subsample_points(vectors, max_points=max_points, seed=seed)
    pca = PCA(n_components=min(pca_dim, sample.shape[1]), random_state=seed)
    points = pca.fit_transform(sample - sample.mean(axis=0, keepdims=True))

    obs = compute_vr_persistence_h1(
        points=points,
        max_eps_quantile=max_eps_quantile,
        max_triangles=max_triangles,
        seed=seed,
    )

    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    jitter = 1e-6 * np.eye(cov.shape[0])
    try:
        chol = np.linalg.cholesky(cov + jitter)
    except np.linalg.LinAlgError:
        jitter = 1e-4 * np.eye(cov.shape[0])
        chol = np.linalg.cholesky(cov + jitter)

    null_max = []
    for _ in range(null_samples):
        z = rng.normal(size=points.shape)
        null_points = z @ chol.T + mean
        null_stats = compute_vr_persistence_h1(
            points=null_points,
            max_eps_quantile=max_eps_quantile,
            max_triangles=max_triangles,
            seed=seed,
        )
        null_max.append(null_stats["max_persistence"])
    null_arr = np.array(null_max)
    p_value = float((np.sum(null_arr >= obs["max_persistence"]) + 1) / (null_samples + 1))

    return {
        "observed": obs,
        "null_max_persistence": null_max,
        "null_mean_max_persistence": float(null_arr.mean()),
        "null_std_max_persistence": float(null_arr.std()),
        "p_value": p_value,
        "pca_dim": int(pca_dim),
        "max_points": int(points.shape[0]),
        "max_eps_quantile": float(max_eps_quantile),
        "null_samples": int(null_samples),
    }


def plot_ring_with_circle(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    labels_info: list[tuple[str, int]],
    title: str,
    annotate_metrics: tuple[float, float] | None = None,
    show_raw: bool = False,
    unit_ring: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    n_labels = len(labels_info)
    cmap = plt.get_cmap("twilight")

    means = []
    for name, idx in labels_info:
        mask = labels == idx
        if not mask.any():
            continue
        mean = coords[mask].mean(axis=0)
        means.append((idx, mean, name))
    means.sort(key=lambda x: x[0])
    if means:
        ring_raw = np.stack([m[1] for m in means], axis=0)
        ring_center, ring_radius = fit_circle(ring_raw)
        ring_proj = project_to_circle(ring_raw, ring_center, ring_radius)
        if unit_ring:
            angles = np.arctan2(
                ring_raw[:, 1] - ring_center[1],
                ring_raw[:, 0] - ring_center[0],
            )
            ring_proj = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            ring_center = np.zeros(2)
            ring_radius = 1.0
        ring_colors = cmap((np.array([m[0] for m in means]).astype(float) / n_labels) % 1.0)

        if show_raw:
            ax.scatter(
                ring_raw[:, 0],
                ring_raw[:, 1],
                s=36,
                facecolors="none",
                edgecolors="#9a9a9a",
                linewidths=0.9,
                alpha=0.9,
                zorder=1,
            )
            for raw, proj in zip(ring_raw, ring_proj):
                ax.plot(
                    [raw[0], proj[0]],
                    [raw[1], proj[1]],
                    color="#b0b0b0",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=1,
                )

        ax.plot(
            ring_proj[:, 0],
            ring_proj[:, 1],
            color="#1f1f1f",
            linewidth=2.6,
            alpha=0.95,
        )
        ax.plot(
            [ring_proj[-1, 0], ring_proj[0, 0]],
            [ring_proj[-1, 1], ring_proj[0, 1]],
            color="#1f1f1f",
            linewidth=2.6,
            alpha=0.95,
        )
        ax.scatter(
            ring_proj[:, 0],
            ring_proj[:, 1],
            s=160,
            c=ring_colors,
            edgecolors="#1f1f1f",
            linewidths=0.9,
            zorder=3,
        )
        circle = plt.Circle(
            ring_center,
            ring_radius,
            color="#1f1f1f",
            linewidth=1.2,
            fill=False,
            alpha=0.3,
            linestyle="--",
        )
        ax.add_artist(circle)

        for k, (_, _, name) in enumerate(means):
            offset = ring_proj[k] - ring_center
            norm = np.linalg.norm(offset)
            if norm > 1e-6:
                offset = offset / norm * (0.25 * ring_radius)
            label_pos = ring_proj[k] + offset
            ax.text(
                label_pos[0],
                label_pos[1],
                name[:3],
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                zorder=4,
            )
            next_idx = (k + 1) % len(ring_proj)
            ax.annotate(
                "",
                xy=ring_proj[next_idx],
                xytext=ring_proj[k],
                arrowprops=dict(arrowstyle="->", color="#3a3a3a", lw=1.1, alpha=0.75),
                zorder=2,
            )

        margin = 1.25 if unit_ring else max(1.2 * ring_radius, 0.5)
        ax.set_xlim(ring_center[0] - margin, ring_center[0] + margin)
        ax.set_ylim(ring_center[1] - margin, ring_center[1] + margin)

    if annotate_metrics is not None:
        order_align, ring_cv = annotate_metrics
        ax.text(
            0.02,
            0.98,
            f"order={order_align:.2f}\nring_cv={ring_cv:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.12, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def get_group_top_tokens(
    sae: SAE,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    group_id: int,
    top_k: int = 20,
) -> list[dict]:
    group_rank = int(sae.cfg.group_rank)
    start = group_id * group_rank
    end = start + group_rank
    w_dec_group = sae.W_dec[start:end]

    direction = w_dec_group.mean(dim=0)
    direction = direction / (direction.norm() + 1e-8)

    if hasattr(model, "W_U"):
        w_u = model.W_U
    elif hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        w_u = model.unembed.W_U
    else:
        raise AttributeError("Model does not expose W_U for unembedding.")

    if w_u.shape[0] == direction.shape[0]:
        token_scores = direction @ w_u
    elif w_u.shape[1] == direction.shape[0]:
        token_scores = direction @ w_u.T
    else:
        raise ValueError("Unembedding matrix shape mismatch.")

    top_vals, top_idx = token_scores.topk(min(top_k, token_scores.shape[0]))
    out: list[dict] = []
    for val, idx in zip(top_vals.detach().cpu().tolist(), top_idx.detach().cpu().tolist()):
        out.append(
            {
                "token_id": int(idx),
                "token": tokenizer.decode([int(idx)]),
                "score": float(val),
            }
        )
    return out


def summarize_top_prompts(
    samples: list[Sample],
    norms: np.ndarray,
    concept: str,
    top_k: int = 6,
) -> list[dict]:
    idx = [i for i, s in enumerate(samples) if s.concept == concept]
    if not idx:
        return []
    subset = [(i, norms[i]) for i in idx]
    subset.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for i, score in subset[:top_k]:
        s = samples[i]
        out.append(
            {
                "prompt": s.prompt,
                "term": s.term,
                "value": s.value,
                "group_norm": float(score),
            }
        )
    return out


def plot_temporal_panels(
    output_dir: Path,
    group_id: int,
    days_order_coords: np.ndarray,
    days_labels: np.ndarray,
    months_order_coords: np.ndarray,
    months_labels: np.ndarray,
    years_vals: np.ndarray,
    years_axis: np.ndarray,
    years_pc: int,
    year_corr: float,
    days_order_alignment: float,
    days_ring_cv: float,
    months_order_alignment: float,
    months_ring_cv: float,
) -> dict[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 5.4),
        gridspec_kw={"width_ratios": [1.1, 1.1, 1.2]},
    )

    plot_ring_with_circle(
        axes[0],
        days_order_coords,
        days_labels,
        DAYS,
        "Days (order-aligned projection)",
        annotate_metrics=(days_order_alignment, days_ring_cv),
        unit_ring=True,
    )
    plot_ring_with_circle(
        axes[1],
        months_order_coords,
        months_labels,
        MONTHS,
        "Months (order-aligned projection)",
        annotate_metrics=(months_order_alignment, months_ring_cv),
        unit_ring=True,
    )

    ax = axes[2]
    cmap = plt.get_cmap("viridis")
    colors = cmap((years_vals - years_vals.min()) / (years_vals.max() - years_vals.min() + 1e-8))
    ax.scatter(years_vals, years_axis, s=34, alpha=0.85, c=colors, edgecolors="none")

    coeffs = np.polyfit(years_vals, years_axis, deg=1)
    x_line = np.linspace(years_vals.min(), years_vals.max(), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, color="#2f2f2f", linewidth=1.4, alpha=0.85)

    ax.set_title(f"Years vs PC{years_pc + 1} (|ρ|={year_corr:.2f})")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"PC{years_pc + 1}")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"TopKSASA Group {group_id}: Temporal Structure", y=1.02, fontsize=14)
    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_paper.png"
    pdf_path = output_dir / "group_1916_temporal_paper.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def plot_temporal_3d(
    output_dir: Path,
    days_proj: np.ndarray,
    days_labels: np.ndarray,
    months_proj: np.ndarray,
    months_labels: np.ndarray,
) -> dict[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(12.8, 5.4))
    ax_days = fig.add_subplot(1, 2, 1, projection="3d")
    ax_months = fig.add_subplot(1, 2, 2, projection="3d")

    def plot_means_3d(ax, proj, labels, labels_info, title):
        means = []
        for name, idx in labels_info:
            mask = labels == idx
            if not mask.any():
                continue
            mean = proj[mask][:, :3].mean(axis=0)
            means.append((idx, mean, name))
        means.sort(key=lambda x: x[0])
        if not means:
            return
        coords = np.stack([m[1] for m in means], axis=0)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#1f1f1f", lw=2.0, alpha=0.85)
        ax.plot(
            [coords[-1, 0], coords[0, 0]],
            [coords[-1, 1], coords[0, 1]],
            [coords[-1, 2], coords[0, 2]],
            color="#1f1f1f",
            lw=2.0,
            alpha=0.85,
        )
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=70, c="#4c78a8", edgecolors="none")
        for (_, mean, name) in means:
            ax.text(mean[0], mean[1], mean[2], name[:3], fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    plot_means_3d(ax_days, days_proj, days_labels, DAYS, "Days (PCA 3D means)")
    plot_means_3d(ax_months, months_proj, months_labels, MONTHS, "Months (PCA 3D means)")

    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_paper_3d.png"
    pdf_path = output_dir / "group_1916_temporal_paper_3d.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def plot_embedding_comparison(
    output_dir: Path,
    group_id: int,
    plot_concept_data: dict[str, dict[str, object]],
    analysis: dict[str, object],
) -> dict[str, str]:
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
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.0))

    for row, concept in enumerate(["days", "months"]):
        data = plot_concept_data[concept]
        labels = data["labels"]
        labels_info = DAYS if concept == "days" else MONTHS

        best_coords = data["best_embed"]
        best_name = data["best_embed_name"]
        best_metrics = data["best_embed_metrics"]
        plot_ring_with_circle(
            axes[row, 0],
            best_coords,
            labels,
            labels_info,
            f"{concept.title()} best ({best_name})",
            annotate_metrics=(best_metrics["order_alignment"], best_metrics["ring_cv"]),
            show_raw=True,
        )

        plot_ring_with_circle(
            axes[row, 1],
            data["order_coords"],
            labels,
            labels_info,
            f"{concept.title()} order projection",
            annotate_metrics=(
                analysis["order_projection"][concept]["order_alignment"],
                analysis["order_projection"][concept]["ring_radius_cv"],
            ),
            show_raw=True,
        )

        plot_ring_with_circle(
            axes[row, 2],
            data["pca_best_coords"],
            labels,
            labels_info,
            f"{concept.title()} PCA best",
            annotate_metrics=(
                data["pca_best_metrics"]["order_alignment"],
                data["pca_best_metrics"]["ring_cv"],
            ),
            show_raw=True,
        )

    fig.suptitle(f"TopKSASA Group {group_id}: Embedding Comparisons", y=1.02, fontsize=14)
    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_paper_embeddings.png"
    pdf_path = output_dir / "group_1916_temporal_paper_embeddings.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def parse_axes_pair(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid axes specification: {text}")
    idx = [int(p) - 1 for p in parts]
    if any(i < 0 for i in idx):
        raise ValueError(f"Axes must be 1-indexed positive ints: {text}")
    return idx[0], idx[1]


def plot_mdf_style_scatter(
    output_dir: Path,
    group_id: int,
    vectors: np.ndarray,
    tokens: list[str],
    year_min: int,
    year_max: int,
    days_axes: tuple[int, int],
    months_axes: tuple[int, int],
    years_axes: tuple[int, int],
) -> dict[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    day_map, month_map = build_token_label_maps()
    norm_tokens = [normalize_token(t) for t in tokens]

    pca = PCA(n_components=min(5, vectors.shape[1]), random_state=0)
    proj = pca.fit_transform(vectors - vectors.mean(axis=0, keepdims=True))
    max_axis = max(days_axes + months_axes + years_axes)
    if max_axis >= proj.shape[1]:
        raise ValueError("Requested MDF PCA axes exceed available components.")

    fig = plt.figure(figsize=(15.4, 4.8))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Days of week
    day_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = []
    for token in norm_tokens:
        if token in day_map:
            colors.append(day_colors[day_map[token]])
        else:
            colors.append("#BBB")
    ax1.scatter(
        proj[:, days_axes[0]],
        proj[:, days_axes[1]],
        s=1,
        c=colors,
        alpha=0.6,
        linewidths=0,
    )
    ax1.set_xlabel(f"PCA axis {days_axes[0] + 1}", fontsize=8, labelpad=-2)
    ax1.set_ylabel(f"PCA axis {days_axes[1] + 1}", fontsize=8, labelpad=-1)
    ax1.set_title("Days of the Week", fontsize=9)
    style_axis(ax1)

    legend_elements_1 = [
        Line2D([0], [0], marker="o", color="w", label="Monday", markerfacecolor=day_colors[0], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Tuesday", markerfacecolor=day_colors[1], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Wednesday", markerfacecolor=day_colors[2], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Thursday", markerfacecolor=day_colors[3], markersize=3),
    ]
    legend_elements_2 = [
        Line2D([0], [0], marker="o", color="w", label="Friday", markerfacecolor=day_colors[4], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Saturday", markerfacecolor=day_colors[5], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Sunday", markerfacecolor=day_colors[6], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Other", markerfacecolor="#BBB", markersize=3),
    ]
    legend1 = ax1.legend(
        handles=legend_elements_1,
        loc="upper left",
        fontsize=4,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    legend2 = ax1.legend(
        handles=legend_elements_2,
        loc="upper right",
        fontsize=4,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    ax1.add_artist(legend1)

    # Months of year
    month_colors = plt.cm.rainbow(np.linspace(0, 1 - 1 / 12, 12))
    colors = []
    for token in norm_tokens:
        if token in month_map:
            colors.append(month_colors[month_map[token] - 1])
        else:
            colors.append("#BBB")
    ax2.scatter(
        proj[:, months_axes[0]],
        proj[:, months_axes[1]],
        s=1,
        c=colors,
        alpha=0.6,
        linewidths=0,
    )
    ax2.set_xlabel(f"PCA axis {months_axes[0] + 1}", fontsize=8, labelpad=-2)
    ax2.set_ylabel(f"PCA axis {months_axes[1] + 1}", fontsize=8, labelpad=-2)
    ax2.set_title("Months of the Year", fontsize=9)
    style_axis(ax2)

    legend_elements_1 = [
        Line2D([0], [0], marker="o", color="w", label="January", markerfacecolor=month_colors[0], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="February", markerfacecolor=month_colors[1], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="March", markerfacecolor=month_colors[2], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="April", markerfacecolor=month_colors[3], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="May", markerfacecolor=month_colors[4], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="June", markerfacecolor=month_colors[5], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="July", markerfacecolor=month_colors[6], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="August", markerfacecolor=month_colors[7], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="September", markerfacecolor=month_colors[8], markersize=3),
    ]
    legend_elements_2 = [
        Line2D([0], [0], marker="o", color="w", label="October", markerfacecolor=month_colors[9], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="November", markerfacecolor=month_colors[10], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="December", markerfacecolor=month_colors[11], markersize=3),
        Line2D([0], [0], marker="o", color="w", label="Other", markerfacecolor="#BBB", markersize=3),
    ]
    legend1 = ax2.legend(
        handles=legend_elements_1,
        loc="upper left",
        fontsize=4,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    legend2 = ax2.legend(
        handles=legend_elements_2,
        loc="upper right",
        fontsize=4,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    ax2.add_artist(legend1)

    # Years of 20th century
    colors = []
    year_values = []
    for token in norm_tokens:
        if YEAR_RE.match(token):
            year = int(token)
            if year_min <= year <= year_max:
                year_values.append(year)
                colors.append(year)
                continue
        year_values.append(None)
        colors.append(None)
    if any(v is not None for v in colors):
        color_vals = np.array([v if v is not None else year_min for v in colors], dtype=float)
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=year_min, vmax=year_max)
        plot_colors = [
            cmap(norm(val)) if year_values[i] is not None else "#BBB"
            for i, val in enumerate(color_vals)
        ]
    else:
        plot_colors = ["#BBB"] * len(colors)
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=year_min, vmax=year_max)

    ax3.scatter(
        proj[:, years_axes[0]],
        proj[:, years_axes[1]],
        s=1,
        c=plot_colors,
        alpha=0.6,
        linewidths=0,
    )
    ax3.set_xlabel(f"PCA axis {years_axes[0] + 1}", fontsize=7, labelpad=-2)
    ax3.set_ylabel(f"PCA axis {years_axes[1] + 1}", fontsize=7, labelpad=-2)
    ax3.set_title("Years of the 20th Century", fontsize=9)
    style_axis(ax3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.865, 0.82, 0.11, 0.02])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([year_min, (year_min + year_max) // 2, year_max])
    cbar.set_ticklabels([str(year_min), str((year_min + year_max) // 2), str(year_max)])

    fig.suptitle(f"TopKSASA Group {group_id}: PCA Scatter (MDF-style)", y=1.02, fontsize=12)
    fig.tight_layout(pad=0.7)

    png_path = output_dir / "group_1916_temporal_pca_mdf_style.png"
    pdf_path = output_dir / "group_1916_temporal_pca_mdf_style.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def plot_pca_scatter_panels(
    output_dir: Path,
    group_id: int,
    concept_data: dict[str, dict[str, object]],
    analysis: dict[str, object],
) -> dict[str, str]:
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

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.6, 5.0),
        gridspec_kw={"width_ratios": [1.1, 1.1, 1.2]},
    )

    def plot_cyclic_scatter(
        ax,
        coords: np.ndarray,
        labels: np.ndarray,
        labels_info: list[tuple[str, int]],
        label_offset: int,
        plane: tuple[int, int],
        title: str,
        metrics: tuple[float, float] | None,
    ) -> None:
        n_labels = len(labels_info)
        labels0 = labels - label_offset
        cmap = plt.get_cmap("twilight")
        colors = cmap((labels0.astype(float) / n_labels) % 1.0)
        ax.scatter(coords[:, 0], coords[:, 1], s=16, c=colors, alpha=0.35, linewidths=0)

        means = []
        for name, idx in labels_info:
            mask = labels == idx
            if not mask.any():
                continue
            mean = coords[mask].mean(axis=0)
            means.append((idx, mean, name))
        means.sort(key=lambda x: x[0])
        if means:
            mean_coords = np.stack([m[1] for m in means], axis=0)
            mean_colors = cmap((np.array([m[0] for m in means]).astype(float) - label_offset) / n_labels)
            ax.plot(
                mean_coords[:, 0],
                mean_coords[:, 1],
                color="#1f1f1f",
                linewidth=2.0,
                alpha=0.8,
            )
            ax.plot(
                [mean_coords[-1, 0], mean_coords[0, 0]],
                [mean_coords[-1, 1], mean_coords[0, 1]],
                color="#1f1f1f",
                linewidth=2.0,
                alpha=0.8,
            )
            ax.scatter(
                mean_coords[:, 0],
                mean_coords[:, 1],
                s=120,
                c=mean_colors,
                edgecolors="#1f1f1f",
                linewidths=0.8,
                zorder=3,
            )
            for (_, mean, name) in means:
                ax.text(
                    mean[0],
                    mean[1],
                    name[:3],
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    zorder=4,
                )

        if metrics is not None:
            order_align, ring_cv = metrics
            ax.text(
                0.02,
                0.98,
                f"order={order_align:.2f}\nring_cv={ring_cv:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

        ax.set_title(title)
        ax.set_xlabel(f"PC{plane[0] + 1}")
        ax.set_ylabel(f"PC{plane[1] + 1}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def plot_years_scatter(
        ax,
        coords: np.ndarray,
        years: np.ndarray,
        plane: tuple[int, int],
        title: str,
    ) -> None:
        cmap = plt.get_cmap("viridis")
        norm = (years - years.min()) / (years.max() - years.min() + 1e-8)
        colors = cmap(norm)
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=16, c=colors, alpha=0.55, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(f"PC{plane[0] + 1}")
        ax.set_ylabel(f"PC{plane[1] + 1}")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Year")

    days = concept_data["days"]
    months = concept_data["months"]
    years = concept_data["years"]

    plot_cyclic_scatter(
        axes[0],
        coords=days["proj"][:, list(days["plane"])],
        labels=days["labels"],
        labels_info=DAYS,
        label_offset=0,
        plane=days["plane"],
        title="Days of the Week",
        metrics=(
            analysis["pca"]["days"]["order_alignment"],
            analysis["pca"]["days"]["ring_radius_cv"],
        ),
    )
    plot_cyclic_scatter(
        axes[1],
        coords=months["proj"][:, list(months["plane"])],
        labels=months["labels"],
        labels_info=MONTHS,
        label_offset=1,
        plane=months["plane"],
        title="Months of the Year",
        metrics=(
            analysis["pca"]["months"]["order_alignment"],
            analysis["pca"]["months"]["ring_radius_cv"],
        ),
    )
    plot_years_scatter(
        axes[2],
        coords=years["proj"][:, list(years["plane"])],
        years=years["labels"],
        plane=years["plane"],
        title="Years (PCA plane)",
    )

    fig.suptitle(f"TopKSASA Group {group_id}: PCA Scatter (MDF-style)", y=1.02, fontsize=14)
    fig.tight_layout()

    png_path = output_dir / "group_1916_temporal_pca_scatter.png"
    pdf_path = output_dir / "group_1916_temporal_pca_scatter.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-style temporal analysis for TopKSASA groups.")
    parser.add_argument(
        "--sae-dir",
        type=str,
        required=True,
        help="Path to TopKSASA SAE directory (cfg.json + sae_weights.safetensors)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/group_1916_temporal_paper",
        help="Output directory for plots and reports",
    )
    parser.add_argument("--group-id", type=int, default=1916)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--year-min", type=int, default=1980)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--max-prompts-per-term", type=int, default=5)
    parser.add_argument("--years-single-token-only", action="store_true", default=True)
    parser.add_argument("--pca-standardize", action="store_true", default=False)
    parser.add_argument(
        "--representation",
        type=str,
        choices=["latent", "recon"],
        default="latent",
        help="Analyze group activations ('latent') or decoder reconstructions ('recon').",
    )
    parser.add_argument(
        "--min-group-norm",
        type=float,
        default=0.0,
        help="Drop samples with group activation norm <= this threshold.",
    )
    parser.add_argument(
        "--plot-source",
        type=str,
        choices=["all", "prompt", "corpus"],
        default="all",
        help="Source to use for ring plots (days/months).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prompt", "corpus", "both"],
        default="corpus",
        help="Use controlled prompts, corpus scan, or both.",
    )
    parser.add_argument("--corpus-dataset", type=str, default="openwebtext")
    parser.add_argument("--corpus-split", type=str, default="train")
    parser.add_argument("--corpus-docs", type=int, default=2500)
    parser.add_argument("--corpus-max-tokens", type=int, default=128)
    parser.add_argument("--corpus-max-samples-per-label", type=int, default=200)
    parser.add_argument(
        "--corpus-include-aliases",
        dest="corpus_include_aliases",
        action="store_true",
        help="Include common abbreviations when scanning the corpus.",
    )
    parser.add_argument(
        "--corpus-no-aliases",
        dest="corpus_include_aliases",
        action="store_false",
        help="Disable abbreviation aliases in corpus scan.",
    )
    parser.set_defaults(corpus_include_aliases=True)
    parser.add_argument(
        "--mdf-style",
        action="store_true",
        default=False,
        help="Generate MDF-style PCA scatter matching MultiDimensionalFeatures.",
    )
    parser.add_argument("--mdf-max-samples", type=int, default=20000)
    parser.add_argument("--mdf-axes-days", type=str, default="2,3")
    parser.add_argument("--mdf-axes-months", type=str, default="2,3")
    parser.add_argument("--mdf-axes-years", type=str, default="3,4")
    parser.add_argument(
        "--circular-regression",
        action="store_true",
        default=False,
        help="Run permutation-tested circular regression on days/months.",
    )
    parser.add_argument("--circular-permutations", type=int, default=200)
    parser.add_argument("--circular-cv-splits", type=int, default=5)
    parser.add_argument("--circular-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--circular-no-standardize",
        dest="circular_standardize",
        action="store_false",
        help="Disable feature standardization for circular regression.",
    )
    parser.set_defaults(circular_standardize=True)
    parser.add_argument(
        "--ph-test",
        action="store_true",
        default=False,
        help="Run persistent homology test (Vietoris-Rips) on days/months.",
    )
    parser.add_argument("--ph-max-points", type=int, default=250)
    parser.add_argument("--ph-pca-dim", type=int, default=5)
    parser.add_argument("--ph-max-eps-quantile", type=float, default=0.2)
    parser.add_argument("--ph-null-samples", type=int, default=50)
    parser.add_argument("--ph-max-triangles", type=int, default=2000000)
    parser.add_argument(
        "--specificity",
        action="store_true",
        default=False,
        help="Measure temporal specificity using corpus tokens.",
    )
    parser.add_argument("--specificity-top-k", type=int, default=8)
    parser.add_argument("--specificity-quantile", type=float, default=0.99)
    parser.add_argument(
        "--specificity-label-set",
        type=str,
        choices=["base", "expanded", "expanded-no-numbers"],
        default="base",
        help="Temporal label set for specificity checks.",
    )
    parser.add_argument(
        "--hierarchy",
        action="store_true",
        default=False,
        help="Analyze temporal hierarchy between day/month/season/year.",
    )
    parser.add_argument("--hierarchy-max-samples-per-label", type=int, default=80)
    parser.add_argument("--hierarchy-min-label-samples", type=int, default=20)
    parser.add_argument("--hierarchy-permutations", type=int, default=200)
    parser.add_argument(
        "--qualitative",
        action="store_true",
        default=False,
        help="Generate qualitative temporal structure figures.",
    )
    parser.add_argument("--qualitative-max-samples-per-label", type=int, default=40)
    parser.add_argument("--qualitative-prompts", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)

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

    model = HookedTransformer.from_pretrained(model_name, device=device, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hook_name = get_hook_name(sae)

    all_vectors: list[np.ndarray] = []
    all_norms: list[np.ndarray] = []
    all_samples: list[Sample] = []

    if args.mode in {"prompt", "both"}:
        samples = build_samples(
            tokenizer=tokenizer,
            year_min=args.year_min,
            year_max=args.year_max,
            max_prompts_per_term=args.max_prompts_per_term,
            years_single_token_only=args.years_single_token_only,
        )

        vectors, norms, kept_samples = collect_group_vectors(
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
            min_group_norm=args.min_group_norm,
        )
        all_vectors.append(vectors)
        all_norms.append(norms)
        all_samples.extend(kept_samples)

    if args.mode in {"corpus", "both"}:
        vectors, norms, kept_samples = collect_group_vectors_from_corpus(
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            group_id=args.group_id,
            hook_name=hook_name,
            device=device,
            dataset_name=args.corpus_dataset,
            dataset_split=args.corpus_split,
            max_docs=args.corpus_docs,
            max_tokens=args.corpus_max_tokens,
            batch_size=args.batch_size,
            max_samples_per_label=args.corpus_max_samples_per_label,
            year_min=args.year_min,
            year_max=args.year_max,
            years_single_token_only=args.years_single_token_only,
            representation=args.representation,
            min_group_norm=args.min_group_norm,
            include_aliases=args.corpus_include_aliases,
        )
        all_vectors.append(vectors)
        all_norms.append(norms)
        all_samples.extend(kept_samples)

    if not all_vectors:
        raise RuntimeError("No vectors collected. Check mode and inputs.")

    vectors = np.concatenate(all_vectors, axis=0)
    norms = np.concatenate(all_norms, axis=0)
    kept_samples = all_samples
    vector_norms = np.linalg.norm(vectors, axis=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis: dict[str, object] = {
        "meta": {
            "sae_dir": args.sae_dir,
            "group_id": args.group_id,
            "n_groups": int(sae.cfg.n_groups),
            "group_rank": int(sae.cfg.group_rank),
            "hook_name": hook_name,
            "model_name": model_name,
            "seed": args.seed,
            "year_min": args.year_min,
            "year_max": args.year_max,
            "years_single_token_only": args.years_single_token_only,
            "representation": args.representation,
            "min_group_norm": args.min_group_norm,
            "plot_source": args.plot_source,
            "mode": args.mode,
            "corpus_dataset": args.corpus_dataset,
            "corpus_split": args.corpus_split,
            "corpus_docs": args.corpus_docs,
            "corpus_max_tokens": args.corpus_max_tokens,
            "corpus_max_samples_per_label": args.corpus_max_samples_per_label,
            "corpus_include_aliases": args.corpus_include_aliases,
            "mdf_style": args.mdf_style,
            "mdf_max_samples": args.mdf_max_samples,
            "mdf_axes_days": args.mdf_axes_days,
            "mdf_axes_months": args.mdf_axes_months,
            "mdf_axes_years": args.mdf_axes_years,
            "circular_regression": args.circular_regression,
            "circular_permutations": args.circular_permutations,
            "circular_cv_splits": args.circular_cv_splits,
            "circular_ridge": args.circular_ridge,
            "circular_standardize": args.circular_standardize,
            "ph_test": args.ph_test,
            "ph_max_points": args.ph_max_points,
            "ph_pca_dim": args.ph_pca_dim,
            "ph_max_eps_quantile": args.ph_max_eps_quantile,
            "ph_null_samples": args.ph_null_samples,
            "ph_max_triangles": args.ph_max_triangles,
            "specificity": args.specificity,
            "specificity_top_k": args.specificity_top_k,
            "specificity_quantile": args.specificity_quantile,
            "specificity_label_set": args.specificity_label_set,
            "hierarchy": args.hierarchy,
            "hierarchy_max_samples_per_label": args.hierarchy_max_samples_per_label,
            "hierarchy_min_label_samples": args.hierarchy_min_label_samples,
            "hierarchy_permutations": args.hierarchy_permutations,
            "qualitative": args.qualitative,
            "qualitative_max_samples_per_label": args.qualitative_max_samples_per_label,
            "qualitative_prompts": args.qualitative_prompts,
        },
        "counts": {},
        "pca": {},
        "order_projection": {},
    }

    concept_masks = {
        "days": np.array([s.concept == "days" for s in kept_samples]),
        "months": np.array([s.concept == "months" for s in kept_samples]),
        "years": np.array([s.concept == "years" for s in kept_samples]),
    }

    concept_data = {}

    for concept, mask in concept_masks.items():
        subset = vectors[mask]
        subset_norms = norms[mask]
        subset_vector_norms = vector_norms[mask]
        labels = np.array([s.value for s in np.array(kept_samples)[mask]])

        analysis["counts"][concept] = {
            "n_samples": int(subset.shape[0]),
            "mean_group_norm": float(subset_norms.mean()),
            "std_group_norm": float(subset_norms.std()),
            "mean_vector_norm": float(subset_vector_norms.mean()),
            "std_vector_norm": float(subset_vector_norms.std()),
        }

        if subset.shape[0] < 5:
            continue

        pca, proj = pca_projection(
            subset,
            n_components=min(5, subset.shape[1]),
            standardize=args.pca_standardize,
        )
        analysis["pca"][concept] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "standardized": bool(args.pca_standardize),
        }

        if concept in {"days", "months"}:
            n_labels = 7 if concept == "days" else 12
            label_offset = 0 if concept == "days" else 1
            labels0 = labels - label_offset
            idx0, idx1, order_score = best_plane_by_order(
                proj, labels0, n_labels
            )
            coords = proj[:, [idx0, idx1]]
            concept_data[concept] = {
                "vectors": subset,
                "coords": coords,
                "labels": labels,
                "plane": (idx0, idx1),
                "order_alignment": float(order_score),
                "ring_cv": ring_radius_cv(coords),
                "proj": proj,
            }
            analysis["pca"][concept].update(
                {
                    "best_plane": [idx0 + 1, idx1 + 1],
                    "order_alignment": float(order_score),
                    "ring_radius_cv": ring_radius_cv(coords),
                }
            )
            analysis["pca"][concept]["planes"] = pca_plane_metrics(
                proj, labels0, n_labels
            )
            analysis["pca"][concept]["pc1_radius_corr_pc2_3"] = pc1_radius_correlation(
                proj, plane=(1, 2)
            )
            order_coords, order_align, order_ring_cv = fit_order_projection(
                subset, labels, n_labels, label_offset=label_offset
            )
            concept_data[concept]["order_coords"] = order_coords
            analysis["order_projection"][concept] = {
                "order_alignment": float(order_align),
                "ring_radius_cv": float(order_ring_cv),
            }
        else:
            pc_idx, pc_corr = best_pc_by_spearman(proj, labels.astype(float))
            if proj.shape[1] < 2:
                raise RuntimeError("Need at least 2 PCA components to plot years.")
            plane_j = 0 if pc_idx != 0 else 1
            if plane_j >= proj.shape[1]:
                plane_j = 1 if pc_idx != 1 else 0
            concept_data[concept] = {
                "vectors": subset,
                "proj": proj,
                "labels": labels,
                "year_pc": pc_idx,
                "year_corr": float(pc_corr),
                "plane": (pc_idx, plane_j),
            }
            analysis["pca"][concept].update(
                {
                    "best_pc": int(pc_idx + 1),
                    "best_pc_spearman": float(pc_corr),
                    "pc1_spearman": float(abs(spearman_corr(labels.astype(float), proj[:, 0]))),
                    "pca_plane": [int(pc_idx + 1), int(plane_j + 1)],
                }
            )

    analysis["top_tokens"] = get_group_top_tokens(
        sae, model, tokenizer, args.group_id, top_k=20
    )
    analysis["top_prompts"] = {
        "days": summarize_top_prompts(kept_samples, norms, "days"),
        "months": summarize_top_prompts(kept_samples, norms, "months"),
        "years": summarize_top_prompts(kept_samples, norms, "years"),
    }

    plot_source = args.plot_source
    if plot_source != "all":
        plot_mask = np.array([s.source == plot_source for s in kept_samples])
        if not plot_mask.any():
            plot_source = "all"
            plot_mask = np.ones(len(kept_samples), dtype=bool)
    else:
        plot_mask = np.ones(len(kept_samples), dtype=bool)

    plot_samples = [s for s, keep in zip(kept_samples, plot_mask) if keep]
    plot_vectors = vectors[plot_mask]

    plot_concept_data = {}
    for concept in ["days", "months"]:
        mask = np.array([s.concept == concept for s in plot_samples])
        if not mask.any():
            mask = np.array([s.concept == concept for s in kept_samples])
            plot_vectors = vectors
            plot_samples = kept_samples
        subset = plot_vectors[mask]
        labels = np.array([s.value for s in np.array(plot_samples)[mask]])
        if subset.shape[0] < 5:
            raise RuntimeError(f"Not enough {concept} samples for plotting.")
        _, proj = pca_projection(subset, n_components=min(5, subset.shape[1]), standardize=args.pca_standardize)
        n_labels = 7 if concept == "days" else 12
        label_offset = 0 if concept == "days" else 1
        labels0 = labels - label_offset
        order_coords, _, _ = fit_order_projection(
            subset, labels, n_labels, label_offset=label_offset
        )
        methods = compute_embedding_methods(subset, random_state=args.seed)
        best_name, best_coords, best_metrics = select_best_embedding(methods, labels0, n_labels)
        pca_i, pca_j, _ = best_plane_by_order(proj, labels0, n_labels)
        pca_best_coords = proj[:, [pca_i, pca_j]]
        pca_best_metrics = {
            "order_alignment": circular_alignment_score(pca_best_coords, labels0, n_labels),
            "ring_cv": ring_radius_cv(pca_best_coords),
        }
        plot_concept_data[concept] = {
            "order_coords": order_coords,
            "labels": labels,
            "proj": proj,
            "best_embed": best_coords,
            "best_embed_name": best_name,
            "best_embed_metrics": best_metrics,
            "pca_best_coords": pca_best_coords,
            "pca_best_metrics": pca_best_metrics,
            "pca_best_plane": (pca_i, pca_j),
        }

    analysis["embeddings"] = {}
    for concept in ["days", "months"]:
        analysis["embeddings"][concept] = {
            "best_method": plot_concept_data[concept]["best_embed_name"],
            "best_order_alignment": plot_concept_data[concept]["best_embed_metrics"]["order_alignment"],
            "best_ring_cv": plot_concept_data[concept]["best_embed_metrics"]["ring_cv"],
            "pca_best_plane": [int(plot_concept_data[concept]["pca_best_plane"][0] + 1), int(plot_concept_data[concept]["pca_best_plane"][1] + 1)],
            "pca_best_order_alignment": plot_concept_data[concept]["pca_best_metrics"]["order_alignment"],
            "pca_best_ring_cv": plot_concept_data[concept]["pca_best_metrics"]["ring_cv"],
        }

    if args.circular_regression:
        analysis["circular_regression"] = {}
        for concept in ["days", "months"]:
            if concept not in concept_data:
                continue
            vectors_circ = concept_data[concept]["vectors"]
            labels_circ = concept_data[concept]["labels"]
            n_labels = 7 if concept == "days" else 12
            label_offset = 0 if concept == "days" else 1
            if vectors_circ.shape[0] < args.circular_cv_splits:
                continue
            analysis["circular_regression"][concept] = permutation_test_circular_regression(
                vectors=vectors_circ,
                labels=labels_circ,
                n_labels=n_labels,
                label_offset=label_offset,
                n_splits=args.circular_cv_splits,
                ridge=args.circular_ridge,
                standardize=args.circular_standardize,
                seed=args.seed,
                n_perm=args.circular_permutations,
            )

    if args.ph_test:
        analysis["persistent_homology"] = {}
        for concept in ["days", "months"]:
            if concept not in concept_data:
                continue
            vectors_ph = concept_data[concept]["vectors"]
            if vectors_ph.shape[0] < 10:
                continue
            analysis["persistent_homology"][concept] = persistent_homology_test(
                vectors=vectors_ph,
                seed=args.seed,
                max_points=args.ph_max_points,
                pca_dim=args.ph_pca_dim,
                max_eps_quantile=args.ph_max_eps_quantile,
                max_triangles=args.ph_max_triangles,
                null_samples=args.ph_null_samples,
            )

    if args.specificity:
        analysis["specificity"] = collect_group_specificity_from_corpus(
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            group_id=args.group_id,
            hook_name=hook_name,
            device=device,
            dataset_name=args.corpus_dataset,
            dataset_split=args.corpus_split,
            max_docs=args.corpus_docs,
            max_tokens=args.corpus_max_tokens,
            batch_size=args.batch_size,
            min_group_norm=args.min_group_norm,
            top_k=args.specificity_top_k,
            quantile=args.specificity_quantile,
            label_set=args.specificity_label_set,
        )

    hierarchy_data = None
    if args.hierarchy or args.qualitative:
        max_samples_per_label = max(
            args.hierarchy_max_samples_per_label,
            args.qualitative_max_samples_per_label,
        )
        hierarchy_data = collect_hierarchy_vectors_from_corpus(
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            group_id=args.group_id,
            hook_name=hook_name,
            device=device,
            dataset_name=args.corpus_dataset,
            dataset_split=args.corpus_split,
            max_docs=args.corpus_docs,
            max_tokens=args.corpus_max_tokens,
            batch_size=args.batch_size,
            min_group_norm=args.min_group_norm,
            max_samples_per_label=max_samples_per_label,
            year_min=args.year_min,
            year_max=args.year_max,
            include_aliases=args.corpus_include_aliases,
            representation=args.representation,
        )
    if args.hierarchy and hierarchy_data is not None:
        analysis["hierarchy"] = compute_hierarchy_metrics(
            vectors=hierarchy_data["vectors"],
            labels=hierarchy_data["labels"],
            categories=hierarchy_data["categories"],
            month_to_season=hierarchy_data["month_to_season"],
            min_label_samples=args.hierarchy_min_label_samples,
            seed=args.seed,
            n_perm=args.hierarchy_permutations,
        )

    plot_paths = plot_temporal_panels(
        output_dir=output_dir,
        group_id=args.group_id,
        days_order_coords=plot_concept_data["days"]["order_coords"],
        days_labels=plot_concept_data["days"]["labels"],
        months_order_coords=plot_concept_data["months"]["order_coords"],
        months_labels=plot_concept_data["months"]["labels"],
        years_vals=concept_data["years"]["labels"],
        years_axis=concept_data["years"]["proj"][:, concept_data["years"]["year_pc"]],
        years_pc=concept_data["years"]["year_pc"],
        year_corr=concept_data["years"]["year_corr"],
        days_order_alignment=analysis["order_projection"]["days"]["order_alignment"],
        days_ring_cv=analysis["order_projection"]["days"]["ring_radius_cv"],
        months_order_alignment=analysis["order_projection"]["months"]["order_alignment"],
        months_ring_cv=analysis["order_projection"]["months"]["ring_radius_cv"],
    )
    plot_3d_paths = plot_temporal_3d(
        output_dir=output_dir,
        days_proj=plot_concept_data["days"]["proj"],
        days_labels=plot_concept_data["days"]["labels"],
        months_proj=plot_concept_data["months"]["proj"],
        months_labels=plot_concept_data["months"]["labels"],
    )
    plot_embed_paths = plot_embedding_comparison(
        output_dir=output_dir,
        group_id=args.group_id,
        plot_concept_data=plot_concept_data,
        analysis=analysis,
    )
    plot_scatter_paths = plot_pca_scatter_panels(
        output_dir=output_dir,
        group_id=args.group_id,
        concept_data=concept_data,
        analysis=analysis,
    )
    analysis["plots"] = {
        **plot_paths,
        "figure3d_png": plot_3d_paths["figure_png"],
        "figure3d_pdf": plot_3d_paths["figure_pdf"],
        "figure_embed_png": plot_embed_paths["figure_png"],
        "figure_embed_pdf": plot_embed_paths["figure_pdf"],
        "figure_scatter_png": plot_scatter_paths["figure_png"],
        "figure_scatter_pdf": plot_scatter_paths["figure_pdf"],
    }

    if args.mdf_style:
        days_axes = parse_axes_pair(args.mdf_axes_days)
        months_axes = parse_axes_pair(args.mdf_axes_months)
        years_axes = parse_axes_pair(args.mdf_axes_years)
        mdf_vectors, mdf_tokens = collect_active_group_vectors_from_corpus(
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            group_id=args.group_id,
            hook_name=hook_name,
            device=device,
            dataset_name=args.corpus_dataset,
            dataset_split=args.corpus_split,
            max_docs=args.corpus_docs,
            max_tokens=args.corpus_max_tokens,
            batch_size=args.batch_size,
            max_samples=args.mdf_max_samples,
            year_min=args.year_min,
            year_max=args.year_max,
            representation=args.representation,
            min_group_norm=args.min_group_norm,
        )
        mdf_plot_paths = plot_mdf_style_scatter(
            output_dir=output_dir,
            group_id=args.group_id,
            vectors=mdf_vectors,
            tokens=mdf_tokens,
            year_min=args.year_min,
            year_max=args.year_max,
            days_axes=days_axes,
            months_axes=months_axes,
            years_axes=years_axes,
        )
        analysis["plots"]["figure_mdf_png"] = mdf_plot_paths["figure_png"]
        analysis["plots"]["figure_mdf_pdf"] = mdf_plot_paths["figure_pdf"]

    if args.qualitative:
        prompts = QUAL_PROMPTS[: max(1, args.qualitative_prompts)]
        token_lists, norm_lists = compute_group_norms_for_prompts(
            prompts=prompts,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            group_id=args.group_id,
            hook_name=hook_name,
            device=device,
            max_length=args.max_length,
        )
        profile_paths = plot_temporal_activation_profiles(
            output_dir=output_dir,
            group_id=args.group_id,
            prompts=prompts,
            token_lists=token_lists,
            norm_lists=norm_lists,
        )
        analysis["plots"]["figure_token_profile_png"] = profile_paths["figure_png"]
        analysis["plots"]["figure_token_profile_pdf"] = profile_paths["figure_pdf"]

        year_axis_vals = concept_data["years"]["proj"][:, concept_data["years"]["year_pc"]]
        year_label = f"PC{concept_data['years']['year_pc'] + 1}"
        year_axis_paths = plot_year_axis_summary(
            output_dir=output_dir,
            group_id=args.group_id,
            years=concept_data["years"]["labels"],
            axis_values=year_axis_vals,
            year_label=year_label,
        )
        analysis["plots"]["figure_year_axis_png"] = year_axis_paths["figure_png"]
        analysis["plots"]["figure_year_axis_pdf"] = year_axis_paths["figure_pdf"]

        if hierarchy_data is not None:
            category_paths = plot_temporal_category_scatter(
                output_dir=output_dir,
                group_id=args.group_id,
                vectors=hierarchy_data["vectors"],
                labels=hierarchy_data["labels"],
                categories=hierarchy_data["categories"],
                month_to_season=hierarchy_data["month_to_season"],
            )
            analysis["plots"]["figure_category_png"] = category_paths["figure_png"]
            analysis["plots"]["figure_category_pdf"] = category_paths["figure_pdf"]
            if "hierarchy" in analysis:
                analysis["hierarchy"]["month_plane"] = category_paths["month_plane"]
                analysis["hierarchy"]["month_order_alignment"] = category_paths["month_order_alignment"]
                analysis["hierarchy"]["month_ring_cv"] = category_paths["month_ring_cv"]
                analysis["hierarchy"]["month_order_corr"] = category_paths["month_order_corr"]
                analysis["hierarchy"]["season_axis_eta2"] = category_paths["season_axis_eta2"]
                analysis["hierarchy"]["month_silhouette"] = category_paths["month_silhouette"]
                analysis["hierarchy"]["season_silhouette"] = category_paths["season_silhouette"]

            season_order_paths = plot_season_order_from_months(
                output_dir=output_dir,
                group_id=args.group_id,
                vectors=hierarchy_data["vectors"],
                labels=hierarchy_data["labels"],
                categories=hierarchy_data["categories"],
                month_to_season=hierarchy_data["month_to_season"],
            )
            if season_order_paths:
                analysis["plots"]["figure_season_order_png"] = season_order_paths["figure_png"]
                analysis["plots"]["figure_season_order_pdf"] = season_order_paths["figure_pdf"]
                if "hierarchy" in analysis:
                    analysis["hierarchy"]["season_order_alignment"] = season_order_paths[
                        "season_order_alignment"
                    ]
                    analysis["hierarchy"]["season_ring_cv"] = season_order_paths["season_ring_cv"]

    with (output_dir / "group_1916_temporal_paper_summary.json").open("w") as f:
        json.dump(analysis, f, indent=2)

    report_lines = []
    report_lines.append("# Group 1916 Temporal Analysis (PCA-Style)")
    report_lines.append("")
    report_lines.append(f"SAE: {args.sae_dir}")
    report_lines.append(f"Model: {model_name} | Hook: {hook_name}")
    report_lines.append(f"Group: {args.group_id} | Group rank: {sae.cfg.group_rank}")
    report_lines.append(
        f"Representation: {args.representation} | min_group_norm: {args.min_group_norm}"
    )
    report_lines.append(
        f"Corpus aliases: {analysis['meta']['corpus_include_aliases']}"
    )
    report_lines.append(f"Plot source: {analysis['meta']['plot_source']}")
    report_lines.append("")
    report_lines.append("## Coverage")
    for concept, stats in analysis["counts"].items():
        report_lines.append(
            f"- {concept}: n={stats['n_samples']}, mean_group_norm={stats['mean_group_norm']:.4f}, std_group_norm={stats['std_group_norm']:.4f}, "
            f"mean_vec_norm={stats['mean_vector_norm']:.4f}, std_vec_norm={stats['std_vector_norm']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## PCA diagnostics")
    for concept, pdata in analysis["pca"].items():
        evr = ", ".join(f"{v:.3f}" for v in pdata["explained_variance_ratio"][:5])
        if concept in {"days", "months"}:
            report_lines.append(
                f"- {concept}: EVR=[{evr}], best_plane=PC{pdata['best_plane'][0]} vs PC{pdata['best_plane'][1]}, order_alignment={pdata['order_alignment']:.3f}, ring_cv={pdata['ring_radius_cv']:.3f}"
            )
        else:
            report_lines.append(
                f"- {concept}: EVR=[{evr}], best_pc=PC{pdata['best_pc']}, spearman(|year, pc|)={pdata['best_pc_spearman']:.3f}"
            )
    report_lines.append("")
    report_lines.append("## PCA scatter planes (MDF-style)")
    report_lines.append(
        f"- days: plane=PC{analysis['pca']['days']['best_plane'][0]}-PC{analysis['pca']['days']['best_plane'][1]}"
    )
    report_lines.append(
        f"- months: plane=PC{analysis['pca']['months']['best_plane'][0]}-PC{analysis['pca']['months']['best_plane'][1]}"
    )
    report_lines.append(
        f"- years: plane=PC{analysis['pca']['years']['pca_plane'][0]}-PC{analysis['pca']['years']['pca_plane'][1]}"
    )
    if args.mdf_style:
        report_lines.append("")
        report_lines.append("## MDF-style scatter (MultiDimensionalFeatures replication)")
        report_lines.append(
            f"- axes: days=PC{args.mdf_axes_days.replace(',', '-').strip()}, "
            f"months=PC{args.mdf_axes_months.replace(',', '-').strip()}, "
            f"years=PC{args.mdf_axes_years.replace(',', '-').strip()}"
        )
        report_lines.append(
            f"- samples: {args.mdf_max_samples} | representation={args.representation} | min_group_norm={args.min_group_norm}"
        )
    report_lines.append("")
    report_lines.append("## PCA plane diagnostics (sequential pairs)")
    for concept in ["days", "months"]:
        planes = analysis["pca"][concept].get("planes", {})
        if not planes:
            continue
        report_lines.append(f"- {concept}:")
        for plane_name, metrics in planes.items():
            report_lines.append(
                f"  - {plane_name}: order_alignment={metrics['order_alignment']:.3f}, ring_cv={metrics['ring_radius_cv']:.3f}"
            )
    report_lines.append("")
    report_lines.append("## Embedding comparisons (plot source)")
    for concept in ["days", "months"]:
        if concept not in analysis.get("embeddings", {}):
            continue
        edata = analysis["embeddings"][concept]
        report_lines.append(
            f"- {concept}: best={edata['best_method']} (order={edata['best_order_alignment']:.3f}, ring_cv={edata['best_ring_cv']:.3f}); "
            f"PCA-best=PC{edata['pca_best_plane'][0]}-PC{edata['pca_best_plane'][1]} (order={edata['pca_best_order_alignment']:.3f}, ring_cv={edata['pca_best_ring_cv']:.3f})"
        )
    report_lines.append("")
    report_lines.append("## Cone check (PC1 vs radius in PC2–PC3)")
    for concept in ["days", "months"]:
        corr = analysis["pca"][concept].get("pc1_radius_corr_pc2_3", float("nan"))
        report_lines.append(f"- {concept}: pc1_radius_corr={corr:.3f}")
    report_lines.append("")
    report_lines.append("## Order-aligned projection (fit to circular labels; metrics on label means)")
    for concept in ["days", "months"]:
        if concept not in analysis["order_projection"]:
            continue
        pdata = analysis["order_projection"][concept]
        report_lines.append(
            f"- {concept}: order_alignment={pdata['order_alignment']:.3f}, ring_cv={pdata['ring_radius_cv']:.3f}"
        )
    report_lines.append("")
    if "circular_regression" in analysis:
        report_lines.append("## Circular regression (permutation-tested)")
        for concept in ["days", "months"]:
            if concept not in analysis["circular_regression"]:
                continue
            cres = analysis["circular_regression"][concept]
            report_lines.append(
                f"- {concept}: mean_r2={cres['observed_mean_r2']:.3f} ± {cres['observed_std_r2']:.3f}, "
                f"null_mean={cres['null_mean_r2']:.3f}, p={cres['p_value']:.4f} (n_perm={cres['n_perm']})"
            )
        report_lines.append("")
    if "persistent_homology" in analysis:
        report_lines.append("## Persistent homology (Vietoris-Rips)")
        for concept in ["days", "months"]:
            if concept not in analysis["persistent_homology"]:
                continue
            pres = analysis["persistent_homology"][concept]
            obs = pres["observed"]
            report_lines.append(
                f"- {concept}: max_persistence={obs['max_persistence']:.3f}, "
                f"null_mean={pres['null_mean_max_persistence']:.3f}, p={pres['p_value']:.4f} "
                f"(n={obs['n_points']}, eps_q={pres['max_eps_quantile']:.2f})"
            )
        report_lines.append("")
    if "specificity" in analysis:
        spec = analysis["specificity"]
        report_lines.append("## Temporal specificity (corpus tokens)")
        report_lines.append(
            f"- temporal_rate={spec['temporal_rate']:.4f}, auc={spec['auc']:.3f}, "
            f"precision@q={spec['precision_at_99pct']:.3f} (q={analysis['meta']['specificity_quantile']:.2f}), "
            f"effect_size={spec['effect_size']:.3f}"
        )
        report_lines.append(
            f"- mean_norm_temporal={spec['mean_norm_temporal']:.3f} ± {spec['std_norm_temporal']:.3f} | "
            f"mean_norm_non_temporal={spec['mean_norm_non_temporal']:.3f} ± {spec['std_norm_non_temporal']:.3f}"
        )
        report_lines.append(
            f"- label_set={spec['label_set']} | temporal_counts: days={spec['day_count']}, months={spec['month_count']}, "
            f"years={spec['year_count']}, other={spec['other_temporal_count']}"
        )
        report_lines.append("")
        report_lines.append("### Top temporal activations")
        for item in spec["top_temporal"]:
            report_lines.append(
                f"- norm={item['norm']:.4f} | token={item['token']} | {item['snippet']}"
            )
        report_lines.append("")
        report_lines.append("### Top non-temporal activations")
        for item in spec["top_non_temporal"]:
            report_lines.append(
                f"- norm={item['norm']:.4f} | token={item['token']} | {item['snippet']}"
            )
        report_lines.append("")
    if "hierarchy" in analysis:
        hier = analysis["hierarchy"]
        report_lines.append("## Temporal hierarchy (day/month/season/year)")
        report_lines.append(
            f"- eta2_category={hier['eta2_category']:.3f}, eta2_label_given_category={hier['eta2_label_given_category']:.3f}"
        )
        report_lines.append(
            f"- rsa_spearman={hier['rsa_spearman']:.3f}, rsa_p={hier['rsa_p_value']:.4f} (n_perm={hier['n_perm']})"
        )
        report_lines.append(
            f"- month_to_season_accuracy={hier['month_to_season_accuracy']:.3f}, month_to_season_p={hier['month_to_season_p_value']:.4f}"
        )
        month_plane = hier.get("month_plane")
        if month_plane:
            if isinstance(month_plane[0], str):
                plane_desc = f"{month_plane[0]} + {month_plane[1]}"
            else:
                plane_desc = f"PC{month_plane[0]}-PC{month_plane[1]}"
            metrics = (
                f"order_alignment={hier['month_order_alignment']:.3f}, "
                f"ring_cv={hier['month_ring_cv']:.3f}"
            )
            if "month_order_corr" in hier:
                metrics += f", order_corr={hier['month_order_corr']:.3f}"
            if "season_axis_eta2" in hier:
                metrics += f", season_axis_eta2={hier['season_axis_eta2']:.3f}"
            if "month_silhouette" in hier:
                metrics += f", month_silhouette={hier['month_silhouette']:.3f}"
            if "season_silhouette" in hier:
                metrics += f", season_silhouette={hier['season_silhouette']:.3f}"
            report_lines.append(f"- month_plane={plane_desc} | {metrics}")
        if "season_order_alignment" in hier:
            report_lines.append(
                f"- season_order_alignment={hier['season_order_alignment']:.3f}, "
                f"season_ring_cv={hier['season_ring_cv']:.3f} (derived from month centroids)"
            )
        report_lines.append(
            f"- n_samples={hier['n_samples']}, n_labels={hier['n_labels']}, categories={', '.join(hier['categories'])}"
        )
        report_lines.append("")
    report_lines.append("## Qualitative interpretation (aligned with 2405.14860v3)")
    report_lines.append(
        "- The figure uses an order-aligned 2D projection (ridge fit to circular labels) to make the cyclic structure for days/months explicit; PCA plane diagnostics above quantify how well PCA axes capture the circle."
    )
    report_lines.append(
        "- Years are evaluated with the most monotonic PCA axis (Spearman), highlighting linear temporal structure rather than a strong circular ordering."
    )
    report_lines.append(
        "- Main ring plots are normalized to a unit circle (angles only), explicitly ignoring radial magnitude to emphasize cyclic ordering; embedding comparisons show raw vs circle-projected points."
    )
    report_lines.append(
        "- MDF-style PCA scatter plots mirror Fig. 1 of 2405.14860v3 by showing the PCA plane with the clearest cyclic structure (days/months) or strongest year gradient."
    )
    if args.qualitative:
        report_lines.append(
            "- Additional qualitative panels show token-level activation profiles, year-axis trends, and category scatter to illustrate temporal selectivity and scale separation."
        )
    report_lines.append("")
    report_lines.append("## Top decoder tokens (group direction)")
    report_lines.append(", ".join([repr(t["token"]) for t in analysis["top_tokens"][:12]]))
    report_lines.append("")
    report_lines.append("## Top activating prompts")
    for concept in ["days", "months", "years"]:
        report_lines.append(f"### {concept}")
        for item in analysis["top_prompts"][concept][:5]:
            report_lines.append(
                f"- norm={item['group_norm']:.4f} | term={item['term']} | {item['prompt']}"
            )
        report_lines.append("")

    (output_dir / "group_1916_temporal_paper_report.md").write_text("\n".join(report_lines))

    print("Saved outputs to", output_dir)


if __name__ == "__main__":
    main()
