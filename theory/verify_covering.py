"""
Theory Verification Experiments for Feature Splitting

This script implements 3 controlled synthetic experiments to empirically verify
the key theoretical predictions from the paper "Why Feature Splitting Happens in SAEs":

1. Epsilon-Scaling: Verify N ~ (t/ε)^(m-1) scaling law
2. Line-Covering Number: Measure actual L_f(ε) vs theoretical bound
3. Directional Richness: Test angular cap size (ρ) effect

Each experiment generates toy data and uses spherical k-means to isolate the
theoretical line-covering phenomena.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy import stats
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str):
    """Print with immediate flush for visibility in conda environments."""
    print(msg, flush=True)


# =============================================================================
# Data Generation Utilities
# =============================================================================


def sample_uniform_sphere(d: int, n_samples: int) -> torch.Tensor:
    """Sample uniformly from S^{d-1}."""
    z = torch.randn(n_samples, d)
    return z / z.norm(dim=-1, keepdim=True)


def sample_from_spherical_cap(
    d: int, rho: float, n_samples: int, center: torch.Tensor = None
) -> torch.Tensor:
    """
    Sample uniformly from a spherical cap of angular radius ρ on S^{d-1}.

    Args:
        d: Dimension of the sphere
        rho: Angular radius of the cap (in radians, 0 to π)
        n_samples: Number of samples
        center: Center of the cap (defaults to e_1)

    Returns:
        Points on S^{d-1} within angular distance ρ from center
    """
    if center is None:
        center = torch.zeros(d)
        center[0] = 1.0

    # Sample theta uniformly in [0, rho] with correct measure
    # The PDF for uniform sampling is proportional to sin^{d-2}(theta)
    # Use rejection sampling for simplicity
    samples = []
    while len(samples) < n_samples:
        # Sample candidate theta
        theta = torch.rand(n_samples * 2) * rho
        # Accept with probability sin^{d-2}(theta) / sin^{d-2}(rho)
        if d > 2:
            accept_prob = (torch.sin(theta) / torch.sin(torch.tensor(rho))).pow(d - 2)
        else:
            accept_prob = torch.ones_like(theta)
        accept = torch.rand_like(theta) < accept_prob
        samples.extend(theta[accept].tolist())

    theta = torch.tensor(samples[:n_samples])

    # Sample perpendicular direction uniformly
    if d == 2:
        # In 2D, perpendicular is just ±1
        perp = torch.ones(n_samples, 1)
        perp[torch.rand(n_samples) < 0.5, 0] = -1
    else:
        perp = sample_uniform_sphere(d - 1, n_samples)

    # Construct points: cos(theta) * center + sin(theta) * perp_in_tangent_space
    points = torch.zeros(n_samples, d)
    points[:, 0] = torch.cos(theta)
    points[:, 1:] = torch.sin(theta).unsqueeze(-1) * perp

    # If center is not e_1, rotate appropriately
    if not torch.allclose(center, torch.eye(d)[0]):
        # Apply rotation that maps e_1 to center
        # For simplicity, use Householder reflection
        e1 = torch.zeros(d)
        e1[0] = 1.0
        v = e1 - center
        if v.norm() > 1e-10:
            v = v / v.norm()
            points = points - 2 * (points @ v).unsqueeze(-1) * v.unsqueeze(0)

    return points


def generate_subspace_basis(
    d_model: int, subspace_dim: int, offset: int = 0
) -> torch.Tensor:
    """Generate an orthonormal basis for a subspace."""
    U = torch.zeros(d_model, subspace_dim)
    U[offset : offset + subspace_dim, :] = torch.eye(subspace_dim)
    return U


def generate_single_feature_data(
    d_model: int,
    feature_dim: int,
    n_samples: int,
    magnitude: float = 1.0,
    noise_std: float = 0.0,
    cap_size: float = math.pi,  # Full sphere by default
    subspace_basis: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data from a single multi-dimensional feature.

    Returns:
        data: [n_samples, d_model] feature embeddings
        directions: [n_samples, feature_dim] the underlying directions
    """
    if subspace_basis is None:
        subspace_basis = generate_subspace_basis(d_model, feature_dim)

    # Sample directions (either full sphere or cap)
    if cap_size >= math.pi:
        directions = sample_uniform_sphere(feature_dim, n_samples)
    else:
        directions = sample_from_spherical_cap(feature_dim, cap_size, n_samples)

    # Embed in ambient space with fixed magnitude
    data = magnitude * (directions @ subspace_basis.T)

    # Add noise
    if noise_std > 0:
        data = data + torch.randn_like(data) * noise_std

    return data, directions


# =============================================================================
# Simple SAE Models
# =============================================================================


class SimpleTopKSAE(nn.Module):
    """
    Minimal TopK SAE for theory verification.
    Uses fixed k atoms per sample with normalized decoder.
    """

    def __init__(self, d_model: int, n_atoms: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_atoms = n_atoms
        self.k = k

        # Decoder: [n_atoms, d_model] - each row is a unit-norm atom
        self.W_dec = nn.Parameter(torch.randn(n_atoms, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x to sparse latent codes."""
        # Similarity to each atom
        x_centered = x - self.b_dec
        similarities = x_centered @ self.W_dec.T  # [batch, n_atoms]

        # TopK selection
        topk_vals, topk_idx = similarities.topk(self.k, dim=-1)

        # Sparse codes (using similarities as activations)
        z = torch.zeros_like(similarities)
        z.scatter_(1, topk_idx, F.relu(topk_vals))

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstruction."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class SimpleGroupSAE(nn.Module):
    """
    Minimal Group-based SAE (ManifoldSAE) for theory verification.
    Uses group-level TopK selection.
    """

    def __init__(self, d_model: int, n_groups: int, group_dim: int, k_groups: int):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.group_dim = group_dim
        self.n_atoms = n_groups * group_dim
        self.k_groups = k_groups

        self.W_dec = nn.Parameter(torch.randn(self.n_atoms, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        similarities = x_centered @ self.W_dec.T  # [batch, n_atoms]

        # Reshape to groups: [batch, n_groups, group_dim]
        batch_size = x.shape[0]
        groups = similarities.view(batch_size, self.n_groups, self.group_dim)
        group_norms = groups.norm(dim=-1)  # [batch, n_groups]

        # TopK on group norms
        topk_vals, topk_idx = group_norms.topk(self.k_groups, dim=-1)

        # Mask non-selected groups
        mask = torch.zeros_like(group_norms)
        mask.scatter_(1, topk_idx, 1.0)

        # Apply mask
        masked_groups = groups * mask.unsqueeze(-1)
        z = masked_groups.view(batch_size, self.n_atoms)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# =============================================================================
# Training Utilities
# =============================================================================


def train_sae(
    model: nn.Module,
    data: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = DEVICE,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict:
    """Train SAE and return training history."""
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    history = {"mse": [], "l0": []}

    epoch_iter = tqdm(range(n_epochs), desc="Training", disable=not show_progress)
    for epoch in epoch_iter:
        epoch_mse = 0.0
        epoch_l0 = 0.0
        n_batches = 0

        for (batch,) in loader:
            x_hat, z = model(batch)
            loss = F.mse_loss(x_hat, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if hasattr(model, "_normalize_decoder"):
                model._normalize_decoder()

            epoch_mse += loss.item()
            epoch_l0 += (z.abs() > 1e-8).float().sum(dim=-1).mean().item()
            n_batches += 1

        scheduler.step()
        history["mse"].append(epoch_mse / n_batches)
        history["l0"].append(epoch_l0 / n_batches)

        epoch_iter.set_postfix(mse=history["mse"][-1], l0=history["l0"][-1])

        if verbose and (epoch + 1) % 20 == 0:
            log(
                f"Epoch {epoch+1}: MSE={history['mse'][-1]:.6f}, L0={history['l0'][-1]:.1f}"
            )

    return history


def evaluate_reconstruction_error(
    model: nn.Module,
    data: torch.Tensor,
    device: str = DEVICE,
) -> float:
    """Compute mean reconstruction error on data."""
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        x_hat, _ = model(data)
        mse = F.mse_loss(x_hat, data).item()

    return mse


def count_atoms_used(
    model: nn.Module,
    data: torch.Tensor,
    subspace_basis: torch.Tensor,
    threshold: float = 0.3,
    device: str = DEVICE,
) -> int:
    """
    Count how many decoder atoms are associated with a feature subspace.

    An atom is associated if its projection onto the subspace exceeds threshold
    of its total norm squared.
    """
    decoder = model.W_dec.detach().cpu()  # [n_atoms, d_model]

    # Project each atom onto subspace
    proj = decoder @ subspace_basis  # [n_atoms, subspace_dim]
    proj_norms_sq = (proj**2).sum(dim=-1)  # [n_atoms]
    atom_norms_sq = (decoder**2).sum(dim=-1)  # [n_atoms]

    overlap = proj_norms_sq / (atom_norms_sq + 1e-10)
    n_associated = (overlap > threshold).sum().item()

    return n_associated


def measure_line_covering(
    model: nn.Module,
    data: torch.Tensor,
    device: str = DEVICE,
) -> tuple[int, float, float]:
    """
    Measure line-covering metrics from SAE decoder.

    Uses paper's definition: worst-case (sup) error over all points.

    Returns:
        n_atoms_used: Number of distinct atoms that achieve best fit for some point
        max_dist: WORST-CASE (sup) distance to best-fit line (paper's definition)
        mean_dist: Mean distance to best-fit line (for comparison)
    """
    model = model.to(device)
    model.eval()

    decoder = model.W_dec.detach()  # [n_atoms, d_model]
    data = data.to(device)

    with torch.no_grad():
        # For each data point, find which atom's line best fits it
        # dist(x, span(w)) = ||x||^2 - <x, w>^2 (since ||w||=1)
        x_norms_sq = (data**2).sum(dim=-1)  # [n_samples]
        dot_products = data @ decoder.T  # [n_samples, n_atoms]

        # Distance to each line
        dist_sq = x_norms_sq.unsqueeze(-1) - dot_products**2  # [n_samples, n_atoms]
        dist_sq = dist_sq.clamp(min=0)  # Numerical stability

        # Best atom for each point
        min_dist_sq, best_atoms = dist_sq.min(dim=-1)
        unique_atoms = best_atoms.unique()

        # WORST-CASE (sup) distance - paper's definition
        max_dist = min_dist_sq.sqrt().max().item()
        # Mean distance for comparison
        mean_dist = min_dist_sq.sqrt().mean().item()

    return len(unique_atoms), max_dist, mean_dist


# =============================================================================
# Experiment 1: Epsilon-Scaling (Line-Covering via Spherical K-Means)
# =============================================================================


def spherical_kmeans(
    data: torch.Tensor, n_clusters: int, n_iter: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spherical k-means: cluster points on unit sphere by angle.

    Returns:
        centers: [n_clusters, d] unit-norm cluster centers
        assignments: [n] cluster assignment for each point
    """
    n, d = data.shape

    # Normalize data to unit sphere
    data_norm = F.normalize(data, dim=-1)

    # Initialize centers randomly from data points
    idx = torch.randperm(n)[:n_clusters]
    centers = data_norm[idx].clone()

    for _ in range(n_iter):
        # Assignment: find closest center (max cosine similarity)
        sims = data_norm @ centers.T  # [n, n_clusters]
        assignments = sims.abs().argmax(dim=-1)  # abs because lines are bidirectional

        # Update: mean of assigned points, re-normalized
        new_centers = torch.zeros_like(centers)
        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                # Lines are bidirectional: align signs before averaging.
                cluster = data_norm[mask]
                dots = cluster @ centers[c]
                signs = torch.where(dots >= 0, 1.0, -1.0).to(cluster.dtype)
                cluster = cluster * signs.unsqueeze(-1)
                cluster_mean = cluster.mean(dim=0)
                new_centers[c] = F.normalize(cluster_mean, dim=0)
            else:
                # Keep old center if cluster is empty
                new_centers[c] = centers[c]

        centers = new_centers

    # Final assignment
    sims = data_norm @ centers.T
    assignments = sims.abs().argmax(dim=-1)

    return centers, assignments


def spherical_kmeans_best(
    data: torch.Tensor,
    n_clusters: int,
    n_iter: int = 80,
    n_restarts: int = 5,
    objective: str = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run spherical k-means with multiple random restarts and pick the best solution.

    Args:
        data: [n, d] (need not be normalized; clustering is on the sphere)
        n_clusters: number of clusters/lines
        n_iter: iterations per restart
        n_restarts: number of random restarts
        objective: "max" or "mean" (minimize max/mean line-covering error on data)
    """
    if objective not in {"max", "mean"}:
        raise ValueError(f"objective must be 'max' or 'mean', got {objective}")

    best_centers = None
    best_assignments = None
    best_score = float("inf")

    for r in range(n_restarts):
        torch.manual_seed(SEED + 12345 + 97 * r + 1000 * n_clusters)
        centers, assignments = spherical_kmeans(
            data, n_clusters=n_clusters, n_iter=n_iter
        )
        eps_max, eps_mean = compute_line_covering_error(data, centers)
        score = eps_max if objective == "max" else eps_mean
        if score < best_score:
            best_score = score
            best_centers = centers.detach().clone()
            best_assignments = assignments.detach().clone()

    assert best_centers is not None and best_assignments is not None
    return best_centers, best_assignments


def compute_line_covering_error(
    data: torch.Tensor, centers: torch.Tensor
) -> tuple[float, float]:
    """
    Compute line-covering error: for each point, distance to best-fit line.

    Paper's definition uses WORST-CASE (sup):
    ε = sup_x min_j dist(x, span(w_j))

    Returns:
        max_error: Worst-case (sup) error - paper's definition
        mean_error: Mean error for comparison
    """
    # Normalize data and centers
    data_norm = F.normalize(data, dim=-1)
    centers_norm = F.normalize(centers, dim=-1)

    # Cosine similarities
    cos_sims = data_norm @ centers_norm.T  # [n, k]

    # Distance to each line: ||x|| * sqrt(1 - cos^2) = ||x|| * |sin|
    data_norms = data.norm(dim=-1, keepdim=True)  # [n, 1]
    sin_sq = 1 - cos_sims**2  # [n, k]
    sin_sq = sin_sq.clamp(min=0)  # Numerical stability

    # Distance to each line
    dist_to_lines = data_norms * sin_sq.sqrt()  # [n, k]

    # Min distance to any line for each point
    min_dist, _ = dist_to_lines.min(dim=-1)  # [n]

    # Paper uses WORST-CASE (sup)
    max_error = min_dist.max().item()
    mean_error = min_dist.mean().item()

    return max_error, mean_error


# =============================================================================
# K-Subspaces (Grassmannian k-means) for r-dimensional group coverings
# =============================================================================


def _orthonormalize_columns(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return an orthonormal basis for the column space of x (via QR)."""
    q, _ = torch.linalg.qr(x, mode="reduced")
    if not torch.isfinite(q).all():
        raise ValueError("Non-finite values during orthonormalization.")
    col_norms = q.norm(dim=0, keepdim=True).clamp(min=eps)
    return q / col_norms


def fit_k_subspaces(
    directions: torch.Tensor,
    n_subspaces: int,
    subspace_dim: int,
    n_iter: int = 30,
    n_restarts: int = 3,
    device: str = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit k r-dimensional subspaces to unit directions using k-subspaces (alt-min).

    Args:
        directions: [n, d] unit vectors in R^d (feature-coordinate space)
        n_subspaces: number of subspaces (k)
        subspace_dim: target subspace dimension (r)
        n_iter: number of assignment/update iterations
        n_restarts: take best of multiple random inits (lowest mean residual)

    Returns:
        bases: [k, d, r_eff] orthonormal bases for each subspace
        assignments: [n] integer assignment per point for best restart
    """
    if directions.dim() != 2:
        raise ValueError(f"directions must be rank-2, got shape {directions.shape}")
    n, d = directions.shape
    if n_subspaces <= 0:
        raise ValueError("n_subspaces must be positive.")
    r_eff = int(min(max(subspace_dim, 1), d))

    X = F.normalize(directions.to(device), dim=-1)

    best_bases = None
    best_assignments = None
    best_mean_residual = float("inf")

    for restart in range(n_restarts):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + 1000 * restart + 17)

        idx = torch.randperm(n, generator=g, device=device)[:n_subspaces]
        init_dirs = X[idx]  # [k, d]

        bases = []
        for j in range(n_subspaces):
            noise = torch.randn(d, r_eff, generator=g, device=device, dtype=X.dtype)
            noise[:, 0] = init_dirs[j]  # bias to include this direction
            B = _orthonormalize_columns(noise)
            bases.append(B)
        bases = torch.stack(bases, dim=0)  # [k, d, r_eff]

        assignments = torch.zeros(n, device=device, dtype=torch.long)

        for _ in range(n_iter):
            # Assignment: choose subspace maximizing ||B^T x||^2
            proj = torch.einsum("nd,kdr->nkr", X, bases)  # [n, k, r]
            proj_norm_sq = (proj**2).sum(dim=-1)  # [n, k]
            assignments = proj_norm_sq.argmax(dim=-1)

            # Update: PCA per cluster
            new_bases = []
            for j in range(n_subspaces):
                mask = assignments == j
                if int(mask.sum()) < r_eff:
                    new_bases.append(bases[j])
                    continue

                Xj = X[mask]  # [n_j, d]
                C = Xj.T @ Xj  # [d, d]
                _, evecs = torch.linalg.eigh(C)
                B = evecs[:, -r_eff:]
                B = _orthonormalize_columns(B)
                new_bases.append(B)
            bases = torch.stack(new_bases, dim=0)

        with torch.no_grad():
            proj = torch.einsum("nd,kdr->nkr", X, bases)
            proj_norm_sq = (proj**2).sum(dim=-1)  # [n, k]
            best_proj = proj_norm_sq.max(dim=-1).values  # [n]
            mean_residual = (1.0 - best_proj).clamp(min=0).sqrt().mean().item()

        if mean_residual < best_mean_residual:
            best_mean_residual = mean_residual
            best_bases = bases.detach().clone()
            best_assignments = assignments.detach().clone()

    assert best_bases is not None and best_assignments is not None
    return best_bases, best_assignments


def compute_subspace_covering_error(
    directions: torch.Tensor,
    bases: torch.Tensor,
    magnitude: float = 1.0,
) -> tuple[float, float]:
    """
    Compute worst-case and mean distance to the nearest subspace.

    Args:
        directions: [n, d] unit vectors (feature-coordinate space)
        bases: [k, d, r] orthonormal bases for subspaces
        magnitude: scalar t (points are h = t * z)
    """
    X = F.normalize(directions, dim=-1)
    proj = torch.einsum("nd,kdr->nkr", X, bases)  # [n, k, r]
    proj_norm_sq = (proj**2).sum(dim=-1)  # [n, k]
    best_proj = proj_norm_sq.max(dim=-1).values  # [n]
    residual = (1.0 - best_proj).clamp(min=0).sqrt()
    dist = magnitude * residual
    return dist.max().item(), dist.mean().item()


def run_epsilon_scaling_experiment(
    d_model: int = 64,
    feature_dims: list[int] = None,
    n_samples: int = 10000,
    magnitude: float = 1.0,
    n_atoms_list: list[int] = None,
    n_trials: int = 3,
    n_epochs: int = 100,  # Keep for compatibility but unused
    output_dir: Path = None,
    save_results: bool = True,
) -> dict:
    """
    Experiment 1: Verify N ~ (t/ε)^(m-1) scaling via DIRECT LINE COVERING.

    Key insight from Theorem 1: For a single m-dimensional feature on sphere,
    the minimum number of lines needed to ε-cover all points scales as
    N ≥ C(t/ε)^(m-1).

    Methodology:
    - For each dimension m ∈ {2, 3, 4, 5}:
      - Sample data uniformly from m-dimensional unit sphere
      - Use spherical k-means to find N optimal line directions
      - Measure achieved ε (mean distance to closest line)
      - Plot log(N) vs log(t/ε), expect slope ~ (m-1)

    This directly tests the geometric line-covering theorem without SAE training.

    This is a standalone experiment that:
    - Creates its own output directory if not provided
    - Saves results to JSON
    - Generates plots
    - Prints summary
    """
    if feature_dims is None:
        feature_dims = [2, 4, 8, 16]  # More realistic dimensions
    if n_atoms_list is None:
        n_atoms_list = [4, 8, 16, 32, 64, 128, 256, 512]
    if output_dir is None:
        output_dir = Path("./experiment_results/epsilon_scaling")

    log("\\n" + "=" * 70)
    log("EXPERIMENT 1: Epsilon-Scaling (Direct Line Covering)")
    log("=" * 70)
    log("Using spherical k-means to find optimal line covering")
    log(f"Testing dimensions m: {feature_dims}")
    log(f"Number of lines N: {n_atoms_list}")
    log(f"Feature magnitude t: {magnitude}")
    log(f"Output directory: {output_dir}")

    results = {"epsilon_scaling": []}

    for d_i in feature_dims:
        log(f"\\n--- Feature dimension d_i = {d_i} ---")
        log(f"    Theorem predicts: log(N) = {d_i-1} * log(t/ε) + C")

        subspace_basis = generate_subspace_basis(d_model, d_i)

        dim_results = {"d_i": d_i, "n_atoms_vs_epsilon": []}

        for n_atoms in n_atoms_list:
            trial_epsilons = []

            for trial in range(n_trials):
                torch.manual_seed(SEED + trial * 100 + d_i * 1000)

                # Generate data uniformly on d_i-dimensional sphere
                data, directions = generate_single_feature_data(
                    d_model=d_model,
                    feature_dim=d_i,
                    n_samples=n_samples,
                    magnitude=magnitude,
                )

                # Find optimal line covering using spherical k-means
                # Work in the d_i-dimensional feature space (not d_model space)
                centers, _ = spherical_kmeans(directions, n_clusters=n_atoms, n_iter=50)

                # Embed centers back into d_model space
                centers_embedded = centers @ subspace_basis.T  # [n_atoms, d_model]

                # Compute line-covering error (paper uses worst-case/max)
                epsilon_max, epsilon_mean = compute_line_covering_error(
                    data, centers_embedded
                )
                trial_epsilons.append(epsilon_max)  # Use max (paper's definition)

            mean_eps = np.mean(trial_epsilons)
            std_eps = np.std(trial_epsilons)

            dim_results["n_atoms_vs_epsilon"].append(
                {
                    "n_atoms": n_atoms,
                    "achieved_epsilon": mean_eps,
                    "epsilon_std": std_eps,
                    "ratio_t_over_eps": magnitude / mean_eps,
                }
            )

            log(
                f"  N={n_atoms:3d}: ε = {mean_eps:.4f} ± {std_eps:.4f}, t/ε = {magnitude/mean_eps:.2f}"
            )

        # Fit log-log: log(N) = (d_i-1) * log(t/ε) + C
        log_n = np.log([r["n_atoms"] for r in dim_results["n_atoms_vs_epsilon"]])
        log_t_over_eps = np.log(
            [r["ratio_t_over_eps"] for r in dim_results["n_atoms_vs_epsilon"]]
        )

        if (
            len(log_n) > 1
            and not np.any(np.isnan(log_t_over_eps))
            and not np.any(np.isinf(log_t_over_eps))
        ):
            slope, intercept, r_value, _, _ = stats.linregress(log_t_over_eps, log_n)
            dim_results["empirical_slope"] = slope
            dim_results["theoretical_slope"] = d_i - 1
            dim_results["r_squared"] = r_value**2

            log(f"  Empirical: log(N) = {slope:.2f} * log(t/ε) + C")
            log(f"  Theory:    log(N) = {d_i-1:.0f} * log(t/ε) + C")
            log(f"  R² = {r_value**2:.3f}")

        results["epsilon_scaling"].append(dim_results)

    # Save results and generate plots if requested
    if save_results:
        save_and_plot_experiment(results, "epsilon_scaling", output_dir)

    return results


# =============================================================================
# Experiment 4: Synthetic validation of feature splitting (single-feature slice)
# =============================================================================


def run_synthetic_feature_splitting_validation(
    d_model: int = 64,
    feature_dims: list[int] = None,
    magnitude: float = 1.0,
    n_train: int = 10000,
    n_holdout: int = 50000,
    n_list_lines: list[int] = None,
    n_list_subspaces: list[int] = None,
    r_list: list[int] = None,
    line_kmeans_restarts: int = 5,
    ksubspaces_iters: int = 30,
    ksubspaces_restarts: int = 3,
    output_dir: Path = None,
    save_results: bool = True,
) -> dict:
    """
    Synthetic single-feature slice experiment (noiseless):

    - z ~ Unif(S^{d_i-1})
    - h = t V_i z (V_i orthonormal), so h ∈ V_i and ||h||=t
    - Fit N line directions via spherical k-means, compute ε_max(N) on held-out directions
    - Fit N r-subspaces via k-subspaces, compute ε_max^{(r)}(N) on held-out directions
    """
    if feature_dims is None:
        feature_dims = [2, 4, 8, 16]
    if n_list_lines is None:
        n_list_lines = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if n_list_subspaces is None:
        # Keep this moderate by default: k-subspaces scales poorly at huge N.
        n_list_subspaces = [1, 2, 4, 8, 16, 32, 64, 128]
    if r_list is None:
        r_list = [1, 2, 4, 8, 16]
    if output_dir is None:
        output_dir = Path("./experiment_results/synthetic_feature_splitting")

    log("\n" + "=" * 70)
    log("EXPERIMENT 4: Synthetic feature splitting validation (single-feature slice)")
    log("=" * 70)
    log(f"Ambient dimension d={d_model} (unused; distances invariant), t={magnitude}")
    log(f"d_i list: {feature_dims}")
    log(f"N list (lines): {n_list_lines}")
    log(f"N list (subspaces): {n_list_subspaces}")
    log(f"r list: {r_list}")
    log(f"Train dirs: {n_train}, Held-out dirs: {n_holdout}")
    log(f"Line k-means restarts: {line_kmeans_restarts}")
    log(f"Output directory: {output_dir}")

    results: dict = {
        "synthetic_feature_splitting": {
            "d_model": d_model,
            "magnitude": magnitude,
            "n_train": n_train,
            "n_holdout": n_holdout,
            "feature_dims": feature_dims,
            "n_list_lines": n_list_lines,
            "n_list_subspaces": n_list_subspaces,
            "r_list": r_list,
            "line_covering": [],
            "subspace_covering": [],
        }
    }

    for d_i in feature_dims:
        log(f"\n--- d_i = {d_i} ---")
        torch.manual_seed(SEED + 10000 + d_i)
        train_dirs = sample_uniform_sphere(d_i, n_train)
        holdout_dirs = sample_uniform_sphere(d_i, n_holdout)

        # (i) Line coverings: ε_max(N) = sup_z min_j dist(t z, span(d_j))
        line_curve = {"d_i": d_i, "N_vs_eps_max": [], "N_vs_eps_mean": []}
        for N in n_list_lines:
            # To keep runtime stable for large N, fit on a sized subset of directions.
            # Heuristic: need at least O(N) points per cluster.
            fit_size = int(min(n_train, max(4000, 80 * int(N))))
            if fit_size < n_train:
                g = torch.Generator()
                g.manual_seed(SEED + 222 + d_i * 100 + int(N))
                idx = torch.randperm(n_train, generator=g)[:fit_size]
                fit_dirs = train_dirs[idx]
            else:
                fit_dirs = train_dirs

            # Fit on training directions, evaluate worst-case on held-out directions.
            centers, _ = spherical_kmeans_best(
                magnitude * fit_dirs,
                n_clusters=N,
                n_iter=80,
                n_restarts=line_kmeans_restarts,
                objective="max",
            )
            eps_max, eps_mean = compute_line_covering_error(
                magnitude * holdout_dirs, magnitude * centers
            )
            line_curve["N_vs_eps_max"].append({"N": int(N), "eps_max": float(eps_max)})
            line_curve["N_vs_eps_mean"].append(
                {"N": int(N), "eps_mean": float(eps_mean)}
            )
            log(f"  Lines: N={N:4d} -> eps_max={eps_max:.4f}, eps_mean={eps_mean:.4f}")

        # Empirical slope of log eps vs log N (theory: -1/(d_i-1))
        Ns = np.array([p["N"] for p in line_curve["N_vs_eps_max"]], dtype=float)
        eps = np.array([p["eps_max"] for p in line_curve["N_vs_eps_max"]], dtype=float)
        valid = (Ns > 0) & np.isfinite(eps) & (eps > 0)
        if valid.sum() >= 2:
            slope, intercept, r_value, _, _ = stats.linregress(
                np.log(Ns[valid]), np.log(eps[valid])
            )
            line_curve["loglog_slope_eps_vs_N"] = float(slope)
            line_curve["theory_slope_eps_vs_N"] = float(-1.0 / (d_i - 1))
            line_curve["r_squared"] = float(r_value**2)
            log(
                f"  Line slope (log eps vs log N): {slope:.3f} (theory {-1.0/(d_i-1):.3f}), R²={r_value**2:.3f}"
            )

        results["synthetic_feature_splitting"]["line_covering"].append(line_curve)

        # (ii) r-subspace coverings: ε_max^{(r)}(N)
        for r in r_list:
            r_eff = min(int(r), d_i)
            sub_curve = {
                "d_i": d_i,
                "r": int(r),
                "r_eff": int(r_eff),
                "N_vs_eps_max": [],
                "N_vs_eps_mean": [],
            }

            for N in n_list_subspaces:
                if r >= d_i:
                    # One subspace spanning all of R^{d_i} achieves ε=0
                    eye = torch.eye(
                        d_i, device=holdout_dirs.device, dtype=holdout_dirs.dtype
                    )
                    bases = eye.unsqueeze(0).repeat(int(N), 1, 1)  # [N, d_i, d_i]
                    eps_max_r, eps_mean_r = compute_subspace_covering_error(
                        holdout_dirs, bases, magnitude=magnitude
                    )
                else:
                    fit_size = int(min(n_train, max(4000, 120 * int(N))))
                    if fit_size < n_train:
                        g = torch.Generator()
                        g.manual_seed(SEED + 333 + d_i * 100 + int(N) + 10 * int(r))
                        idx = torch.randperm(n_train, generator=g)[:fit_size]
                        fit_dirs = train_dirs[idx]
                    else:
                        fit_dirs = train_dirs

                    bases, _ = fit_k_subspaces(
                        fit_dirs,
                        n_subspaces=int(N),
                        subspace_dim=int(r),
                        n_iter=ksubspaces_iters,
                        n_restarts=ksubspaces_restarts,
                        device=DEVICE,
                    )
                    eps_max_r, eps_mean_r = compute_subspace_covering_error(
                        holdout_dirs.to(bases.device), bases, magnitude=magnitude
                    )

                sub_curve["N_vs_eps_max"].append(
                    {"N": int(N), "eps_max": float(eps_max_r)}
                )
                sub_curve["N_vs_eps_mean"].append(
                    {"N": int(N), "eps_mean": float(eps_mean_r)}
                )
                log(
                    f"  Subspaces: r={r:2d}, N={N:4d} -> eps_max={eps_max_r:.4f}, eps_mean={eps_mean_r:.4f}"
                )

            results["synthetic_feature_splitting"]["subspace_covering"].append(
                sub_curve
            )

    if save_results:
        save_and_plot_experiment(results, "synthetic_feature_splitting", output_dir)

    return results


# =============================================================================
# Experiment 2: Line-Covering Number L_f(ε) Measurement
# =============================================================================


def compute_optimal_line_covering_error(
    directions: torch.Tensor,
    subspace_basis: torch.Tensor,
    n_atoms: int,
    n_iter: int = 50,
) -> float:
    """Compute optimal ε_max achievable with n_atoms lines using spherical k-means."""
    centers, _ = spherical_kmeans(directions, n_clusters=n_atoms, n_iter=n_iter)
    centers_embedded = centers @ subspace_basis.T

    # Compute data in ambient space
    data = directions @ subspace_basis.T

    eps_max, _ = compute_line_covering_error(data, centers_embedded)
    return eps_max


def find_minimum_atoms_for_epsilon(
    directions: torch.Tensor,
    subspace_basis: torch.Tensor,
    target_epsilon: float,
    max_atoms: int = 10000,
) -> int:
    """
    Binary search for minimum N such that optimal line-covering achieves ε_max ≤ target_epsilon.

    This computes L_f(target_epsilon) from Definition 3 of the paper.
    """
    low, high = 1, max_atoms

    # First check if max_atoms is sufficient
    eps_at_max = compute_optimal_line_covering_error(
        directions, subspace_basis, max_atoms
    )
    if eps_at_max > target_epsilon:
        return max_atoms  # Would need more than max_atoms

    # Binary search
    while low < high:
        mid = (low + high) // 2
        eps_mid = compute_optimal_line_covering_error(directions, subspace_basis, mid)

        if eps_mid <= target_epsilon:
            high = mid  # Can achieve with mid atoms, try fewer
        else:
            low = mid + 1  # Need more atoms

    return low


def run_line_covering_experiment(
    d_model: int = 256,
    feature_dims: list[int] = None,
    target_epsilons: list[float] = None,
    n_samples: int = 10000,
    magnitude: float = 1.0,
    max_atoms: int = 10000,
    output_dir: Path = None,
    save_results: bool = True,
) -> dict:
    """
    Experiment 2: Compute the line-covering number L_f(ε) for various target ε values.

    Paper's Definition (Def 3):
        L_f(ε) = min{N : ∃ w_1,...,w_N s.t. sup_x min_j dist(x, span(w_j)) ≤ ε}

    For each feature dimension m and target ε:
        1. Binary search for minimum N achieving ε_max ≤ ε
        2. This gives L_f(ε) empirically
        3. Verify: L_f(ε) ≥ C(t/ε)^(m-1) (Theorem 1)

    This is a standalone experiment that:
    - Creates its own output directory if not provided
    - Saves results to JSON
    - Generates plots
    - Prints summary
    """
    if feature_dims is None:
        feature_dims = [2, 4, 8]
    if target_epsilons is None:
        target_epsilons = [0.5, 0.3, 0.2, 0.1, 0.05]
    if output_dir is None:
        output_dir = Path("./experiment_results/line_covering")

    log("\\n" + "=" * 70)
    log("EXPERIMENT 2: Line-Covering Number L_f(ε) Measurement")
    log("=" * 70)
    log(f"Ambient dimension: {d_model}")
    log(f"Feature dimensions (m): {feature_dims}")
    log(f"Target ε values: {target_epsilons}")
    log(f"Magnitude t: {magnitude}")
    log(f"Output directory: {output_dir}")
    log("\\nTheory (Theorem 1): L_f(ε) ≥ C(t/ε)^(m-1)")

    results = {"line_covering_number": []}

    for m in feature_dims:
        log(f"\\n{'='*50}")
        log(f"Feature dimension m = {m}")
        log(f"Theory predicts: L_f(ε) ≥ C(t/ε)^{m-1}")
        log(f"{'='*50}")

        # Generate data
        subspace_basis = generate_subspace_basis(d_model, m)
        _, directions = generate_single_feature_data(
            d_model=d_model,
            feature_dim=m,
            n_samples=n_samples,
            magnitude=magnitude,
        )

        dim_results = {"m": m, "epsilon_vs_L_f": []}

        for target_eps in target_epsilons:
            log(f"\\n  Target ε = {target_eps}")

            # Find minimum N to achieve this ε
            L_f = find_minimum_atoms_for_epsilon(
                directions, subspace_basis, target_eps, max_atoms=max_atoms
            )

            # Verify achieved ε
            achieved_eps = compute_optimal_line_covering_error(
                directions, subspace_basis, L_f
            )

            # Theoretical lower bound: L_f(ε) ≥ C(t/ε)^(m-1)
            C = 0.5  # Approximate constant from paper
            theoretical_bound = C * (magnitude / target_eps) ** (m - 1)

            # Check if empirical ≥ theoretical
            satisfies_bound = L_f >= theoretical_bound

            result = {
                "target_epsilon": target_eps,
                "L_f_empirical": L_f,
                "achieved_epsilon": achieved_eps,
                "theoretical_lower_bound": theoretical_bound,
                "satisfies_theorem_1": satisfies_bound,
            }
            dim_results["epsilon_vs_L_f"].append(result)

            status = "✓" if satisfies_bound else "✗"
            log(f"    L_f({target_eps}) = {L_f}")
            log(f"    Achieved ε_max = {achieved_eps:.4f}")
            log(f"    Theory: L_f ≥ {theoretical_bound:.1f}")
            log(f"    Satisfies Theorem 1: {status}")

        # Compute empirical slope: log(L_f) vs log(t/ε)
        eps_values = [r["target_epsilon"] for r in dim_results["epsilon_vs_L_f"]]
        L_f_values = [r["L_f_empirical"] for r in dim_results["epsilon_vs_L_f"]]

        if len(eps_values) >= 2 and all(L > 0 for L in L_f_values):
            log_t_over_eps = np.log([magnitude / e for e in eps_values])
            log_L_f = np.log(L_f_values)

            slope, intercept, r_value, _, _ = stats.linregress(log_t_over_eps, log_L_f)
            dim_results["empirical_slope"] = slope
            dim_results["theoretical_slope"] = m - 1
            dim_results["r_squared"] = r_value**2

            log(f"\\n  Slope Analysis:")
            log(f"    Empirical: log(L_f) = {slope:.2f} * log(t/ε) + {intercept:.2f}")
            log(f"    Theory:    log(L_f) = {m-1} * log(t/ε) + C")
            log(f"    R² = {r_value**2:.4f}")

        results["line_covering_number"].append(dim_results)

    # Save results and generate plots if requested
    if save_results:
        save_and_plot_experiment(results, "line_covering", output_dir)

    return results


# =============================================================================
# Experiment 3: Directional Richness (ρ variation) - Tests Lemma 1
# =============================================================================


def run_directional_richness_experiment(
    d_model: int = 256,
    feature_dim: int = 4,
    n_samples: int = 10000,
    cap_sizes: list[float] = None,
    target_epsilon: float = 0.3,
    max_atoms: int = 5000,
    output_dir: Path = None,
    save_results: bool = True,
) -> dict:
    """
    Experiment 3: Test Lemma 1 by varying angular cap size ρ.

    Lemma 1: If data fills cap of size ρ, then N(ε) ≥ c·ε^{-(m-1)}

    Smaller ρ (cap) = less directional diversity = fewer atoms needed.

    For each ρ:
    1. Generate data in spherical cap of size ρ
    2. Compute L_f(target_epsilon) using spherical k-means
    3. Verify: L_f increases with ρ

    This is a standalone experiment that:
    - Creates its own output directory if not provided
    - Saves results to JSON
    - Generates plots
    - Prints summary
    """
    if cap_sizes is None:
        # Restrict to ρ ≤ π/2 to avoid antipodal redundancy (lines are bidirectional)
        cap_sizes = [
            math.pi / 16,
            math.pi / 8,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            math.pi / 2,
        ]
    if output_dir is None:
        output_dir = Path("./experiment_results/directional_richness")

    log("\n" + "=" * 70)
    log("EXPERIMENT 3: Directional Richness (ρ Variation) - Testing Lemma 1")
    log("=" * 70)
    log(f"Ambient dimension: {d_model}")
    log(f"Feature dimension (m): {feature_dim}")
    log(f"Cap sizes (radians): {[f'{c:.3f}' for c in cap_sizes]}")
    log(f"Target ε: {target_epsilon}")
    log(f"Output directory: {output_dir}")
    log("\nLemma 1 predicts: L_f(ε) ≥ C·(ρ/ε)^(m-1) for cap of size ρ")
    log("Note: ρ ≤ π/2 to avoid antipodal line redundancy")

    # Compute the constant C from paper's formula
    # C = (m-1) * σ_{m-1}(cap) / σ_{m-2}(S^{m-2})
    # For cap of angle ρ: σ_{m-1}(cap) ≈ σ_{m-2}(S^{m-2}) * ∫_0^ρ sin^{m-2}(θ) dθ
    # Approximation for small ρ: ∫_0^ρ sin^{m-2}(θ) dθ ≈ ρ^{m-1} / (m-1)
    # So C ≈ 1 for the scaling L_f ≈ (ρ/ε)^{m-1}

    results = {"directional_richness": []}

    for rho in cap_sizes:
        log(f"\n--- ρ = {rho:.3f} ({np.degrees(rho):.1f}°) ---")

        # Generate subspace and data
        subspace_basis = generate_subspace_basis(d_model, feature_dim)
        _, directions = generate_single_feature_data(
            d_model=d_model,
            feature_dim=feature_dim,
            n_samples=n_samples,
            magnitude=1.0,
            cap_size=rho,
            subspace_basis=subspace_basis,
        )

        # Binary search for L_f(target_epsilon) using spherical k-means
        L_f = find_minimum_atoms_for_epsilon(
            directions, subspace_basis, target_epsilon, max_atoms=max_atoms
        )

        # Verify achieved epsilon
        achieved_eps = compute_optimal_line_covering_error(
            directions, subspace_basis, L_f
        )

        # Compute C precisely from Lemma 1's proof:
        # c = (m-1) * ∫_0^ρ sin^{m-2}(θ) dθ
        # This integral can be computed analytically or numerically
        from scipy import integrate

        def integrand(theta):
            return np.sin(theta) ** (feature_dim - 2) if feature_dim > 2 else 1.0

        integral_value, _ = integrate.quad(integrand, 0, rho)
        C_exact = (feature_dim - 1) * integral_value

        # Theoretical lower bound from Lemma 1:
        # N(U_f(r), d_angle, ε) ≥ c * ε^{-(m-1)}
        # where ε here is the angular error ≈ target_epsilon / r (with r=1)
        theoretical_bound = C_exact * (1.0 / target_epsilon) ** (feature_dim - 1)

        result = {
            "rho_radians": rho,
            "rho_degrees": np.degrees(rho),
            "L_f_epsilon": L_f,
            "achieved_epsilon": achieved_eps,
            "target_epsilon": target_epsilon,
            "theoretical_bound": theoretical_bound,
            "satisfies_lemma_1": L_f >= theoretical_bound,
        }
        results["directional_richness"].append(result)

        status = "✓" if L_f >= theoretical_bound else "✗"
        log(f"  L_f({target_epsilon}) = {L_f}")
        log(f"  Achieved ε_max = {achieved_eps:.4f}")
        log(f"  Theory: L_f ≥ {theoretical_bound:.1f}")
        log(f"  Satisfies Lemma 1: {status}")

    # Verify monotonicity: L_f should increase with ρ
    L_f_values = [r["L_f_epsilon"] for r in results["directional_richness"]]
    is_monotonic = all(
        L_f_values[i] <= L_f_values[i + 1] for i in range(len(L_f_values) - 1)
    )

    log(f"\nMonotonicity check (L_f increases with ρ): {'✓' if is_monotonic else '✗'}")
    log(f"L_f values: {L_f_values}")

    results["monotonic"] = is_monotonic

    # Save results and generate plots if requested
    if save_results:
        save_and_plot_experiment(results, "directional_richness", output_dir)

    return results


# =============================================================================
# Utility Functions
# =============================================================================


def make_serializable(obj):
    """Convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj


def parse_int_list(csv: str) -> list[int]:
    """Parse comma-separated ints (e.g. '1,2,4'). Empty -> []."""
    csv = (csv or "").strip()
    if csv == "":
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip() != ""]


def save_and_plot_experiment(results: dict, experiment_name: str, output_dir: Path):
    """
    Standalone utility to save results and generate plots for a single experiment.

    Args:
        results: Dictionary containing experiment results
        experiment_name: Name of the experiment (e.g., "epsilon_scaling")
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results to JSON
    results_path = output_dir / f"{experiment_name}_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    log(f"\nSaved results to {results_path}")

    # Generate plot
    if experiment_name == "epsilon_scaling" and "epsilon_scaling" in results:
        plot_epsilon_scaling(results, output_dir / f"{experiment_name}.pdf")
    elif experiment_name == "line_covering" and "line_covering_number" in results:
        plot_line_covering(results, output_dir / f"{experiment_name}.pdf")
    elif (
        experiment_name == "directional_richness" and "directional_richness" in results
    ):
        plot_directional_richness(results, output_dir / f"{experiment_name}.pdf")
    elif (
        experiment_name == "synthetic_feature_splitting"
        and "synthetic_feature_splitting" in results
    ):
        plot_synthetic_feature_splitting(
            results, output_dir / "synthetic_feature_splitting_lines.pdf"
        )
        plot_synthetic_group_covering(
            results, output_dir / "synthetic_feature_splitting_subspaces.pdf"
        )

    # Print summary
    log("\n" + "=" * 70)
    log(f"EXPERIMENT SUMMARY: {experiment_name.upper().replace('_', ' ')}")
    log("=" * 70)

    if experiment_name == "epsilon_scaling" and "epsilon_scaling" in results:
        for r in results["epsilon_scaling"]:
            d_i = r["d_i"]
            slope = r.get("empirical_slope", r.get("log_log_slope", 0))
            if slope == 0:
                slope = -r.get("log_log_slope", 0)
            theory = d_i - 1
            log(f"  d_i={d_i}: measured slope = {slope:.2f}, theory = {theory}")


# =============================================================================
# Visualization
# =============================================================================

# Configure matplotlib for high-quality paper plots
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "figure.dpi": 300,
    }
)


def plot_epsilon_scaling(results: dict, output_path: Path = None):
    """Plot log-log scaling of atoms vs epsilon with theoretical lines."""
    eps_results = results.get("epsilon_scaling", [])
    if not eps_results:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # fig.suptitle("Line-covering Scaling", fontsize=14, fontweight="bold", y=1.02)

    colors = plt.cm.Set2(np.linspace(0, 1, len(eps_results)))

    for i, dim_result in enumerate(eps_results):
        d_i = dim_result["d_i"]
        eps_data = dim_result.get(
            "n_atoms_vs_epsilon", dim_result.get("epsilon_vs_atoms", [])
        )

        if not eps_data:
            continue

        # Handle new format (n_atoms, achieved_epsilon) or old format (epsilon, atoms_needed)
        if "achieved_epsilon" in eps_data[0]:
            epsilons = [r["achieved_epsilon"] for r in eps_data]
            atoms = [r["n_atoms"] for r in eps_data]
        else:
            epsilons = [r["epsilon"] for r in eps_data]
            atoms = [r["atoms_needed"] for r in eps_data]

        # Plot empirical data points with solid lines
        ax.loglog(
            epsilons,
            atoms,
            "o-",
            color=colors[i],
            label=f"$d_i={d_i}$ (empirical)",
            markersize=8,
            linewidth=2.5,
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

        # Plot theoretical line: N ~ (t/ε)^(d_i-1), i.e., N ~ ε^{-(d_i-1)}
        # On log-log plot, this is a line with slope -(d_i-1)
        eps_arr = np.array(epsilons)
        theoretical_slope = -(d_i - 1)
        # Use the first data point as anchor for the theoretical line
        log_eps = np.log(eps_arr)
        log_atoms = np.log(atoms)
        # Fit intercept using first point
        intercept = log_atoms[0] - theoretical_slope * log_eps[0]
        theoretical_atoms = np.exp(intercept + theoretical_slope * log_eps)

        ax.loglog(
            eps_arr,
            theoretical_atoms,
            "--",
            color=colors[i],
            label=f"$d_i={d_i}$ (theory: slope={d_i-1})",
            linewidth=2,
            alpha=0.7,
        )

    ax.set_xlabel("Reconstruction Error $\\varepsilon$", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Lines $N$", fontsize=13, fontweight="bold")
    ax.set_title("$N \\sim (1/\\varepsilon)^{d_i-1}$", fontsize=12)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        log(f"Saved: {output_path}")

    plt.close()


def plot_line_covering(results: dict, output_path: Path = None):
    """Plot line-covering number L_f(ε) vs epsilon."""
    lc_results = results.get("line_covering_number", [])
    if not lc_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Line-Covering Number $L_f(\\varepsilon)$",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(lc_results)))

    # Plot 1: Log-log of L_f vs epsilon
    ax1 = axes[0]
    for i, dim_result in enumerate(lc_results):
        m = dim_result["m"]
        lc_data = dim_result.get("epsilon_vs_L_f", [])

        if not lc_data:
            continue

        epsilons = [r["target_epsilon"] for r in lc_data]
        L_f_values = [r["L_f_empirical"] for r in lc_data]
        theoretical_bounds = [r["theoretical_lower_bound"] for r in lc_data]

        ax1.loglog(
            epsilons,
            L_f_values,
            "o-",
            color=colors[i],
            label=f"$L_f(\\varepsilon), m={m}$",
            markersize=8,
            linewidth=2.5,
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )
        ax1.loglog(
            epsilons,
            theoretical_bounds,
            "--",
            color=colors[i],
            linewidth=2,
            alpha=0.6,
            label=f"Theory (lower bound), $m={m}$",
        )

    ax1.set_xlabel("Target Error $\\varepsilon$", fontsize=13, fontweight="bold")
    ax1.set_ylabel(
        "Line-Covering Number $L_f(\\varepsilon)$", fontsize=13, fontweight="bold"
    )
    ax1.set_title("Empirical vs Theoretical Lower Bound", fontsize=12)
    ax1.legend(
        frameon=True, fancybox=True, shadow=True, loc="best", framealpha=0.95, ncol=1
    )
    ax1.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot 2: Slope verification
    ax2 = axes[1]
    dims = []
    empirical_slopes = []
    theoretical_slopes = []
    r_squared = []

    for dim_result in lc_results:
        if "empirical_slope" in dim_result:
            dims.append(dim_result["m"])
            empirical_slopes.append(dim_result["empirical_slope"])
            theoretical_slopes.append(dim_result["theoretical_slope"])
            r_squared.append(dim_result.get("r_squared", 0))

    if dims:
        x = np.arange(len(dims))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            empirical_slopes,
            width,
            label="Empirical",
            color="#2E86AB",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.8,
        )
        bars2 = ax2.bar(
            x + width / 2,
            theoretical_slopes,
            width,
            label="Theory: $m-1$",
            color="#A23B72",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.8,
        )

        # Add R² values as text
        for i, (bar, r2) in enumerate(zip(bars1, r_squared)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"$R^2={r2:.3f}$",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax2.set_xlabel("Feature Dimension $m$", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Scaling Exponent", fontsize=13, fontweight="bold")
        ax2.set_title("Slope Verification", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"$m={d}$" for d in dims])
        ax2.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        ax2.grid(True, alpha=0.4, linestyle="--", linewidth=0.8, axis="y")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        log(f"Saved: {output_path}")

    plt.close()


def plot_directional_richness(results: dict, output_path: Path = None):
    """Plot directional richness (ρ variation) results."""
    dr_results = results.get("directional_richness", [])
    if not dr_results or not isinstance(dr_results, list) or len(dr_results) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Directional Richness", fontsize=14, fontweight="bold", y=0.98)

    rhos_rad = [r["rho_radians"] for r in dr_results]
    rhos_deg = [r["rho_degrees"] for r in dr_results]
    L_f_values = [r["L_f_epsilon"] for r in dr_results]
    theoretical_bounds = [r["theoretical_bound"] for r in dr_results]

    ax.plot(
        rhos_deg,
        L_f_values,
        "o-",
        color="#2E86AB",
        label="Empirical $L_f(\\varepsilon)$",
        markersize=9,
        linewidth=2.5,
        markeredgewidth=1.5,
        markeredgecolor="white",
        alpha=0.9,
    )
    ax.plot(
        rhos_deg,
        theoretical_bounds,
        "--",
        color="#A23B72",
        label="Theoretical Lower Bound",
        linewidth=2.5,
        alpha=0.7,
    )

    ax.set_xlabel("Angular Cap Size $\\rho$ (degrees)", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "Line-Covering Number $L_f(\\varepsilon)$", fontsize=13, fontweight="bold"
    )
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95, loc="best")
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        log(f"Saved: {output_path}")

    plt.close()


def plot_synthetic_feature_splitting(results: dict, output_path: Path = None):
    """Plot ε_max(N) curves for line coverings across d_i (log-log)."""
    block = results.get("synthetic_feature_splitting", {})
    curves = block.get("line_covering", [])
    if not curves:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(curves)))

    for i, curve in enumerate(curves):
        d_i = curve["d_i"]
        pts = curve.get("N_vs_eps_max", [])
        if not pts:
            continue
        Ns = np.array([p["N"] for p in pts], dtype=float)
        eps = np.array([p["eps_max"] for p in pts], dtype=float)

        ax.loglog(
            Ns,
            eps,
            "o-",
            color=colors[i],
            label=f"$d_i={d_i}$",
            markersize=8,
            linewidth=2.5,
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

        if len(Ns) >= 2 and np.all(np.isfinite(eps)) and np.all(eps > 0):
            slope = -1.0 / (d_i - 1)
            logN0, loge0 = np.log(Ns[0]), np.log(eps[0])
            eps_theory = np.exp(loge0 + slope * (np.log(Ns) - logN0))
            ax.loglog(
                Ns,
                eps_theory,
                "--",
                color=colors[i],
                linewidth=2,
                alpha=0.7,
                label=f"$d_i={d_i}$ theory slope $-1/({d_i-1})$",
            )

    ax.set_xlabel("Number of Lines $N$", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "Worst-case Error $\\varepsilon_{\\max}(N)$", fontsize=13, fontweight="bold"
    )
    ax.set_title("Single-feature slice: line coverings", fontsize=12)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        plt.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        log(f"Saved: {output_path}")

    plt.close()


def plot_synthetic_group_covering(results: dict, output_path: Path = None):
    """
    Plot ε_max^{(r)}(N) curves for r-subspace coverings across (d_i, r).

    One panel per d_i, with a curve per r.
    """
    block = results.get("synthetic_feature_splitting", {})
    curves = block.get("subspace_covering", [])
    if not curves:
        return

    by_dim: dict[int, list[dict]] = {}
    for c in curves:
        by_dim.setdefault(int(c["d_i"]), []).append(c)
    dims = sorted(by_dim.keys())

    ncols = 2
    nrows = int(math.ceil(len(dims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, d_i in enumerate(dims):
        ax = axes[ax_idx]
        group = sorted(by_dim[d_i], key=lambda x: int(x["r"]))
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(group)))

        for i, curve in enumerate(group):
            r = int(curve["r"])
            pts = curve.get("N_vs_eps_max", [])
            if not pts:
                continue
            Ns = np.array([p["N"] for p in pts], dtype=float)
            eps = np.array([p["eps_max"] for p in pts], dtype=float)
            ax.loglog(
                Ns,
                eps,
                "o-",
                color=colors[i],
                label=f"$r={r}$",
                markersize=7,
                linewidth=2.2,
                markeredgewidth=1.2,
                markeredgecolor="white",
                alpha=0.9,
            )

        ax.set_title(f"$d_i={d_i}$", fontsize=12)
        ax.set_xlabel("Number of Subspaces $N$", fontsize=12, fontweight="bold")
        ax.set_ylabel("$\\varepsilon_{\\max}^{(r)}(N)$", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95, loc="best")

    for j in range(len(dims), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        log(f"Saved: {output_path}")

    plt.close()


def plot_all_results(results: dict, output_dir: Path):
    """Generate all visualization plots with high-quality paper formatting."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "epsilon_scaling" in results:
        plot_epsilon_scaling(results, output_dir / "epsilon_scaling.pdf")

    if "line_covering_number" in results:
        plot_line_covering(results, output_dir / "line_covering.pdf")

    if "directional_richness" in results:
        plot_directional_richness(results, output_dir / "directional_richness.pdf")

    if "synthetic_feature_splitting" in results:
        plot_synthetic_feature_splitting(
            results, output_dir / "synthetic_feature_splitting_lines.pdf"
        )
        plot_synthetic_group_covering(
            results, output_dir / "synthetic_feature_splitting_subspaces.pdf"
        )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Theory Verification Experiments")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=[
            "epsilon_scaling",
            "line_covering",
            "directional_richness",
            "synthetic_feature_splitting",
            "all",
        ],
        help="Which experiment to run",
    )
    parser.add_argument("--d_model", type=int, default=64, help="Ambient dimension")
    parser.add_argument("--n_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_results/theory_verification",
        help="Output directory",
    )
    # Synthetic feature splitting experiment knobs
    parser.add_argument("--synth_feature_dims", type=str, default="2,4,8,16")
    parser.add_argument("--synth_n_train", type=int, default=10000)
    parser.add_argument("--synth_n_holdout", type=int, default=50000)
    parser.add_argument(
        "--synth_n_list_lines", type=str, default="1,2,4,8,16,32,64,128,256,512"
    )
    parser.add_argument(
        "--synth_n_list_subspaces", type=str, default="1,2,4,8,16,32,64,128"
    )
    parser.add_argument("--synth_r_list", type=str, default="1,2,4,8,16")
    parser.add_argument("--synth_line_restarts", type=int, default=5)
    parser.add_argument("--synth_ksub_iters", type=int, default=30)
    parser.add_argument("--synth_ksub_restarts", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments - each experiment is standalone and saves its own results
    if args.exp in ["epsilon_scaling", "all"]:
        exp_output_dir = (
            output_dir / "epsilon_scaling" if args.exp == "all" else output_dir
        )
        results = run_epsilon_scaling_experiment(
            d_model=args.d_model,
            n_epochs=args.n_epochs,
            output_dir=exp_output_dir,
            save_results=True,
        )
        all_results.update(results)

    if args.exp in ["line_covering", "all"]:
        exp_output_dir = (
            output_dir / "line_covering" if args.exp == "all" else output_dir
        )
        results = run_line_covering_experiment(
            d_model=args.d_model,
            output_dir=exp_output_dir,
            save_results=True,
        )
        all_results.update(results)

    if args.exp in ["directional_richness", "all"]:
        exp_output_dir = (
            output_dir / "directional_richness" if args.exp == "all" else output_dir
        )
        results = run_directional_richness_experiment(
            d_model=args.d_model,
            output_dir=exp_output_dir,
            save_results=True,
        )
        all_results.update(results)

    if args.exp in ["synthetic_feature_splitting", "all"]:
        exp_output_dir = (
            output_dir / "synthetic_feature_splitting"
            if args.exp == "all"
            else output_dir
        )
        results = run_synthetic_feature_splitting_validation(
            d_model=args.d_model,
            feature_dims=parse_int_list(args.synth_feature_dims),
            n_train=args.synth_n_train,
            n_holdout=args.synth_n_holdout,
            n_list_lines=parse_int_list(args.synth_n_list_lines),
            n_list_subspaces=parse_int_list(args.synth_n_list_subspaces),
            r_list=parse_int_list(args.synth_r_list),
            line_kmeans_restarts=args.synth_line_restarts,
            ksubspaces_iters=args.synth_ksub_iters,
            ksubspaces_restarts=args.synth_ksub_restarts,
            output_dir=exp_output_dir,
            save_results=True,
        )
        all_results.update(results)

    # If running all experiments, also save aggregated results
    if args.exp == "all" and all_results:
        results_path = output_dir / f"all_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(make_serializable(all_results), f, indent=2)
        log(f"\nSaved aggregated results to {results_path}")

        # Generate all plots in the main output directory
        plot_all_results(all_results, output_dir)

        # Print aggregated summary
        log("\n" + "=" * 70)
        log("AGGREGATED SUMMARY")
        log("=" * 70)

        if "epsilon_scaling" in all_results:
            log("\nEpsilon-Scaling:")
            for r in all_results["epsilon_scaling"]:
                d_i = r["d_i"]
                slope = r.get("empirical_slope", r.get("log_log_slope", 0))
                if slope == 0:
                    # Try negative log_log_slope
                    slope = -r.get("log_log_slope", 0)
                theory = d_i - 1
                log(f"  d_i={d_i}: measured slope = {slope:.2f}, theory = {theory}")


if __name__ == "__main__":
    main()
