"""
BatchTopK ManifoldSAE: Uses BatchTopK sparsity on groups.

Reference: "BatchTopK Sparse Autoencoders" (arXiv:2412.06410)
"""

import math
from dataclasses import dataclass
from typing import Any

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import torch
import torch.nn as nn

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)


def _batch_topk_mask(
    group_norms: torch.Tensor, k_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute BatchTopK mask on group norms in float32 for stability.
    Returns the mask and the threshold used.
    """
    batch_size = group_norms.shape[0]
    flat_norms = group_norms.float().reshape(-1)
    k_total = k_groups * batch_size

    if k_total <= 0:
        # mask should be same dtype/device as group_norms
        return torch.zeros_like(group_norms), flat_norms.min()

    if k_total < flat_norms.numel():
        # IMPORTANT: avoid a >= threshold mask because bf16 quantization can
        # create many ties at the cutoff, leading to >k_total active groups.
        topk_vals, topk_idx = flat_norms.topk(k_total, largest=True, sorted=True)
        threshold = topk_vals[-1]
        mask_f32 = torch.zeros_like(flat_norms, dtype=torch.float32)
        mask_f32[topk_idx] = 1.0
        mask = mask_f32.view_as(group_norms).to(dtype=group_norms.dtype)
        return mask, threshold

    # If k_total >= all groups, everything is active.
    threshold = flat_norms.min()
    return torch.ones_like(group_norms), threshold


def _per_sample_topk_mask(group_norms: torch.Tensor, k_groups: int) -> torch.Tensor:
    """
    Per-sample TopK mask over groups (instance-wise).

    This is often preferable at inference/eval time (e.g., RAVEL) because it avoids
    coupling the sparsity pattern across different examples in the same batch.
    Returns a mask of shape [batch, n_groups] with exactly k_groups ones per sample
    (or fewer if k_groups > n_groups).
    """
    B, G = group_norms.shape
    k = min(max(int(k_groups), 0), int(G))
    if k == 0:
        return torch.zeros_like(group_norms)
    # compute in float32 for stability, scatter in float32, then cast back
    topk_idx = group_norms.float().topk(k, dim=-1, largest=True, sorted=False).indices
    mask_f32 = torch.zeros((B, G), device=group_norms.device, dtype=torch.float32)
    mask_f32.scatter_(1, topk_idx, 1.0)
    return mask_f32.to(dtype=group_norms.dtype)


@dataclass
class BatchTopKManifoldSAEConfig(TrainingSAEConfig):
    """Configuration for BatchTopK ManifoldSAE training."""

    n_groups: int = 1536
    group_rank: int = 16

    # BatchTopK parameters
    k_groups: int = 10
    k_aux: int = 256
    aux_coefficient: float = 1.0
    ortho_coefficient: float = 0.01
    trace_coefficient: float = 0.001

    rescale_by_decoder_norm: bool = False
    encoder_norm_renorm: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk_manifold_sae"

    def __post_init__(self):
        super().__post_init__()
        if self.d_sae != self.n_groups * self.group_rank:
            print(
                f"Warning: Setting d_sae to {self.n_groups * self.group_rank} to match n_groups * group_rank"
            )
            self.d_sae = self.n_groups * self.group_rank


class BatchTopKManifoldSAE(TrainingSAE[BatchTopKManifoldSAEConfig]):
    """
    ManifoldSAE with BatchTopK activation on groups.
    """

    cfg: BatchTopKManifoldSAEConfig
    b_enc: nn.Parameter

    def __init__(self, cfg: BatchTopKManifoldSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.n_groups = cfg.n_groups
        self.group_rank = cfg.group_rank
        self.k_groups = cfg.k_groups

        # Initialize b_enc
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and return both feature activations and pre-activations."""
        sae_in = self.process_sae_in(x)
        pre_acts = sae_in @ self.W_enc + self.b_enc

        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)  # [batch, n_groups]

        # BatchTopK on group norms
        mask, _ = _batch_topk_mask(group_norms, self.k_groups)

        # Apply mask to groups
        active_groups = groups * mask.unsqueeze(-1)  # [batch, n_groups, group_rank]
        feature_acts = active_groups.view(-1, self.cfg.d_sae)

        return feature_acts, pre_acts

    @override
    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Training forward pass with BatchTopK on groups."""
        # Normalize decoder weights to unit norm (for stable training)
        with torch.no_grad():
            W_dec_norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data = self.W_dec.data / W_dec_norms
            if self.cfg.encoder_norm_renorm:
                W_enc_norms = self.W_enc.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self.W_enc.data = self.W_enc.data / W_enc_norms

        x = step_input.sae_in

        sae_in = self.process_sae_in(x)
        pre_acts = sae_in @ self.W_enc + self.b_enc

        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)  # [batch, n_groups]

        # BatchTopK: select top K * batch_size groups based on group L2 norm
        # (Since decoder is unit-normalized, no need to rescale by decoder norm)
        mask, _ = _batch_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)  # [batch, n_groups, group_rank]
        feature_acts = active_groups.view(-1, self.cfg.d_sae)
        avg_active_groups = mask.sum(dim=-1).mean()
        group_coverage = (mask.sum(dim=0) > 0).float().mean()

        # Decode
        x_reconstruct = self.decode(feature_acts)

        # Losses
        mse_loss = ((x_reconstruct - x) ** 2).sum(dim=-1).mean()
        cos_sim = (x * x_reconstruct).sum(dim=-1) / (
            x.norm(dim=-1) * x_reconstruct.norm(dim=-1) + 1e-8
        )
        cos_loss = (1.0 - cos_sim).mean()

        # Auxiliary loss for dead group prevention
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,  # Use actual feature activations for proper tracking
            hidden_pre=pre_acts,
            sae_out=x_reconstruct,
        )

        # Decoder orthogonality: encourage atoms within group to be orthogonal
        W_dec_groups = self.W_dec.view(self.n_groups, self.group_rank, -1)
        gram = W_dec_groups @ W_dec_groups.transpose(-1, -2)
        eye = torch.eye(self.group_rank, device=gram.device, dtype=gram.dtype)
        ortho_loss = self.cfg.ortho_coefficient * ((gram - eye) ** 2).mean()

        # Trace regularization: penalize squared norm of active group activations
        trace_loss = self.cfg.trace_coefficient * (active_groups**2).sum(dim=-1).mean()

        total_loss = mse_loss + ortho_loss + trace_loss
        losses = {
            "mse_loss": mse_loss,
            "ortho_loss": ortho_loss,
            "trace_loss": trace_loss,
            "avg_active_groups": avg_active_groups,
        }

        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            losses["aux_loss"] = aux_losses
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=x,
            sae_out=x_reconstruct,
            feature_acts=feature_acts,  # Return actual feature activations for proper tracking
            hidden_pre=pre_acts,
            loss=total_loss,
            losses=losses,
            metrics={"cos_loss": cos_loss, "group_coverage": group_coverage},
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature activations using BatchTopK on groups."""
        original_shape = x.shape
        needs_flat = x.dim() == 3
        if needs_flat:
            x = x.reshape(-1, x.shape[-1])  # Flatten to [batch*seq, d_model]

        sae_in = self.process_sae_in(x)
        pre_acts = sae_in @ self.W_enc + self.b_enc

        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)  # [batch, n_groups]

        # BatchTopK on group norms
        mask, _ = _batch_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.cfg.d_sae)

        # Restore original batch/seq structure
        if needs_flat:
            feature_acts = feature_acts.view(original_shape[0], original_shape[1], -1)

        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations to reconstruction."""
        sae_out = feature_acts @ self.W_dec + self.b_dec
        if (
            self.cfg.normalize_activations == "layer_norm"
            and sae_out.dim() == 3
            and hasattr(self, "ln_mu")
            and hasattr(self, "ln_std")
        ):
            # Align stored LN stats with sequence shape for correct broadcasting
            sae_out_batch, sae_out_seq, _ = sae_out.shape
            self.ln_mu = self.ln_mu.view(sae_out_batch, sae_out_seq, -1)  # type: ignore[attr-defined]
            self.ln_std = self.ln_std.view(sae_out_batch, sae_out_seq, -1)  # type: ignore[attr-defined]
        sae_out = self.hook_sae_recons(sae_out)
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        return self.reshape_fn_out(sae_out, self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass. Preserves input shape."""
        feature_acts = self.encode(x)
        return self.decode(feature_acts)

    @override
    def get_inference_config_class(self) -> type:
        return BatchTopKManifoldSAEInferenceConfig

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Auxiliary loss focused on per-feature deadness (not just groups)."""
        dead_neuron_mask = step_input.dead_neuron_mask
        if dead_neuron_mask is None:
            return {"aux_loss": sae_out.new_tensor(0.0)}

        dead_features = dead_neuron_mask.view(-1)  # [d_sae] bool
        num_dead_features = int(dead_features.sum())
        if num_dead_features == 0:
            return {"aux_loss": sae_out.new_tensor(0.0)}

        x = step_input.sae_in
        residual = (x - sae_out).detach()  # [batch, d_in]

        # Pre-activations per feature [batch, d_sae]
        hidden_pre_flat = hidden_pre.view(-1, self.cfg.d_sae)

        # Limit k_aux to available dead features
        k_aux = min(self.cfg.k_aux, num_dead_features, self.cfg.d_in // 2)
        if k_aux == 0:
            return {"aux_loss": sae_out.new_tensor(0.0)}

        # Mask living features to -inf to avoid selecting them
        masked_pre = torch.where(
            dead_features[None, :],
            hidden_pre_flat,
            torch.full_like(hidden_pre_flat, -torch.inf),
        )

        # Select top-k dead features per sample
        _, topk_indices = masked_pre.topk(k_aux, dim=-1)

        # Build aux activations, keeping gradients through hidden_pre
        aux_feature_acts = torch.zeros_like(hidden_pre_flat)
        aux_feature_acts.scatter_(1, topk_indices, masked_pre.gather(1, topk_indices))

        # Decode aux activations into x-space
        aux_recon = aux_feature_acts @ self.W_dec + self.b_dec
        aux_recon = self.run_time_activation_norm_fn_out(aux_recon)

        aux_mse = ((aux_recon - residual) ** 2).sum(dim=-1).mean()
        scale = min(num_dead_features / k_aux, 1.0)
        aux_loss = self.cfg.aux_coefficient * scale * aux_mse

        return {"aux_loss": aux_loss}

    @override
    def get_coefficients(self) -> dict[str, float]:
        """Return auxiliary coefficient."""
        return {"aux": self.cfg.aux_coefficient}


# ============ Inference SAE for loading saved models ============


@dataclass
class BatchTopKManifoldSAEInferenceConfig(SAEConfig):
    """Configuration for BatchTopK ManifoldSAE inference."""

    n_groups: int = 3072
    group_rank: int = 16
    k_groups: int = 10

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk_manifold_sae"


class BatchTopKManifoldSAEInference(SAE[BatchTopKManifoldSAEInferenceConfig]):
    """
    BatchTopK ManifoldSAE for inference.
    """

    cfg: BatchTopKManifoldSAEInferenceConfig
    b_enc: nn.Parameter

    def __init__(
        self, cfg: BatchTopKManifoldSAEInferenceConfig, use_error_term: bool = False
    ):
        self.n_groups = cfg.n_groups
        self.group_rank = cfg.group_rank
        self.k_groups = cfg.k_groups
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using BatchTopK on groups."""
        original_shape = x.shape
        needs_flat = x.dim() == 3
        if needs_flat:
            batch, seq, _ = x.shape
            x = x.reshape(-1, x.shape[-1])

        sae_in = self.process_sae_in(x)
        if needs_flat and self.cfg.normalize_activations == "layer_norm":
            self.ln_mu = self.ln_mu.view(batch, seq, -1)  # type: ignore[attr-defined]
            self.ln_std = self.ln_std.view(batch, seq, -1)  # type: ignore[attr-defined]
        # Cast sae_in to match weight dtype (layer norm may produce float32)
        sae_in = sae_in.to(self.W_enc.dtype)
        pre_acts = sae_in @ self.W_enc + self.b_enc

        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)

        # IMPORTANT:
        # Use the SAME BatchTopK selection rule as training (global TopK across the batch).
        # This materially affects reconstruction quality for batchtopk-trained SAEs.
        # When `needs_flat=True`, the "batch" here is actually (batch * seq) tokens, which
        # matches how SAE-Lens commonly feeds activations during training.
        mask, _ = _batch_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.cfg.d_sae)

        if needs_flat:
            feature_acts = feature_acts.view(original_shape[0], original_shape[1], -1)

        return self.hook_sae_acts_post(feature_acts)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations."""
        sae_out = feature_acts @ self.W_dec + self.b_dec
        if (
            self.cfg.normalize_activations == "layer_norm"
            and sae_out.dim() == 3
            and hasattr(self, "ln_mu")
            and hasattr(self, "ln_std")
        ):
            sae_out_batch, sae_out_seq, _ = sae_out.shape
            self.ln_mu = self.ln_mu.view(sae_out_batch, sae_out_seq, -1)  # type: ignore[attr-defined]
            self.ln_std = self.ln_std.view(sae_out_batch, sae_out_seq, -1)  # type: ignore[attr-defined]
        sae_out = self.hook_sae_recons(sae_out)
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        return self.reshape_fn_out(sae_out, self.d_head)
