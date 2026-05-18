from dataclasses import dataclass
from typing_extensions import override

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


def _per_sample_topk_mask(group_norms: torch.Tensor, k_groups: int) -> torch.Tensor:
    B, G = group_norms.shape
    k = min(max(int(k_groups), 0), int(G))
    if k == 0:
        return torch.zeros_like(group_norms)
    topk_idx = group_norms.float().topk(k, dim=-1, largest=True, sorted=False).indices
    mask_f32 = torch.zeros((B, G), device=group_norms.device, dtype=torch.float32)
    mask_f32.scatter_(1, topk_idx, 1.0)
    return mask_f32.to(dtype=group_norms.dtype)


def _batched_nuclear_norm(matrix_batch: torch.Tensor) -> torch.Tensor:
    m = matrix_batch.shape[-2]
    n = matrix_batch.shape[-1]
    # Use the smaller Gram matrix to avoid a full SVD.
    if m <= n:
        gram = matrix_batch @ matrix_batch.transpose(-1, -2)
    else:
        gram = matrix_batch.transpose(-1, -2) @ matrix_batch
    if gram.dtype in (torch.float16, torch.bfloat16):
        gram = gram.float()
    eigvals = torch.linalg.eigvalsh(gram)
    return eigvals.clamp(min=0).sqrt().sum(dim=-1)


@dataclass
class TopKSASAConfig(TrainingSAEConfig):
    n_groups: int = 1536
    group_rank: int = 16
    k_groups: int = 10
    k_aux: int = 512
    aux_coefficient: float = 1.0
    nuclear_coefficient: float = 100
    rescale_by_decoder_norm: bool = False
    encoder_norm_renorm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk_sasa"

    def __post_init__(self):
        super().__post_init__()
        d_sae_expected = self.n_groups * self.group_rank
        if self.d_sae != d_sae_expected:
            print(
                f"Warning: Setting d_sae to {d_sae_expected} to match n_groups * group_rank"
            )
            self.d_sae = d_sae_expected


class TopKSASA(TrainingSAE[TopKSASAConfig]):
    cfg: TopKSASAConfig
    b_enc: nn.Parameter

    def __init__(self, cfg: TopKSASAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.n_groups = cfg.n_groups
        self.group_rank = cfg.group_rank
        self.k_groups = cfg.k_groups
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self._step = 0

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        pre_acts = sae_in @ self.W_enc + self.b_enc
        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)
        mask = _per_sample_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.cfg.d_sae)
        return feature_acts, pre_acts

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step += 1
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
        group_norms = groups.norm(dim=-1)
        mask = _per_sample_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.cfg.d_sae)

        avg_active_groups = mask.sum(dim=-1).mean()
        group_coverage = (mask.sum(dim=0) > 0).float().mean()

        x_reconstruct = self.decode(feature_acts)

        mse_loss = ((x_reconstruct - x) ** 2).sum(dim=-1).mean()
        cos_sim = (x * x_reconstruct).sum(dim=-1) / (
            x.norm(dim=-1) * x_reconstruct.norm(dim=-1) + 1e-8
        )
        cos_loss = (1.0 - cos_sim).mean()

        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=pre_acts,
            sae_out=x_reconstruct,
        )

        W_dec_groups = self.W_dec.view(self.n_groups, self.group_rank, -1)
        if self._step % 1 == 0:
            nuclear_loss = (
                self.cfg.nuclear_coefficient
                * _batched_nuclear_norm(W_dec_groups).mean()
            )
        else:
            nuclear_loss = mse_loss.new_zeros(())

        total_loss = mse_loss + nuclear_loss
        losses: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss,
            "nuclear_loss": nuclear_loss,
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
            feature_acts=feature_acts,
            hidden_pre=pre_acts,
            loss=total_loss,
            losses=losses,
            metrics={"cos_loss": cos_loss, "group_coverage": group_coverage},
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        needs_flat = x.dim() == 3
        if needs_flat:
            x = x.reshape(-1, x.shape[-1])

        sae_in = self.process_sae_in(x)
        pre_acts = sae_in @ self.W_enc + self.b_enc

        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)
        mask = _per_sample_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.cfg.d_sae)

        if needs_flat:
            feature_acts = feature_acts.view(original_shape[0], original_shape[1], -1)
        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_acts = self.encode(x)
        return self.decode(feature_acts)

    @override
    def get_inference_config_class(self) -> type:
        return TopKSASAInferenceConfig

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dead_neuron_mask = step_input.dead_neuron_mask
        if dead_neuron_mask is None:
            return {"aux_loss": sae_out.new_tensor(0.0)}

        dead_features = dead_neuron_mask.view(-1)
        num_dead_features = int(dead_features.sum())
        if num_dead_features == 0:
            return {"aux_loss": sae_out.new_tensor(0.0)}

        x = step_input.sae_in
        residual = (x - sae_out).detach()
        hidden_pre_flat = hidden_pre.view(-1, self.cfg.d_sae)

        k_aux = min(self.cfg.k_aux, num_dead_features, self.cfg.d_in // 2)
        if k_aux == 0:
            return {"aux_loss": sae_out.new_tensor(0.0)}

        masked_pre = torch.where(
            dead_features[None, :],
            hidden_pre_flat,
            torch.full_like(hidden_pre_flat, -torch.inf),
        )

        _, topk_indices = masked_pre.topk(k_aux, dim=-1)

        aux_feature_acts = torch.zeros_like(hidden_pre_flat)
        aux_feature_acts.scatter_(1, topk_indices, masked_pre.gather(1, topk_indices))

        aux_recon = aux_feature_acts @ self.W_dec + self.b_dec
        aux_recon = self.run_time_activation_norm_fn_out(aux_recon)

        aux_mse = ((aux_recon - residual) ** 2).sum(dim=-1).mean()
        scale = min(num_dead_features / k_aux, 1.0)
        aux_loss = self.cfg.aux_coefficient * scale * aux_mse
        return {"aux_loss": aux_loss}

    @override
    def get_coefficients(self) -> dict[str, float]:
        return {"aux": self.cfg.aux_coefficient}


@dataclass
class TopKSASAInferenceConfig(SAEConfig):
    n_groups: int = 3072
    group_rank: int = 8
    k_groups: int = 10

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk_sasa"


class TopKSASAInference(SAE[TopKSASAInferenceConfig]):
    cfg: TopKSASAInferenceConfig
    b_enc: nn.Parameter

    def __init__(self, cfg: TopKSASAInferenceConfig, use_error_term: bool = False):
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
        original_shape = x.shape
        needs_flat = x.dim() == 3
        if needs_flat:
            batch, seq, _ = x.shape
            x = x.reshape(-1, x.shape[-1])

        sae_in = self.process_sae_in(x)
        if needs_flat and self.cfg.normalize_activations == "layer_norm":
            self.ln_mu = self.ln_mu.view(batch, seq, -1)  # type: ignore[attr-defined]
            self.ln_std = self.ln_std.view(batch, seq, -1)  # type: ignore[attr-defined]
        pre_acts = sae_in @ self.W_enc + self.b_enc

        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)
        mask = _per_sample_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.cfg.d_sae)

        if needs_flat:
            feature_acts = feature_acts.view(original_shape[0], original_shape[1], -1)

        return self.hook_sae_acts_post(feature_acts)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
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
