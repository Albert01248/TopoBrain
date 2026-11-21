"""Lightweight frozen-codebook spatial/temporal adaptation, not a new foundation model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    num_classes: int
    domain: str = "both"
    hidden_per_branch: int = 32
    graph_depth: int = 1
    temporal_window: int = 3
    dropout: float = 0.1

    def __post_init__(self):
        if self.num_classes < 2:
            raise ValueError("num_classes must be >=2")
        if self.domain not in {"both", "temporal", "frequency"}:
            raise ValueError("domain must be both, temporal, or frequency")
        if self.hidden_per_branch < 1 or self.graph_depth not in {0, 1, 2}:
            raise ValueError("hidden_per_branch must be positive; graph_depth must be 0, 1, or 2")
        if self.temporal_window < 1 or self.temporal_window % 2 != 1:
            raise ValueError("temporal_window must be a positive odd number of patches")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class ResidualSpatialLayer(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, adjacency, observed):
        # B,C,P,D; adjacency is already per-sample normalized after missingness.
        message = torch.einsum("bij,bjpd->bipd", adjacency, self.norm(features))
        output = features + self.dropout(F.gelu(self.linear(message)))
        return output.masked_fill(~observed[:, :, None, None], 0)


class LocalTemporalLayer(nn.Module):
    def __init__(self, width, window, dropout):
        super().__init__()
        heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
        self.window = window
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, observed):
        batch, channels, patches, width = features.shape
        sequence = features.reshape(batch * channels, patches, width)
        positions = torch.arange(patches, device=features.device)
        blocked = (positions[:, None] - positions[None, :]).abs() > self.window // 2
        normed = self.norm_attention(sequence)
        # Each channel is a separate batch item. Missing channels are processed
        # as zero input then cleared; all-masked attention rows would create NaN.
        attended, _ = self.attention(normed, normed, normed, attn_mask=blocked, need_weights=False)
        sequence = sequence + self.dropout(attended)
        sequence = sequence + self.dropout(self.ffn(self.norm_ffn(sequence)))
        output = sequence.reshape(batch, channels, patches, width)
        return output.masked_fill(~observed[:, :, None, None], 0)


class TopoBrain(nn.Module):
    def __init__(self, codebook_t, codebook_f, adjacency, config: ModelConfig):
        super().__init__()
        self.config = config
        for name, codebook in (("codebook_t", codebook_t), ("codebook_f", codebook_f)):
            tensor = torch.as_tensor(codebook, dtype=torch.float32).detach().clone()
            if tensor.ndim != 2 or min(tensor.shape) < 1 or not torch.isfinite(tensor).all():
                raise ValueError("codebooks must be finite nonempty [K,D]")
            self.register_buffer(name, tensor)
        a = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
            raise ValueError("adjacency must be nonempty square [C,C]")
        if not torch.isfinite(a).all() or torch.any(a < 0) or not torch.allclose(a, a.T):
            raise ValueError("adjacency must be finite, nonnegative, and symmetric")
        # A=I and an all-zero adjacency both mean no inter-channel edges.
        # Self loops are added later only for observed nodes.
        a.fill_diagonal_(0)
        self.register_buffer("adjacency", a)
        domains = ("temporal", "frequency") if config.domain == "both" else (config.domain,)
        self.domains = domains
        total_width = 2 * config.hidden_per_branch
        branch_width = config.hidden_per_branch if config.domain == "both" else total_width
        dimensions = {"temporal": self.codebook_t.shape[1], "frequency": self.codebook_f.shape[1]}
        self.projections = nn.ModuleDict({key: nn.Linear(dimensions[key], branch_width) for key in domains})
        self.spatial = nn.ModuleDict({key: nn.ModuleList([
            ResidualSpatialLayer(branch_width, config.dropout) for _ in range(config.graph_depth)
        ]) for key in domains})
        self.temporal = LocalTemporalLayer(total_width, config.temporal_window, config.dropout)
        self.readout_norm = nn.LayerNorm(total_width)
        self.classifier = nn.Linear(total_width, config.num_classes)

    def _validate_inputs(self, codes_t, codes_f, valid_mask):
        if codes_t.ndim != 3 or codes_f.shape != codes_t.shape or min(codes_t.shape) < 1:
            raise ValueError("codes_t/f must share nonempty [B,C,P] shape")
        if codes_t.dtype != torch.long or codes_f.dtype != torch.long:
            raise ValueError("code tensors must use torch.long")
        if codes_t.device != self.adjacency.device or codes_f.device != codes_t.device:
            raise ValueError("codes and model must be on the same device")
        if codes_t.shape[1] != self.adjacency.shape[0]:
            raise ValueError("code channel count differs from adjacency")
        for codes, book in ((codes_t, self.codebook_t), (codes_f, self.codebook_f)):
            if torch.any(codes < 0) or torch.any(codes >= len(book)):
                raise ValueError("code ID outside the supplied codebook")
        if valid_mask is None:
            valid_mask = torch.ones(codes_t.shape[:2], dtype=torch.bool, device=codes_t.device)
        elif valid_mask.shape != codes_t.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,C]")
        elif valid_mask.device != codes_t.device:
            raise ValueError("valid_mask and code tensors must share device")
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError("every sample needs at least one observed channel; all-missing input is invalid")
        return valid_mask

    def _normalized_adjacency(self, observed):
        observed_float = observed.to(self.adjacency.dtype)
        a = self.adjacency[None] * observed_float[:, :, None] * observed_float[:, None, :]
        a = a + torch.diag_embed(observed_float)
        degrees = a.sum(dim=-1)
        inv_sqrt = torch.where(degrees > 0, degrees.clamp_min(1).rsqrt(), torch.zeros_like(degrees))
        return a * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]

    def encode(self, codes_t, codes_f, valid_mask=None):
        """Return [B,C,P,2*hidden_per_branch] features for controlled diagnostics."""
        observed = self._validate_inputs(codes_t, codes_f, valid_mask)
        normalized_a = self._normalized_adjacency(observed)
        inputs = {"temporal": (codes_t, self.codebook_t), "frequency": (codes_f, self.codebook_f)}
        branches = []
        for key in self.domains:
            codes, book = inputs[key]
            h = self.projections[key](F.embedding(codes, book))
            h = h.masked_fill(~observed[:, :, None, None], 0)
            for layer in self.spatial[key]:
                h = layer(h, normalized_a, observed)
            branches.append(h)
        combined = torch.cat(branches, dim=-1) if len(branches) == 2 else branches[0]
        return self.temporal(combined, observed)

    def forward(self, codes_t, codes_f, valid_mask=None, *, valid_channel_mask=None):
        if valid_channel_mask is not None:
            if valid_mask is not None:
                raise ValueError("pass valid_mask or valid_channel_mask, not both")
            valid_mask = valid_channel_mask
        observed = self._validate_inputs(codes_t, codes_f, valid_mask)
        features = self.encode(codes_t, codes_f, observed)
        features = self.readout_norm(features).masked_fill(~observed[:, :, None, None], 0)
        denominator = observed.sum(dim=1, keepdim=True).to(features.dtype) * features.shape[2]
        pooled = features.sum(dim=(1, 2)) / denominator
        return self.classifier(pooled)

    def parameter_report(self):
        return {"trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "all_parameters": sum(p.numel() for p in self.parameters()),
                "frozen_codebook_elements": self.codebook_t.numel() + self.codebook_f.numel(),
                "domain": self.config.domain,
                "note": "Single-domain heads keep total output width, but parameter counts are not exactly equal; report actual counts."}
