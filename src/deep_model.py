"""Feature-token mixture-of-experts for calibrated control probability."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ControlMoE(nn.Module):
    """Three subtype-supervised failure experts mixed by a context gate."""

    expert_names = ("middle_risk", "wild_risk", "reverse_risk")

    def __init__(
        self, n_numeric: int, cardinalities: list[int], width: int = 192,
        depth: int = 3, dropout: float = 0.12, expert_prior=(1.0, 1.0, 1.0),
        numeric_importance: list[float] | None = None,
    ):
        super().__init__()
        self.n_numeric = n_numeric
        self.cardinalities = cardinalities
        initial_importance = numeric_importance or [1.0] * n_numeric
        self.numeric_log_gate = nn.Parameter(torch.tensor(initial_importance).clamp_min(0.1).log())
        emb_dim = 16
        self.embeddings = nn.ModuleList([nn.Embedding(n, emb_dim) for n in cardinalities])
        input_dim = n_numeric + emb_dim * len(cardinalities)
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.backbone = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width // 2, 1))
            for _ in range(3)
        ])
        self.gate = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 3))
        self.register_buffer("expert_prior", torch.tensor(expert_prior, dtype=torch.float32))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> dict[str, torch.Tensor]:
        cat = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        # Positive learned gates expose which normalized numeric inputs matter.
        gated_numeric = numeric * self.numeric_log_gate.exp().clamp(max=5.0)
        h = self.backbone(self.stem(torch.cat([gated_numeric, *cat], dim=1)))
        expert_risk = torch.sigmoid(torch.cat([head(h) for head in self.experts], dim=1))
        gate_logits = self.gate(h) + self.expert_prior.clamp_min(1e-6).log()
        gate = torch.softmax(gate_logits, dim=1)
        # A noisy-OR supports overlapping failure causes. The context gate gives
        # each cause a tunable row-wise importance; x3 makes a uniform gate neutral.
        contribution = (gate * 3.0 * expert_risk).clamp(0.0, 1.0 - 1e-6)
        failure = 1.0 - torch.prod(1.0 - contribution, dim=1)
        success = (1.0 - failure).clamp(1e-6, 1.0 - 1e-6)
        return {"success": success, "failure": failure, "expert_risk": expert_risk,
                "gate": gate, "contribution": contribution}
