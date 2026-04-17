from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
N_ACTIONS = 5

class QNetwork(nn.Module):
    def __init__(self, in_dim: int = OBS_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

_model: Optional[QNetwork] = None

def _load_once() -> None:
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"weights.pth not found at {wpath}")
    m = QNetwork()
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return ACTIONS[int(_model(x).argmax())]