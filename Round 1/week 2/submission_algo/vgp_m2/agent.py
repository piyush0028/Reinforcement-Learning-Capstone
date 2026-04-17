# agent.py
from __future__ import annotations
import os
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

_model: Optional[PolicyNetwork] = None

def _load_once():
    global _model
    if _model is not None:
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    m = PolicyNetwork()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    probs = _model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    return ACTIONS[int(probs.argmax())]