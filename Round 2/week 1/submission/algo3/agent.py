from __future__ import annotations

import os
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn

FRAME_STACK = 16
OBS_DIM_RAW = 18
OBS_DIM     = 20
INPUT_DIM   = OBS_DIM * FRAME_STACK
HIDDEN_DIM  = 256

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights.pth"
)

class DuelingDDQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        h = hidden_dim // 2
        self.value_stream     = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

MAX_STEPS = 1000

_net:        DuelingDDQN | None = None
_frame_buf:  deque | None       = None
_step_count: int                = 0
_prev_stuck: float              = 0.0
_prev_ir:    float              = 0.0

def _load_agent() -> None:
    global _net
    _net = DuelingDDQN(INPUT_DIM, N_ACTIONS, HIDDEN_DIM)
    ckpt = torch.load(WEIGHTS_FILE, map_location="cpu")
    _net.load_state_dict(ckpt["online_net"])
    _net.eval()

def _reset_episode() -> None:
    global _frame_buf, _step_count, _prev_stuck, _prev_ir
    _frame_buf = deque(
        [np.zeros(OBS_DIM, dtype=np.float32)] * FRAME_STACK,
        maxlen=FRAME_STACK,
    )
    _step_count = 0
    _prev_stuck = 0.0
    _prev_ir    = 0.0

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _net, _frame_buf, _step_count, _prev_stuck, _prev_ir

    if _net is None:
        _load_agent()
        _reset_episode()

    if _step_count >= MAX_STEPS:
        _reset_episode()

    _step_count += 1

    augmented_obs = np.append(obs.astype(np.float32), [_prev_stuck, _prev_ir])

    _prev_ir    = 1.0 if obs[16] > 0 else 0.0
    _prev_stuck = 1.0 if obs[17] > 0 else 0.0

    _frame_buf.append(augmented_obs)
    state   = np.concatenate(list(_frame_buf)).astype(np.float32)
    state_t = torch.FloatTensor(state).unsqueeze(0)   
    q       = _net(state_t)                           

    return ACTIONS[int(q.argmax(dim=1).item())]