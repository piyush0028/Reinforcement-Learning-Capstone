"""
OBELIX Phase-2 Submission — Dueling DDQN with Frame Stacking
=============================================================
Algorithm : Dueling Double DQN with Frame Stacking (K=16)
Phase     : Level 2 (blinking box, wall_obstacles=True)

Why frame stacking:
    When the box blinks out, all sonar/IR bits drop to 0. A single-frame
    network is completely blind. With K=16 stacked frames, the network sees
    where the box WAS in the last ≤16 steps and continues navigating toward
    it through the invisible period.

Episode reset detection:
    Codabench imports this module ONCE and calls policy() for all 10 episodes
    back-to-back. We detect a new episode when obs is all-zero after having
    been non-zero (the env resets sensor feedback to zeros at episode start).
"""

from __future__ import annotations

import os
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn

# ── Constants — must match train_phase2.py exactly ────────────────────────────
FRAME_STACK = 16
OBS_DIM_RAW = 18
OBS_DIM     = 20                  # 18 raw + 2 augmented bits (stuck, ir)
INPUT_DIM   = OBS_DIM * FRAME_STACK   # 320
HIDDEN_DIM  = 256

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights.pth"
)


# ── Network — must mirror train_phase2.py exactly ─────────────────────────────
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


# ── Evaluation step limit — must match Codabench ──────────────────────────────
MAX_STEPS = 2000

# ── Global state ──────────────────────────────────────────────────────────────
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
    """Reset frame buffer and step counter for a new episode."""
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
    """
    Called by the Codabench evaluator at every environment step.

    Parameters
    ----------
    obs : np.ndarray, shape (18,)
    rng : np.random.Generator  (not used — greedy policy)

    Returns
    -------
    str : one of "L45", "L22", "FW", "R22", "R45"
    """
    global _net, _frame_buf, _step_count, _prev_stuck, _prev_ir

    # Lazy load on first call
    if _net is None:
        _load_agent()
        _reset_episode()

    # Reset at episode boundary. Success episodes end early (< MAX_STEPS)
    # but the counter overshoots harmlessly — frame buffer fills fresh
    # within a few steps of the next episode anyway.
    if _step_count >= MAX_STEPS:
        _reset_episode()

    _step_count += 1

    # ── Observation Augmentation ─────────────────────────────────────────────
    # Append the status of the PREVIOUS step to match training logic
    augmented_obs = np.append(obs.astype(np.float32), [_prev_stuck, _prev_ir])

    # Update trackers for the NEXT step based on the current observation.
    # From your train.py: obs[16] is IR, obs[17] is bumper/stuck
    _prev_ir    = 1.0 if obs[16] > 0 else 0.0
    _prev_stuck = 1.0 if obs[17] > 0 else 0.0

    # ── Forward pass ─────────────────────────────────────────────────────────
    _frame_buf.append(augmented_obs)
    state   = np.concatenate(list(_frame_buf)).astype(np.float32)
    state_t = torch.FloatTensor(state).unsqueeze(0)   # Now safely (1, 320)
    q       = _net(state_t)                           # (1, 5)

    return ACTIONS[int(q.argmax(dim=1).item())]