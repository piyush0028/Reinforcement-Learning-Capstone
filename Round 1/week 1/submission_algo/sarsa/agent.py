# agent.py  (SARSA version)
from __future__ import annotations
import os
from collections import defaultdict
from typing import Optional

import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

_Q: Optional[dict] = None


def _load_once() -> None:
    global _Q
    if _Q is not None:
        return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_sarsa.npy")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"weights_sarsa.npy not found at {wpath}")
    # np.save saves a dict as a 0-d object array — load with allow_pickle
    _Q = np.load(wpath, allow_pickle=True).item()


def obs_to_state(obs: np.ndarray) -> int:
    bits   = np.array(obs, dtype=int)
    powers = 1 << np.arange(len(bits))
    return int(np.sum(bits * powers))


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    state  = obs_to_state(obs)
    q_vals = [_Q.get((state, a), 0.0) for a in range(N_ACTIONS)]
    return ACTIONS[int(np.argmax(q_vals))]