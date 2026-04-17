from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS     = ["L45", "L22", "FW", "R22", "R45"]
FRAME_STACK = 32
OBS_DIM_RAW = 18
OBS_DIM     = OBS_DIM_RAW + 2          
STATE_DIM   = OBS_DIM * FRAME_STACK    
MAX_STEPS   = 2000

ZERO_RESET_THRESHOLD = 200

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights.pth"
)


class DuelingDDQN(nn.Module):
    def __init__(self, input_dim=STATE_DIM, n_actions=5, hidden_dim=256):
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


def augment_obs(raw_obs: np.ndarray,
                prev_reward: float,
                prev_ir: bool) -> np.ndarray:
    
    was_stuck  = 1.0 if prev_reward <= -199.0 else 0.0
    was_ir_hit = 1.0 if prev_ir else 0.0
    return np.append(raw_obs.astype(np.float32), [was_stuck, was_ir_hit])


_model:        DuelingDDQN | None = None
_state:        np.ndarray | None  = None
_step_count:   int                = 0
_prev_reward:  float              = 0.0
_prev_ir:      bool               = False
_consec_zeros: int                = 0


def _load() -> None:
    global _model
    _model = DuelingDDQN()
    ckpt = torch.load(WEIGHTS_FILE, map_location="cpu")
    sd   = ckpt.get("online_net", ckpt)
    _model.load_state_dict(sd)
    _model.eval()


def _reset() -> None:
    global _state, _step_count, _prev_reward, _prev_ir, _consec_zeros
    _state        = None
    _step_count   = 0
    _prev_reward  = 0.0
    _prev_ir      = False
    _consec_zeros = 0


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _model, _state, _step_count, _prev_reward, _prev_ir, _consec_zeros

    if _model is None:
        _load()
        _reset()

    obs_arr = np.array(obs, dtype=np.float32)
    obs_sum = float(obs_arr.sum())

    if obs_sum == 0.0:
        _consec_zeros += 1
    else:
        _consec_zeros = 0

    if _step_count >= MAX_STEPS:
        _reset()

    elif _consec_zeros >= ZERO_RESET_THRESHOLD and _step_count > 100:
        _reset()

    _step_count += 1


    _prev_ir = bool(obs[16] > 0)

    aug = augment_obs(obs_arr, _prev_reward, _prev_ir)

    if _state is None:
        _state = np.tile(aug, FRAME_STACK).astype(np.float32)
    else:
        _state[:-OBS_DIM] = _state[OBS_DIM:]
        _state[-OBS_DIM:] = aug

    x      = torch.from_numpy(_state).unsqueeze(0)
    qs     = _model(x).squeeze(0).numpy()
    action = ACTIONS[int(np.argmax(qs))]

    _prev_reward = -200.0 if bool(obs[17] > 0) else 0.0

    return action