from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
FRAME_STACK = 32
OBS_DIM_RAW = 18
OBS_DIM = OBS_DIM_RAW + 2
STATE_DIM = OBS_DIM * FRAME_STACK
MAX_STEPS = 2000

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weightsp.pth"
)

SEARCH_PATIENCE = 15
BOUNCE_THRESHOLD = 4
SWEEP_FW_STEPS = 30
SWEEP_TURN_EVERY = 80

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
        self.value_stream = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

def augment_obs(raw_obs: np.ndarray, prev_reward: float, prev_ir: bool) -> np.ndarray:
    was_stuck = 1.0 if prev_reward <= -199.0 else 0.0
    was_ir_hit = 1.0 if prev_ir else 0.0
    return np.append(raw_obs.astype(np.float32), [was_stuck, was_ir_hit])

class SweepController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.fw_counter = 0

    def next_action(self) -> str:
        self.steps += 1
        if self.steps % SWEEP_TURN_EVERY == 0:
            return "L45" if (self.steps // SWEEP_TURN_EVERY) % 2 == 1 else "R45"
        self.fw_counter += 1
        if self.fw_counter < SWEEP_FW_STEPS:
            return "FW"
        self.fw_counter = 0
        return "L22"

_model: DuelingDDQN | None = None
_state: np.ndarray | None = None
_step_count: int = 0
_prev_reward: float = 0.0
_prev_ir: bool = False
_consec_zeros: int = 0
_consec_stuck: int = 0
_sweep: SweepController = SweepController()

def _load() -> None:
    global _model
    _model = DuelingDDQN()
    ckpt = torch.load(WEIGHTS_FILE, map_location="cpu")
    sd = ckpt.get("online_net", ckpt)
    _model.load_state_dict(sd)
    _model.eval()

def _reset() -> None:
    global _state, _step_count, _prev_reward, _prev_ir
    global _consec_zeros, _consec_stuck
    _state = None
    _step_count = 0
    _prev_reward = 0.0
    _prev_ir = False
    _consec_zeros = 0
    _consec_stuck = 0
    _sweep.reset()

def _net_action() -> str:
    x = torch.from_numpy(_state).unsqueeze(0)
    qs = _model(x).squeeze(0).numpy()
    return ACTIONS[int(np.argmax(qs))]

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _model, _state, _step_count, _prev_reward, _prev_ir
    global _consec_zeros, _consec_stuck

    if _model is None:
        _load()
        _reset()

    obs_arr = np.array(obs, dtype=np.float32)
    obs_sum = float(obs_arr.sum())

    _consec_zeros = _consec_zeros + 1 if obs_sum == 0.0 else 0

    if _step_count >= MAX_STEPS:
        _reset()
    elif _consec_zeros >= 200 and _step_count > 100:
        _reset()

    _step_count += 1

    any_sensor = bool(np.any(obs_arr[:17] > 0))
    stuck_now = bool(obs_arr[17] > 0)
    ir_active = bool(obs_arr[16] > 0)

    in_push = ir_active and any_sensor

    _prev_ir = ir_active

    aug = augment_obs(obs_arr, _prev_reward, _prev_ir)
    if _state is None:
        _state = np.tile(aug, FRAME_STACK).astype(np.float32)
    else:
        _state[:-OBS_DIM] = _state[OBS_DIM:]
        _state[-OBS_DIM:] = aug

    if stuck_now and not in_push:
        _consec_stuck += 1
    else:
        _consec_stuck = 0

    if _consec_stuck >= BOUNCE_THRESHOLD:
        _consec_stuck = 0
        _sweep.reset()
        _prev_reward = -200.0
        return "L45"

    if any_sensor or in_push:
        _sweep.reset()
        action = _net_action()
    elif _consec_zeros >= SEARCH_PATIENCE:
        action = _sweep.next_action()
    else:
        action = _net_action()

    _prev_reward = -200.0 if stuck_now else 0.0
    return action