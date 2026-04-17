"""
train.py — OBELIX Phase 2 Training
====================================
Algorithm : Dueling Double DQN with Frame Stacking (K=16)
Phase     : Level 2 (blinking box, wall_obstacles=True)

Usage:
    python train.py
    python train.py --stage1_eps 1000 --stage2_eps 2000

Output:
    weights_stage1.pth   checkpoint after Stage 1
    weights_phase2.pth   final submission weights (submit this)
"""

import argparse
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIG — must match Codabench exactly
# ═══════════════════════════════════════════════════════════════════════════════
SCALING_FACTOR = 5
ARENA_SIZE     = 500
WALL_OBSTACLES = True
MAX_STEPS      = 1000


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME STACKING + OBSERVATION AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
FRAME_STACK = 16
OBS_DIM_RAW = 18    # raw environment observation
# Two extra bits appended to each frame before stacking:
#   bit 18: was_stuck   — reward was -200 last step (sensor hit a wall)
#   bit 19: was_ir_hit  — IR sensor fired last step (confirmed box contact)
# This gives the network the reward-signal disambiguation you identified:
#   sensors=1 + was_stuck=1   → that was a wall, don't approach
#   sensors=1 + was_stuck=0   → likely the box, keep approaching
#   ir_hit=1                  → confirmed box, commit to attachment
OBS_DIM   = OBS_DIM_RAW + 2   # 20
INPUT_DIM = OBS_DIM * FRAME_STACK  # 320


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
HIDDEN_DIM = 256


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
GAMMA              = 0.99
LR                 = 3e-4
BATCH_SIZE         = 128
REPLAY_BUFFER_SIZE = 150_000
TARGET_UPDATE_FREQ = 1000
# Fill buffer with 5 full episodes of random experience before learning.
# Previously 2000 steps (~2 eps) — too little. With ε=1.0 for the first
# ~30 eps now, we want at least 5 eps of random transitions to seed the buffer
# with diverse (obs, action, reward) tuples before the network starts learning.
MIN_REPLAY_SIZE    = 5000

EPSILON_START = 1.0
EPSILON_END   = 0.05
# Decay PER EPISODE, not per step.
# 0.9985^2000 ≈ 0.05 — reaches minimum right at end of Stage 1.
# Previous value (0.9995 per STEP, ~1000 steps/ep) collapsed to 0.05
# in only ~6 episodes, making the agent greedy before learning anything.
EPSILON_DECAY = 0.9985

STAGE1_EPISODES = 2000
STAGE2_EPISODES = 3000

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

STAGE1_SAVE = "weights_stage1.pth"
STAGE2_SAVE = "weights_phase2.pth"


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD SHAPING  (training only — env reward is never modified)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Problem: agent found a local optimum — spin in place, take -1/step,
# never risk -200 wall penalties. Expected return ≈ -1000. Rational but useless.
#
# Fix 1 — Distance progress reward:
#   Each step, reward = env_reward + α * (prev_dist - curr_dist)
#   Getting closer to box  → positive shaping bonus
#   Getting farther away   → negative shaping penalty
#   This makes any approach strictly better than spinning, regardless of walls.
#   α=0.5: closing 10px per step adds +5 shaping, comparable to IR bonus (+5).
#
# Fix 2 — Anti-oscillation penalty:
#   Track last OSCILLATION_WINDOW bot positions. If the agent hasn't moved
#   more than OSCILLATION_THRESHOLD pixels from its mean position, add penalty.
#   This directly kills spin-in-place without touching any other behaviour.
#
# Both are potential-based (Ng et al., 1999) — they accelerate learning
# without changing the optimal policy. The agent that maximises shaped reward
# is the same agent that would maximise env reward given enough exploration.

DIST_SHAPING_ALPHA   = 0.5   # reward per pixel of progress toward box
PUSH_SHAPING_ALPHA   = 1.0   # stronger shaping during push phase

OSCILLATION_WINDOW    = 30   # steps to look back
OSCILLATION_THRESHOLD = 15   # px — if std(positions) < this, agent is spinning
OSCILLATION_PENALTY   = -2.0 # added per step when oscillation detected


# ═══════════════════════════════════════════════════════════════════════════════
# DUELING DDQN NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
class DuelingDDQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 256):
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
        self.value_stream = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME STACK
# ═══════════════════════════════════════════════════════════════════════════════
class FrameStack:
    def __init__(self, k: int, obs_dim: int):
        self.k       = k
        self.obs_dim = obs_dim
        self.frames: deque = deque(maxlen=k)
        self.reset()

    def reset(self, first_obs: np.ndarray | None = None) -> np.ndarray:
        obs = (
            np.zeros(self.obs_dim, dtype=np.float32)
            if first_obs is None
            else first_obs.astype(np.float32)
        )
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self.get()

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.astype(np.float32))
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
def augment_obs(raw_obs: np.ndarray, prev_reward: float, prev_ir: bool) -> np.ndarray:
    """
    Append 2 reward-signal bits to the raw 18-dim observation → 20-dim.

    bit 18 — was_stuck  : 1.0 if previous reward was -200 (hit a wall)
    bit 19 — was_ir_hit : 1.0 if IR sensor fired on the previous step

    Why this solves the wall/box ambiguity:
        The raw sensors fire identically for walls and boxes. The only
        ground truth is the reward:
            sensors=1, was_stuck=0 → probably the box, safe to approach
            sensors=1, was_stuck=1 → wall, turn away
        Without these bits the network sees identical observations for
        "box ahead" and "wall ahead" and cannot learn different responses.

    In agent.py at inference these bits are computed the same way —
    was_stuck from whether last reward was -200, was_ir_hit from last obs[16].
    """
    was_stuck  = 1.0 if prev_reward <= -199.0 else 0.0
    was_ir_hit = 1.0 if prev_ir else 0.0
    return np.append(raw_obs.astype(np.float32), [was_stuck, was_ir_hit])


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD SHAPING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
class OscillationTracker:
    """
    Detects spin-in-place behaviour by checking whether the bot has covered
    meaningful ground over the last OSCILLATION_WINDOW steps.

    Mechanism:
        Keep a rolling deque of (x, y) positions. At each step, compute the
        std of x-coords and y-coords. If both are below OSCILLATION_THRESHOLD,
        the bot is oscillating in a small patch — apply the penalty.

    Why std and not displacement:
        Displacement (current - first) misses oscillation that returns to
        start. Std catches any motion that stays in a small region.
    """

    def __init__(self, window: int = OSCILLATION_WINDOW,
                 threshold: float = OSCILLATION_THRESHOLD):
        self.window    = window
        self.threshold = threshold
        self.xs: deque = deque(maxlen=window)
        self.ys: deque = deque(maxlen=window)

    def reset(self):
        self.xs.clear()
        self.ys.clear()

    def update(self, x: float, y: float) -> bool:
        """Add position, return True if oscillation detected."""
        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) < self.window:
            return False  # not enough history yet
        return (float(np.std(self.xs)) < self.threshold and
                float(np.std(self.ys)) < self.threshold)


def shape_reward(env_reward: float, env: OBELIX,
                 prev_dist: float, osc_detected: bool,
                 box_visible: bool) -> tuple[float, float]:
    """
    Apply training-only reward shaping on top of the environment reward.
    Returns (shaped_reward, curr_dist) so caller can use curr_dist as
    prev_dist on the next step.

    Shaping terms:
    1. Distance progress:  α * (prev_dist - curr_dist)
       Positive when moving toward box, negative when moving away.
       Uses push-phase alpha (stronger) once attached.
       Works even during blink-off — env always tracks true box position.

    2. Anti-oscillation:   OSCILLATION_PENALTY if spinning detected.
       Skipped when box is invisible (Level 2 blink-off) because holding
       a heading during a blink looks identical to spinning — same low
       positional variance — but is actually the correct behaviour.
    """
    dx       = float(env.bot_center_x - env.box_center_x)
    dy       = float(env.bot_center_y - env.box_center_y)
    curr_dist = float(np.sqrt(dx*dx + dy*dy))

    alpha    = PUSH_SHAPING_ALPHA if env.enable_push else DIST_SHAPING_ALPHA
    progress = prev_dist - curr_dist   # positive = got closer

    shaped = env_reward + alpha * progress

    # Only penalise oscillation when box is visible — during blink-off the
    # agent SHOULD hold its heading (low positional variance = correct).
    if osc_detected and box_visible:
        shaped += OSCILLATION_PENALTY

    return shaped, curr_dist


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.stack(ns),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
class EpisodeDiagnostics:
    """
    Tracks the 4-phase funnel per episode so you can pinpoint exactly
    where the agent is breaking down.

    Funnel phases:
        SEARCH  → did any sensor ever fire?         (sensor_fired)
        APPROACH→ did the IR sensor ever fire?       (ir_fired)
        ATTACH  → did the robot ever attach to box?  (attached)
        PUSH    → did the box reach the boundary?    (success)

    Reading the 100-ep summary:
    ┌────────────────────────────────────────────────────────────────────┐ 
    │ sensor ≈ 0%              → SEARCH failure. Agent is wandering      │
    │                            blind, never gets close enough.         │
    │ sensor high, IR ≈ 0%    → APPROACH failure. Detects box far away   │
    │                            but can't navigate to IR range.         │
    │ IR high, attach ≈ 0%    → ALIGN failure. Reaches IR range but      │
    │                            can't make contact (keeps overshooting).│
    │ attach high, success ≈ 0→ PUSH failure. Attached but wall blocks   │
    │                            the path to boundary.                   │
    │ stuck > 50 steps/ep     → Wall collisions are a dominant cost.     │
    │ blind > 400 steps/ep    → Too much time with zero sensor signal.   │
    │   (Stage 2 only — some blind steps are expected from blinking)     │
    └────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_steps       = 0
        self.cumulative_reward = 0.0

        # ── Phase funnel — uses actual box distance to split wall vs box hits ──
        # sensor_fired     : any sensor fired AND robot was within sonar range of box
        # wall_sensor_fired: sensor fired but robot was NOT near box (hit a wall)
        self.sensor_fired      = False   # confirmed box detection
        self.wall_sensor_fired = False   # sensor fired but it was a wall
        self.ir_fired          = False
        self.attached          = False
        self.success           = False

        self.step_at_sensor    = None
        self.step_at_ir        = None
        self.step_at_attach    = None

        # Per-step counters
        self.blind_steps      = 0
        self.stuck_steps      = 0
        self.push_stuck_steps = 0
        self.push_steps       = 0

        # Action distribution: L45=0  L22=1  FW=2  R22=3  R45=4
        self.action_counts = [0] * 5

        # Reward decomposition
        self.stuck_penalty_total = 0.0
        self.step_penalty_total  = 0.0
        self.sensor_bonus_total  = 0.0
        self.success_bonus       = 0.0

        # Q-value spread
        self.q_spreads: list = []

    def update(self, obs: np.ndarray, reward: float, env: OBELIX,
               action_idx: int = -1, q_values: np.ndarray | None = None):
        self.total_steps       += 1
        self.cumulative_reward += reward

        # Use raw obs bits (first 18) even if obs is augmented (20-dim)
        raw = obs[:18]
        sonar_active = bool(np.any(raw[:16] > 0))
        ir_active    = bool(raw[16] > 0)
        stuck        = bool(raw[17] > 0) and not env.enable_push
        push_stuck   = bool(raw[17] > 0) and env.enable_push
        in_push      = env.enable_push

        # ── Wall vs box disambiguation using actual box distance ───────────────
        # During training we have oracle access to the real box position.
        # sonar_far_range = 30 * scaling_factor. If robot is within that range
        # of the box AND sensors fired, it's a genuine box detection.
        # If sensors fired but robot is far from box → it was a wall hit.
        dx = env.bot_center_x - env.box_center_x
        dy = env.bot_center_y - env.box_center_y
        dist_to_box  = float(np.sqrt(dx*dx + dy*dy))
        sonar_range  = 30 * env.scaling_factor   # same formula as env
        near_box     = dist_to_box < (sonar_range + env.bot_radius)

        # ── Phase detection ───────────────────────────────────────────────────
        if (sonar_active or ir_active):
            if near_box and not self.sensor_fired:
                self.sensor_fired   = True
                self.step_at_sensor = self.total_steps
            elif not near_box and not self.wall_sensor_fired:
                self.wall_sensor_fired = True   # first wall false-positive

        if ir_active and not self.ir_fired:
            self.ir_fired   = True
            self.step_at_ir = self.total_steps

        if in_push and not self.attached:
            self.attached       = True
            self.step_at_attach = self.total_steps

        if reward >= 1999.0:
            self.success = True

        # ── Step counters ─────────────────────────────────────────────────────
        if not sonar_active and not ir_active:
            self.blind_steps += 1
        if stuck:
            self.stuck_steps += 1
        if push_stuck:
            self.push_stuck_steps += 1
        if in_push:
            self.push_steps += 1

        # ── Action distribution ───────────────────────────────────────────────
        if 0 <= action_idx < 5:
            self.action_counts[action_idx] += 1

        # ── Reward decomposition ──────────────────────────────────────────────
        if stuck or push_stuck:
            self.stuck_penalty_total += reward
        elif reward >= 1999.0:
            self.success_bonus        = reward
        elif reward > 0:
            self.sensor_bonus_total  += reward
        else:
            self.step_penalty_total  += reward

        # ── Q-value spread ────────────────────────────────────────────────────
        if q_values is not None and len(q_values) == 5:
            self.q_spreads.append(float(np.max(q_values) - np.min(q_values)))


class WindowStats:
    """Accumulates EpisodeDiagnostics over a rolling window."""

    def __init__(self, window: int = 100):
        self.window  = window
        self.history: deque = deque(maxlen=window)

    def add(self, d: EpisodeDiagnostics):
        self.history.append(d)

    def report(self) -> dict:
        n = len(self.history)
        if n == 0:
            return {}

        def pct(attr):
            return 100.0 * sum(getattr(d, attr) for d in self.history) / n

        def avg(attr):
            return float(np.mean([getattr(d, attr) for d in self.history]))

        def avg_if(val_attr, gate_attr):
            vals = [
                getattr(d, val_attr)
                for d in self.history
                if getattr(d, gate_attr) is not None
            ]
            return float(np.mean(vals)) if vals else float("nan")

        def avg_list(attr):
            """Average of per-episode lists (e.g. q_spreads)."""
            all_vals = []
            for d in self.history:
                all_vals.extend(getattr(d, attr))
            return float(np.mean(all_vals)) if all_vals else float("nan")

        # ── Action distribution ───────────────────────────────────────────────
        total_actions = sum(
            sum(d.action_counts) for d in self.history
        ) or 1
        action_pcts = [
            100.0 * sum(d.action_counts[i] for d in self.history) / total_actions
            for i in range(5)
        ]
        # Aggregate: big turns vs fine turns vs forward
        big_turn_pct  = action_pcts[0] + action_pcts[4]   # L45 + R45
        fine_turn_pct = action_pcts[1] + action_pcts[3]   # L22 + R22
        fw_pct        = action_pcts[2]                     # FW

        # ── Reward decomposition ──────────────────────────────────────────────
        avg_stuck_pen  = avg("stuck_penalty_total")
        avg_step_pen   = avg("step_penalty_total")
        avg_sensor_bon = avg("sensor_bonus_total")
        avg_success_b  = avg("success_bonus")

        # ── Approach efficiency ───────────────────────────────────────────────
        # Gap between first sensor detection and IR range — measures navigation
        # quality specifically in the approach phase. NaN if box never detected.
        approach_gaps = [
            d.step_at_ir - d.step_at_sensor
            for d in self.history
            if d.step_at_ir is not None and d.step_at_sensor is not None
        ]
        approach_gap = float(np.mean(approach_gaps)) if approach_gaps else float("nan")

        return {
            # Funnel — sensor_pct is now CONFIRMED box detections only
            "sensor_pct"      : pct("sensor_fired"),
            "wall_sensor_pct" : pct("wall_sensor_fired"),  # false positives (wall hits)
            "ir_pct"          : pct("ir_fired"),
            "attach_pct"      : pct("attached"),
            "success_pct"     : pct("success"),

            # Timing
            "steps_to_sensor" : avg_if("step_at_sensor", "step_at_sensor"),
            "steps_to_ir"     : avg_if("step_at_ir",     "step_at_ir"),
            "steps_to_attach" : avg_if("step_at_attach",  "step_at_attach"),
            "approach_gap"    : approach_gap,

            # Health
            "avg_blind"      : avg("blind_steps"),
            "avg_stuck"      : avg("stuck_steps"),
            "avg_push_stuck" : avg("push_stuck_steps"),
            "avg_push"       : avg("push_steps"),
            "avg_ep_len"     : avg("total_steps"),
            "avg_reward"     : avg("cumulative_reward"),

            # Action distribution
            "fw_pct"        : fw_pct,
            "fine_turn_pct" : fine_turn_pct,
            "big_turn_pct"  : big_turn_pct,

            # Reward decomposition
            "avg_stuck_pen"  : avg_stuck_pen,
            "avg_step_pen"   : avg_step_pen,
            "avg_sensor_bon" : avg_sensor_bon,
            "avg_success_b"  : avg_success_b,

            # Q-value spread
            "avg_q_spread"  : avg_list("q_spreads"),
        }


def fmt(v, decimals=0):
    """Format a float; show n/a if NaN."""
    if np.isnan(v):
        return "  n/a"
    fmt_str = f"{{:.{decimals}f}}"
    return fmt_str.format(v)


def print_diagnostics(tag: str, ep: int, n_eps: int,
                       stats: dict, epsilon: float) -> None:
    """
    Five-line diagnostic block every 100 episodes.

    [L0+wall] Ep   100/2000  ε=0.860
               FUNNEL   sensor= 95.0%  IR= 52.0%  attach= 18.0%  SUCCESS=  6.0%
               TIMING   sensor@   14  IR@  198  approach_gap=  184  attach@  312  ep=  876
               ACTIONS  FW= 38.2%  fine_turn= 35.1%  big_turn= 26.7%
               REWARDS  stuck=  -8240  step=  -612  sensor=  +18  success=   +80
               HEALTH   blind=  310  stuck=   41  push_stuck=   8  q_spread= 2.41

    Reading guide:
      ACTIONS : FW < 30%        → stuck penalty poisoned the forward Q-value
      REWARDS : stuck dominates → wall problem. step dominates → too slow.
                sensor rising   → agent getting closer to box over training.
      TIMING  : approach_gap rising → finds box but loses it during approach
      HEALTH  : q_spread < 0.5  → advantage stream not discriminating actions
                push_stuck > 0  → box wedging against wall during push phase
    """
    s = stats

    print(f"[{tag}] Ep {ep:5d}/{n_eps}  ε={epsilon:.3f}")
    print(
        f"           FUNNEL   "
        f"box_sensor={s['sensor_pct']:5.1f}%  "
        f"wall_sensor={s['wall_sensor_pct']:5.1f}%  "
        f"IR={s['ir_pct']:5.1f}%  "
        f"attach={s['attach_pct']:5.1f}%  "
        f"SUCCESS={s['success_pct']:5.1f}%"
    )
    print(
        f"           TIMING   "
        f"sensor@{fmt(s['steps_to_sensor']):>5}  "
        f"IR@{fmt(s['steps_to_ir']):>5}  "
        f"approach_gap={fmt(s['approach_gap']):>5}  "
        f"attach@{fmt(s['steps_to_attach']):>5}  "
        f"ep={fmt(s['avg_ep_len']):>5}"
    )
    print(
        f"           ACTIONS  "
        f"FW={s['fw_pct']:5.1f}%  "
        f"fine_turn={s['fine_turn_pct']:5.1f}%  "
        f"big_turn={s['big_turn_pct']:5.1f}%"
    )
    print(
        f"           REWARDS  "
        f"stuck={s['avg_stuck_pen']:>8.0f}  "
        f"step={s['avg_step_pen']:>6.0f}  "
        f"sensor={s['avg_sensor_bon']:>5.0f}  "
        f"success={s['avg_success_b']:>6.0f}"
    )
    print(
        f"           HEALTH   "
        f"blind={fmt(s['avg_blind']):>5}  "
        f"stuck={fmt(s['avg_stuck']):>4}  "
        f"push_stuck={fmt(s['avg_push_stuck']):>4}  "
        f"q_spread={s['avg_q_spread']:.2f}"
    )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN AGENT
# ═══════════════════════════════════════════════════════════════════════════════
class DDQNAgent:
    def __init__(self, input_dim: int, n_actions: int):
        self.n_actions  = n_actions
        self.online_net = DuelingDDQN(input_dim, n_actions, HIDDEN_DIM)
        self.target_net = DuelingDDQN(input_dim, n_actions, HIDDEN_DIM)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.epsilon       = EPSILON_START
        self.total_steps   = 0

    def select_action(self, state: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Returns (action_idx, q_values). q_values is None on random actions."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions), None
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            q = self.online_net(t)
            return int(q.argmax(dim=1).item()), q.squeeze(0).numpy()

    def update(self) -> float:
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return 0.0

        s, a, r, ns, d = self.replay_buffer.sample(BATCH_SIZE)
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a)
        r  = torch.FloatTensor(r)
        ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d)

        with torch.no_grad():
            next_acts = self.online_net(ns).argmax(dim=1)
            next_q    = self.target_net(ns).gather(1, next_acts.unsqueeze(1)).squeeze(1)
            target_q  = r + GAMMA * next_q * (1.0 - d)

        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.HuberLoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        torch.save({
            "online_net" : self.online_net.state_dict(),
            "epsilon"    : self.epsilon,
            "total_steps": self.total_steps,
        }, path)
        print(f"  [saved] {path}\n")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["online_net"])
        self.epsilon     = ckpt.get("epsilon",     EPSILON_END)
        self.total_steps = ckpt.get("total_steps", 0)
        print(f"  [loaded] {path}  ε={self.epsilon:.3f}  steps={self.total_steps}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def run_stage(
    agent:       DDQNAgent,
    env:         OBELIX,
    frame_stack: FrameStack,
    n_episodes:  int,
    tag:         str,
) -> None:
    window_stats = WindowStats(window=100)

    for ep in range(n_episodes):
        seed     = random.randint(0, 9_999)
        raw_obs  = env.reset(seed=seed)

        # Augment: prev reward and IR start at 0 (episode beginning)
        prev_reward = 0.0
        prev_ir     = False
        obs         = augment_obs(raw_obs, prev_reward, prev_ir)
        state       = frame_stack.reset(obs)

        # Shaping state — reset each episode
        osc_tracker = OscillationTracker()
        dx0 = float(env.bot_center_x - env.box_center_x)
        dy0 = float(env.bot_center_y - env.box_center_y)
        prev_dist = float(np.sqrt(dx0*dx0 + dy0*dy0))

        diag = EpisodeDiagnostics()
        done = False

        while not done:
            action_idx, q_vals           = agent.select_action(state)
            raw_next_obs, env_reward, done = env.step(ACTIONS[action_idx], render=False)

            # ── Reward shaping (training only) ────────────────────────────────
            osc_detected             = osc_tracker.update(env.bot_center_x,
                                                          env.bot_center_y)
            shaped_reward, prev_dist = shape_reward(
                env_reward, env, prev_dist, osc_detected,
                box_visible=env.box_visible or env.enable_push
            )

            # Build augmented next obs using THIS step's env reward
            next_obs   = augment_obs(raw_next_obs, env_reward,
                                     bool(raw_next_obs[16] > 0))
            next_state = frame_stack.push(next_obs)

            # Store SHAPED reward in replay buffer — this is what the network learns from
            agent.replay_buffer.push(state, action_idx, shaped_reward, next_state, done)
            agent.update()

            # Diagnostics always use raw env reward (not shaped) for honest reporting
            diag.update(raw_next_obs, env_reward, env,
                        action_idx=action_idx, q_values=q_vals)

            state       = next_state
            prev_reward = env_reward
            prev_ir     = bool(raw_next_obs[16] > 0)

        window_stats.add(diag)

        # Decay epsilon once per episode — NOT per gradient step.
        # Per-step decay (0.9995^1000 steps/ep) collapses ε to 0.05 in ~6
        # episodes. Per-episode decay (0.9985^ep) reaches 0.05 after ~2000 eps,
        # giving the agent time to explore before going greedy.
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if (ep + 1) % 100 == 0:
            print_diagnostics(
                tag, ep + 1, n_episodes,
                window_stats.report(), agent.epsilon
            )

    print(f"[{tag}] Stage complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main(args: argparse.Namespace) -> None:
    agent       = DDQNAgent(INPUT_DIM, N_ACTIONS)
    frame_stack = FrameStack(FRAME_STACK, OBS_DIM)

    # ── Header — printed once so you always have the diagnosis guide handy ────
    print("=" * 72)
    print("  OBELIX Phase 2 Training — Dueling DDQN + Frame Stacking (K=16)")
    print(f"  sf={SCALING_FACTOR}  arena={ARENA_SIZE}  max_steps={MAX_STEPS}  wall={WALL_OBSTACLES}")
    print(f"  INPUT_DIM={INPUT_DIM}  HIDDEN={HIDDEN_DIM}  batch={BATCH_SIZE}  buf={REPLAY_BUFFER_SIZE}")
    print()
    print("  DIAGNOSTIC GUIDE")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  FUNNEL shows % of 100 eps that reached each phase:")
    print("    sensor → IR → attach → SUCCESS")
    print()
    print("  If sensor ≈ 0%           SEARCH broken  — never finds box")
    print("  If sensor ok, IR ≈ 0%    APPROACH broken — can't close to IR range")
    print("  If IR ok, attach ≈ 0%    ALIGN broken   — overshoots, can't contact")
    print("  If attach ok, success≈0% PUSH broken    — wall blocks path to edge")
    print()
    print("  HEALTH flags:")
    print("    stuck > 50/ep  → wall collisions dominating (-200 each)")
    print("    blind > 400/ep → too long without sensor signal")
    print("    (Stage 2: higher blind is expected — box blinks out)")
    print("=" * 72)
    print()

    # ── Stage 1: Level 0 + walls ───────────────────────────────────────────────
    print(f"  Stage 1: Level 0 + walls  ({args.stage1_eps} episodes)")
    print()

    env0 = OBELIX(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        difficulty=0,
        wall_obstacles=WALL_OBSTACLES,
    )
    run_stage(agent, env0, frame_stack, args.stage1_eps, tag="L0+wall")
    agent.save(STAGE1_SAVE)

    # ── Stage 2: Level 2 + walls ───────────────────────────────────────────────
    print(f"  Stage 2: Level 2 + walls  ({args.stage2_eps} episodes)")
    print("  Resetting ε=0.35 — agent knows the basics, re-explores for blinking")
    print()
    print("  Stage 2 vs Stage 1: blind steps will rise (blinking is working).")
    print("  Watch for: attach_pct dropping — that is the Phase 2 failure mode.")
    print()

    agent.epsilon = 0.35

    env2 = OBELIX(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        difficulty=2,
        wall_obstacles=WALL_OBSTACLES,
    )
    run_stage(agent, env2, frame_stack, args.stage2_eps, tag="L2+wall")
    agent.save(STAGE2_SAVE)

    print(f"Training complete.  Submit: agent.py + {STAGE2_SAVE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBELIX Phase 2 Training")
    parser.add_argument(
        "--stage1_eps", type=int, default=STAGE1_EPISODES,
        help=f"Stage 1 episodes (default: {STAGE1_EPISODES})"
    )
    parser.add_argument(
        "--stage2_eps", type=int, default=STAGE2_EPISODES,
        help=f"Stage 2 episodes (default: {STAGE2_EPISODES})"
    )
    args = parser.parse_args()
    main(args)