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

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


SCALING_FACTOR = 5
ARENA_SIZE     = 500
WALL_OBSTACLES = True
MAX_STEPS      = 1000


FRAME_STACK = 16
OBS_DIM_RAW = 18
OBS_DIM     = OBS_DIM_RAW + 2
INPUT_DIM   = OBS_DIM * FRAME_STACK


HIDDEN_DIM = 256


GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 150_000
TARGET_UPDATE_FREQ = 1000
MIN_REPLAY_SIZE = 5000

EPSILON_END = 0.05
EPSILON_DECAY = 0.9985

STAGE2_EPISODES = 3000


STAGE2_EPSILON_RESET = 0.35

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

STAGE1_SAVE = "weights_stage1.pth"
STAGE2_SAVE = "weights_phase2.pth"

DIST_SHAPING_ALPHA   = 0.5
PUSH_SHAPING_ALPHA   = 1.0

OSCILLATION_WINDOW    = 30
OSCILLATION_THRESHOLD = 15
OSCILLATION_PENALTY   = -2.0


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


def augment_obs(raw_obs: np.ndarray, prev_reward: float, prev_ir: bool) -> np.ndarray:
    was_stuck  = 1.0 if prev_reward <= -199.0 else 0.0
    was_ir_hit = 1.0 if prev_ir else 0.0
    return np.append(raw_obs.astype(np.float32), [was_stuck, was_ir_hit])


class OscillationTracker:
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
        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) < self.window:
            return False
        return (float(np.std(self.xs)) < self.threshold and
                float(np.std(self.ys)) < self.threshold)


def shape_reward(env_reward: float, env: OBELIX,
                 prev_dist: float, osc_detected: bool,
                 box_visible: bool) -> tuple[float, float]:
    dx = float(env.bot_center_x - env.box_center_x)
    dy        = float(env.bot_center_y - env.box_center_y)
    curr_dist = float(np.sqrt(dx*dx + dy*dy))

    alpha    = PUSH_SHAPING_ALPHA if env.enable_push else DIST_SHAPING_ALPHA
    progress = prev_dist - curr_dist

    shaped = env_reward + alpha * progress

    if osc_detected and box_visible:
        shaped += OSCILLATION_PENALTY

    return shaped, curr_dist


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


class EpisodeDiagnostics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_steps       = 0
        self.cumulative_reward = 0.0

        self.sensor_fired      = False
        self.wall_sensor_fired = False
        self.ir_fired          = False
        self.attached          = False
        self.success           = False

        self.step_at_sensor    = None
        self.step_at_ir        = None
        self.step_at_attach    = None

        self.blind_steps      = 0
        self.stuck_steps      = 0
        self.push_stuck_steps = 0
        self.push_steps       = 0

        self.action_counts = [0] * 5

        self.stuck_penalty_total = 0.0
        self.step_penalty_total  = 0.0
        self.sensor_bonus_total  = 0.0
        self.success_bonus       = 0.0

        self.q_spreads: list = []

    def update(self, obs: np.ndarray, reward: float, env: OBELIX,
               action_idx: int = -1, q_values: np.ndarray | None = None):
        self.total_steps       += 1
        self.cumulative_reward += reward

        raw          = obs[:18]
        sonar_active = bool(np.any(raw[:16] > 0))
        ir_active    = bool(raw[16] > 0)
        stuck        = bool(raw[17] > 0) and not env.enable_push
        push_stuck   = bool(raw[17] > 0) and env.enable_push
        in_push      = env.enable_push

        dx          = env.bot_center_x - env.box_center_x
        dy          = env.bot_center_y - env.box_center_y
        dist_to_box = float(np.sqrt(dx*dx + dy*dy))
        sonar_range = 30 * env.scaling_factor
        near_box    = dist_to_box < (sonar_range + env.bot_radius)

        if sonar_active or ir_active:
            if near_box and not self.sensor_fired:
                self.sensor_fired   = True
                self.step_at_sensor = self.total_steps
            elif not near_box and not self.wall_sensor_fired:
                self.wall_sensor_fired = True

        if ir_active and not self.ir_fired:
            self.ir_fired   = True
            self.step_at_ir = self.total_steps

        if in_push and not self.attached:
            self.attached       = True
            self.step_at_attach = self.total_steps

        if reward >= 1999.0:
            self.success = True

        if not sonar_active and not ir_active:
            self.blind_steps += 1
        if stuck:
            self.stuck_steps += 1
        if push_stuck:
            self.push_stuck_steps += 1
        if in_push:
            self.push_steps += 1

        if 0 <= action_idx < 5:
            self.action_counts[action_idx] += 1

        if stuck or push_stuck:
            self.stuck_penalty_total += reward
        elif reward >= 1999.0:
            self.success_bonus        = reward
        elif reward > 0:
            self.sensor_bonus_total  += reward
        else:
            self.step_penalty_total  += reward

        if q_values is not None and len(q_values) == 5:
            self.q_spreads.append(float(np.max(q_values) - np.min(q_values)))


class WindowStats:
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
            all_vals = []
            for d in self.history:
                all_vals.extend(getattr(d, attr))
            return float(np.mean(all_vals)) if all_vals else float("nan")

        total_actions = sum(sum(d.action_counts) for d in self.history) or 1
        action_pcts = [
            100.0 * sum(d.action_counts[i] for d in self.history) / total_actions
            for i in range(5)
        ]
        big_turn_pct  = action_pcts[0] + action_pcts[4]
        fine_turn_pct = action_pcts[1] + action_pcts[3]
        fw_pct        = action_pcts[2]

        approach_gaps = [
            d.step_at_ir - d.step_at_sensor
            for d in self.history
            if d.step_at_ir is not None and d.step_at_sensor is not None
        ]
        approach_gap = float(np.mean(approach_gaps)) if approach_gaps else float("nan")

        return {
            "sensor_pct"      : pct("sensor_fired"),
            "wall_sensor_pct" : pct("wall_sensor_fired"),
            "ir_pct"          : pct("ir_fired"),
            "attach_pct"      : pct("attached"),
            "success_pct"     : pct("success"),
            "steps_to_sensor" : avg_if("step_at_sensor", "step_at_sensor"),
            "steps_to_ir"     : avg_if("step_at_ir",     "step_at_ir"),
            "steps_to_attach" : avg_if("step_at_attach",  "step_at_attach"),
            "approach_gap"    : approach_gap,
            "avg_blind"       : avg("blind_steps"),
            "avg_stuck"       : avg("stuck_steps"),
            "avg_push_stuck"  : avg("push_stuck_steps"),
            "avg_push"        : avg("push_steps"),
            "avg_ep_len"      : avg("total_steps"),
            "avg_reward"      : avg("cumulative_reward"),
            "fw_pct"          : fw_pct,
            "fine_turn_pct"   : fine_turn_pct,
            "big_turn_pct"    : big_turn_pct,
            "avg_stuck_pen"   : avg("stuck_penalty_total"),
            "avg_step_pen"    : avg("step_penalty_total"),
            "avg_sensor_bon"  : avg("sensor_bonus_total"),
            "avg_success_b"   : avg("success_bonus"),
            "avg_q_spread"    : avg_list("q_spreads"),
        }


def fmt(v, decimals=0):
    if np.isnan(v):
        return "  n/a"
    return f"{{:.{decimals}f}}".format(v)


def print_diagnostics(tag: str, ep: int, n_eps: int,
                      stats: dict, epsilon: float) -> None:
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


class DDQNAgent:
    def __init__(self, input_dim: int, n_actions: int):
        self.n_actions  = n_actions
        self.online_net = DuelingDDQN(input_dim, n_actions, HIDDEN_DIM)
        self.target_net = DuelingDDQN(input_dim, n_actions, HIDDEN_DIM)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.epsilon       = EPSILON_END
        self.total_steps   = 0

    def select_action(self, state: np.ndarray) -> tuple[int, np.ndarray | None]:
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


def run_stage(
    agent:       DDQNAgent,
    env:         OBELIX,
    frame_stack: FrameStack,
    n_episodes:  int,
    tag:         str,
) -> None:
    window_stats = WindowStats(window=100)
    best_success = 0.0

    for ep in range(n_episodes):
        seed     = random.randint(0, 9_999)
        raw_obs  = env.reset(seed=seed)

        prev_reward = 0.0
        prev_ir     = False
        obs         = augment_obs(raw_obs, prev_reward, prev_ir)
        state       = frame_stack.reset(obs)

        osc_tracker = OscillationTracker()
        dx0 = float(env.bot_center_x - env.box_center_x)
        dy0 = float(env.bot_center_y - env.box_center_y)
        prev_dist = float(np.sqrt(dx0*dx0 + dy0*dy0))

        diag = EpisodeDiagnostics()
        done = False

        while not done:
            action_idx, q_vals             = agent.select_action(state)
            raw_next_obs, env_reward, done = env.step(ACTIONS[action_idx], render=False)

            osc_detected             = osc_tracker.update(env.bot_center_x,
                                                          env.bot_center_y)
            shaped_reward, prev_dist = shape_reward(
                env_reward, env, prev_dist, osc_detected,
                box_visible=env.box_visible or env.enable_push
            )

            next_obs   = augment_obs(raw_next_obs, env_reward,
                                     bool(raw_next_obs[16] > 0))
            next_state = frame_stack.push(next_obs)

            agent.replay_buffer.push(state, action_idx, shaped_reward, next_state, done)
            agent.update()

            diag.update(raw_next_obs, env_reward, env,
                        action_idx=action_idx, q_values=q_vals)

            state       = next_state
            prev_reward = env_reward
            prev_ir     = bool(raw_next_obs[16] > 0)

        window_stats.add(diag)

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if (ep + 1) % 100 == 0:
            stats = window_stats.report()
            print_diagnostics(tag, ep + 1, n_episodes, stats, agent.epsilon)

            periodic_path = f"weights_{tag}_ep{ep + 1}.pth"
            agent.save(periodic_path)

            if stats["success_pct"] > best_success:
                best_success = stats["success_pct"]
                best_path    = f"weights_{tag}_best.pth"
                agent.save(best_path)
                print(f"  [best] New best success={best_success:.1f}%"
                      f" → saved {best_path}\n")

    print(f"[{tag}] Stage complete.\n")


def main(args: argparse.Namespace) -> None:

    if not os.path.exists(STAGE1_SAVE):
        print(f"ERROR: {STAGE1_SAVE} not found in current directory.")
        print("       Stage 1 must be completed before running this script.")
        sys.exit(1)

    agent       = DDQNAgent(INPUT_DIM, N_ACTIONS)
    frame_stack = FrameStack(FRAME_STACK, OBS_DIM)

    print("=" * 72)
    print("  OBELIX Phase 2 Training — Stage 2 only (blinking box)")
    print(f"  sf={SCALING_FACTOR}  arena={ARENA_SIZE}  max_steps={MAX_STEPS}  wall={WALL_OBSTACLES}")
    print(f"  INPUT_DIM={INPUT_DIM}  HIDDEN={HIDDEN_DIM}  batch={BATCH_SIZE}  buf={REPLAY_BUFFER_SIZE}")
    print()
    print("  Checkpointing:")
    print("    weights_L2+wall_ep<N>.pth  — saved every 100 episodes")
    print("    weights_L2+wall_best.pth   — saved on every new success% peak")
    print("    weights_phase2.pth         — saved at end of training")
    print()
    print("  SUBMIT: weights_L2+wall_best.pth  (not necessarily the final weights)")
    print("=" * 72)
    print()

    print(f"  Loading Stage 1 weights from {STAGE1_SAVE} ...")
    agent.load(STAGE1_SAVE)

    agent.epsilon = STAGE2_EPSILON_RESET
    print(f"  Epsilon reset to {STAGE2_EPSILON_RESET} for Stage 2 re-exploration")
    print()

    print(f"  Stage 2: Level 2 + walls  ({args.stage2_eps} episodes)")
    print("  blind steps will rise vs Stage 1 — the box is blinking, this is correct.")
    print("  Key metric to watch: attach% must stay above 35%.")
    print()

    env2 = OBELIX(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        difficulty=2,
        wall_obstacles=WALL_OBSTACLES,
    )
    run_stage(agent, env2, frame_stack, args.stage2_eps, tag="L2+wall")
    agent.save(STAGE2_SAVE)

    print("Training complete.")
    print(f"  Best weights : weights_L2+wall_best.pth  <- submit this to Codabench")
    print(f"  Final weights: {STAGE2_SAVE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBELIX Phase 2 — Stage 2 Training")
    parser.add_argument(
        "--stage2_eps", type=int, default=STAGE2_EPISODES,
        help=f"Stage 2 episodes (default: {STAGE2_EPISODES})"
    )
    args = parser.parse_args()
    main(args)