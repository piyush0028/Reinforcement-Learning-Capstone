from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


SCALING_FACTOR = 5
ARENA_SIZE     = 500
MAX_STEPS      = 2000
BOX_SPEED      = 2


FRAME_STACK = 32
OBS_DIM_RAW = 18
OBS_DIM     = OBS_DIM_RAW + 2
INPUT_DIM   = OBS_DIM * FRAME_STACK

LEFT_IDX  = [0, 1, 2, 3, 4, 5]
RIGHT_IDX = [6, 7, 8, 9, 10, 11]

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


GAMMA              = 0.99
LR                 = 1e-4
BATCH_SIZE         = 128
REPLAY_BUFFER_SIZE = 200_000
TARGET_UPDATE_FREQ = 1000
MIN_REPLAY_SIZE    = 5000

EPSILON_START      = 0.20
EPSILON_END        = 0.03
EPSILON_DECAY      = 0.9985

HIDDEN_DIM = 256


DIST_ALPHA          = 1.5
PUSH_ALPHA          = 2.5
GRID_BONUS          = 1.5
EDGE_CUSHION        = 50.0
EDGE_APPROACH_PEN   = -5.0
EDGE_RETREAT_BON    =  2.0
SENSOR_STEER_BON    =  3.0
SENSOR_STEER_PEN    = -5.0
GHOST_MOMENTUM_BON  =  2.0
WIGGLE_PEN          = -15.0
TURN_SPAM_PEN       = -20.0
PUSH_BIGTURN_PEN    = -10.0
PUSH_MOMENTUM_BON   =  3.0
PUSH_NO_MOVE_PEN    = -1.0
WALL_APPROACH_PEN   = -4.0
WALL_RETREAT_BON    =  2.0
GHOST_TIMER_RESET   = 30


def augment_obs(raw_obs: np.ndarray,
                prev_reward: float,
                prev_ir: bool) -> np.ndarray:
    return np.append(
        raw_obs.astype(np.float32),
        [1.0 if prev_reward <= -199.0 else 0.0,
         1.0 if prev_ir else 0.0]
    )


class FrameStack:
    def __init__(self, k: int, obs_dim: int):
        self.k       = k
        self.obs_dim = obs_dim
        self.frames: deque = deque(maxlen=k)
        self.reset()

    def reset(self, first_obs: np.ndarray | None = None) -> np.ndarray:
        obs = (np.zeros(self.obs_dim, dtype=np.float32)
               if first_obs is None else first_obs.astype(np.float32))
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self.get()

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.astype(np.float32))
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32), int(action), float(reward),
            np.array(next_state, dtype=np.float32), float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.stack(s), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.stack(ns),
                np.array(d, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDDQN(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM,
                 n_actions: int = N_ACTIONS,
                 hidden_dim: int = HIDDEN_DIM):
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
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)


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
        s  = torch.FloatTensor(s);  a  = torch.LongTensor(a)
        r  = torch.FloatTensor(r);  ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d)
        with torch.no_grad():
            na = self.online_net(ns).argmax(dim=1)
            nq = self.target_net(ns).gather(1, na.unsqueeze(1)).squeeze(1)
            tq = r + GAMMA * nq * (1.0 - d)
        cq   = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.HuberLoss()(cq, tq)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.total_steps += 1
        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()

    def save(self, path: str) -> None:
        torch.save({"online_net": self.online_net.state_dict(),
                    "epsilon": self.epsilon,
                    "total_steps": self.total_steps}, path)
        print(f"  [saved] {path}")

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu")
        sd = ck.get("online_net", ck)
        try:
            self.online_net.load_state_dict(sd, strict=True)
            self.target_net.load_state_dict(sd)
            self.epsilon     = ck.get("epsilon",     EPSILON_START)
            self.total_steps = ck.get("total_steps", 0)
            print(f"  [loaded] {path}  ε={self.epsilon:.3f}")
        except RuntimeError as e:
            print(f"  [warn] Could not load weights ({e}). Starting fresh.")


class BoxInterceptTracker:
    def __init__(self, history_len: int = 6, lookahead: int = 10):
        self.history_len = history_len
        self.lookahead   = lookahead
        self.xs: deque   = deque(maxlen=history_len)
        self.ys: deque   = deque(maxlen=history_len)

    def reset(self, x: float, y: float) -> None:
        self.xs.clear()
        self.ys.clear()
        self.xs.append(x)
        self.ys.append(y)

    def observe(self, x: float, y: float) -> None:
        self.xs.append(x)
        self.ys.append(y)

    def _velocity(self) -> tuple[float, float]:
        if len(self.xs) < 2:
            return 0.0, 0.0
        n  = len(self.xs) - 1
        vx = (self.xs[-1] - self.xs[0]) / n
        vy = (self.ys[-1] - self.ys[0]) / n
        return vx, vy

    def intercept(self, arena: int = ARENA_SIZE) -> tuple[float, float]:
        if not self.xs:
            return float(arena / 2), float(arena / 2)
        vx, vy = self._velocity()
        tx = float(np.clip(self.xs[-1] + vx * self.lookahead, 20, arena - 20))
        ty = float(np.clip(self.ys[-1] + vy * self.lookahead, 20, arena - 20))
        return tx, ty


def route_through_gap(bot_x: float, bot_y: float,
                      target_x: float, target_y: float,
                      use_walls: bool,
                      arena: int = ARENA_SIZE) -> tuple[float, float]:
    if not use_walls:
        return target_x, target_y

    cx       = arena / 2.0
    bot_left = bot_x < cx
    tgt_left = target_x < cx

    if bot_left == tgt_left:
        return target_x, target_y

    gap_y = arena / 2.0
    gap_x = cx + (25.0 if not bot_left else -25.0)
    return gap_x, gap_y


class OscillationTracker:
    def __init__(self, window: int = 20, threshold: float = 20.0):
        self.window    = window
        self.threshold = threshold
        self.xs: deque = deque(maxlen=window)
        self.ys: deque = deque(maxlen=window)

    def reset(self) -> None:
        self.xs.clear()
        self.ys.clear()

    def update(self, x: float, y: float) -> bool:
        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) < self.window:
            return False
        return (float(np.std(self.xs)) < self.threshold and
                float(np.std(self.ys)) < self.threshold)


def shape_reward(
    env_reward:     float,
    env,
    use_walls:      bool,
    bot_x:          float,
    bot_y:          float,
    box_x:          float,
    box_y:          float,
    prev_bot_x:     float,
    prev_bot_y:     float,
    prev_box_x:     float,
    prev_box_y:     float,
    prev_dist:      float,
    prev_bound_dist: float | None,
    action_str:     str,
    prev_action:    str,
    consec_turns:   int,
    ghost_timer:    int,
    sensors_clear:  bool,
    box_visible:    bool,
    visited_grid:   np.ndarray,
    intercept:      BoxInterceptTracker,
    osc_detected:   bool,
    arena:          int = ARENA_SIZE,
) -> tuple[float, float, float | None]:
    shaped     = env_reward
    arena_half = arena / 2.0

    tx, ty   = intercept.intercept(arena)
    tx, ty   = route_through_gap(bot_x, bot_y, tx, ty, use_walls, arena)
    curr_dist = math.hypot(bot_x - tx, bot_y - ty)

    if not env.enable_push:
        agent_moved = abs(bot_x - prev_bot_x) > 0 or abs(bot_y - prev_bot_y) > 0

        shaped += DIST_ALPHA * (prev_dist - curr_dist)

        cell_x = min(24, int(bot_x / (arena / 25)))
        cell_y = min(24, int(bot_y / (arena / 25)))
        if not visited_grid[cell_y, cell_x] and agent_moved:
            visited_grid[cell_y, cell_x] = True
            shaped += GRID_BONUS

        edge_now  = min(bot_x, arena - bot_x, bot_y, arena - bot_y)
        edge_prev = min(prev_bot_x, arena - prev_bot_x,
                        prev_bot_y, arena - prev_bot_y)
        if edge_now < EDGE_CUSHION:
            shaped += EDGE_APPROACH_PEN if edge_now < edge_prev else EDGE_RETREAT_BON

        is_stuck_flag = bool(env.sensor_feedback[17] > 0)
        if not is_stuck_flag and not sensors_clear:
            left_on  = any(env.sensor_feedback[i] > 0 for i in LEFT_IDX)
            right_on = any(env.sensor_feedback[i] > 0 for i in RIGHT_IDX)
            if left_on and not right_on:
                shaped += (SENSOR_STEER_BON if action_str in ["L22", "L45"]
                           else SENSOR_STEER_PEN if action_str in ["R22", "R45"]
                           else 0.0)
            elif right_on and not left_on:
                shaped += (SENSOR_STEER_BON if action_str in ["R22", "R45"]
                           else SENSOR_STEER_PEN if action_str in ["L22", "L45"]
                           else 0.0)

        if action_str == "FW" and prev_action == "FW":
            if not sensors_clear or ghost_timer > 0:
                shaped += GHOST_MOMENTUM_BON

        if osc_detected and box_visible:
            shaped += -3.0

    else:
        dist_to_edge_x = min(box_x - 10, arena - 10 - box_x)
        dist_to_edge_y = min(box_y - 10, arena - 10 - box_y)
        curr_bound     = min(dist_to_edge_x, dist_to_edge_y)

        if prev_bound_dist is not None:
            box_moved = abs(box_x - prev_box_x) > 0 or abs(box_y - prev_box_y) > 0
            if box_moved:
                shaped += PUSH_ALPHA * (prev_bound_dist - curr_bound)
                if use_walls:
                    shaped += (WALL_APPROACH_PEN
                               if abs(box_x - arena_half) < abs(prev_box_x - arena_half)
                               else WALL_RETREAT_BON)
            else:
                shaped += PUSH_NO_MOVE_PEN

        prev_bound_dist = curr_bound

        if action_str in ["L45", "R45"]:
            shaped += PUSH_BIGTURN_PEN
        if action_str == "FW" and prev_action == "FW":
            shaped += PUSH_MOMENTUM_BON

    if ((prev_action in ["L22", "L45"] and action_str in ["R22", "R45"]) or
            (prev_action in ["R22", "R45"] and action_str in ["L22", "L45"])):
        shaped += WIGGLE_PEN

    if action_str != "FW" and not bool(env.sensor_feedback[17] > 0):
        if consec_turns >= 4:
            shaped += TURN_SPAM_PEN

    return shaped, curr_dist, prev_bound_dist


class EpisodeDiagnostics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_steps = 0;  self.cumulative_reward = 0.0
        self.sensor_fired = False;  self.wall_sensor_fired = False
        self.ir_fired = False;  self.attached = False;  self.success = False
        self.step_at_sensor = None;  self.step_at_ir = None
        self.step_at_attach = None
        self.blind_steps = 0;  self.stuck_steps = 0
        self.push_stuck_steps = 0;  self.push_steps = 0
        self.action_counts = [0] * 5
        self.stuck_pen = 0.0;  self.step_pen = 0.0
        self.sensor_bon = 0.0;  self.success_bon = 0.0
        self.q_spreads: list = []

    def update(self, raw_obs: np.ndarray, reward: float, env,
               action_idx: int = -1, q_vals: np.ndarray | None = None):
        self.total_steps += 1
        self.cumulative_reward += reward

        sonar = bool(np.any(raw_obs[:16] > 0))
        ir    = bool(raw_obs[16] > 0)
        stuck = bool(raw_obs[17] > 0) and not env.enable_push
        pstk  = bool(raw_obs[17] > 0) and env.enable_push

        dx = env.bot_center_x - env.box_center_x
        dy = env.bot_center_y - env.box_center_y
        near_box = (math.hypot(dx, dy) < 30 * env.scaling_factor + env.bot_radius)

        if sonar or ir:
            if near_box and not self.sensor_fired:
                self.sensor_fired = True; self.step_at_sensor = self.total_steps
            elif not near_box and not self.wall_sensor_fired:
                self.wall_sensor_fired = True

        if ir and not self.ir_fired:
            self.ir_fired = True; self.step_at_ir = self.total_steps
        if env.enable_push and not self.attached:
            self.attached = True; self.step_at_attach = self.total_steps
        if reward >= 1999.0:
            self.success = True

        if not sonar and not ir: self.blind_steps += 1
        if stuck:  self.stuck_steps += 1
        if pstk:   self.push_stuck_steps += 1
        if env.enable_push: self.push_steps += 1
        if 0 <= action_idx < 5: self.action_counts[action_idx] += 1

        if stuck or pstk:   self.stuck_pen += reward
        elif reward >= 1999: self.success_bon = reward
        elif reward > 0:    self.sensor_bon += reward
        else:               self.step_pen += reward

        if q_vals is not None and len(q_vals) == 5:
            self.q_spreads.append(float(np.max(q_vals) - np.min(q_vals)))


class WindowStats:
    def __init__(self, window: int = 100):
        self.window = window
        self.history: deque = deque(maxlen=window)

    def add(self, d: EpisodeDiagnostics):
        self.history.append(d)

    def report(self) -> dict:
        n = len(self.history)
        if n == 0: return {}
        pct  = lambda a: 100.0 * sum(getattr(d, a) for d in self.history) / n
        avg  = lambda a: float(np.mean([getattr(d, a) for d in self.history]))
        avgi = lambda v, g: float(np.mean([getattr(d, v) for d in self.history
                                           if getattr(d, g) is not None])) \
               if any(getattr(d, g) is not None for d in self.history) else float("nan")
        avgl = lambda a: (float(np.mean([x for d in self.history for x in getattr(d, a)]))
                          if any(getattr(d, a) for d in self.history) else float("nan"))
        ta = sum(sum(d.action_counts) for d in self.history) or 1
        ap = [100.0 * sum(d.action_counts[i] for d in self.history) / ta for i in range(5)]
        gaps = [d.step_at_ir - d.step_at_sensor for d in self.history
                if d.step_at_ir is not None and d.step_at_sensor is not None]
        return {
            "sensor_pct": pct("sensor_fired"), "wall_pct": pct("wall_sensor_fired"),
            "ir_pct": pct("ir_fired"), "attach_pct": pct("attached"),
            "success_pct": pct("success"),
            "sensor_step": avgi("step_at_sensor", "step_at_sensor"),
            "ir_step": avgi("step_at_ir", "step_at_ir"),
            "attach_step": avgi("step_at_attach", "step_at_attach"),
            "gap": float(np.mean(gaps)) if gaps else float("nan"),
            "blind": avg("blind_steps"), "stuck": avg("stuck_steps"),
            "push_stuck": avg("push_stuck_steps"), "push": avg("push_steps"),
            "ep_len": avg("total_steps"), "reward": avg("cumulative_reward"),
            "fw": ap[2], "fine": ap[1] + ap[3], "big": ap[0] + ap[4],
            "stuck_pen": avg("stuck_pen"), "step_pen": avg("step_pen"),
            "sensor_bon": avg("sensor_bon"), "success_bon": avg("success_bon"),
            "q_spread": avgl("q_spreads"),
        }


def _f(v):
    return "  n/a" if (isinstance(v, float) and math.isnan(v)) else f"{v:5.0f}"


def print_diagnostics(tag: str, ep: int, n_eps: int,
                      s: dict, eps: float) -> None:
    print(f"[{tag}] Ep {ep:5d}/{n_eps}  ε={eps:.3f}")
    print(f"         FUNNEL  box={s['sensor_pct']:5.1f}%  wall={s['wall_pct']:5.1f}%  "
          f"IR={s['ir_pct']:5.1f}%  attach={s['attach_pct']:5.1f}%  "
          f"SUCCESS={s['success_pct']:5.1f}%")
    print(f"         TIMING  sensor@{_f(s['sensor_step'])}  IR@{_f(s['ir_step'])}  "
          f"gap={_f(s['gap'])}  attach@{_f(s['attach_step'])}  ep={_f(s['ep_len'])}")
    print(f"         ACTIONS FW={s['fw']:5.1f}%  fine={s['fine']:5.1f}%  "
          f"big={s['big']:5.1f}%")
    print(f"         REWARDS stuck={s['stuck_pen']:8.0f}  step={s['step_pen']:7.0f}  "
          f"sensor={s['sensor_bon']:5.0f}  success={s['success_bon']:6.0f}")
    print(f"         HEALTH  blind={_f(s['blind'])}  stuck={s['stuck']:4.0f}  "
          f"push_stuck={s['push_stuck']:4.0f}  q_spread={s['q_spread']:.2f}")
    print()


def run_stage(agent: DDQNAgent, OBELIX, frame_stack: FrameStack,
              n_episodes: int, difficulty: int, wall_prob: float,
              tag: str, save_best_as: str | None = None,
              box_speed: int = BOX_SPEED) -> float:
    window_stats = WindowStats(100)
    best_success = 0.0

    for ep in range(n_episodes):
        use_walls = random.random() < wall_prob
        seed      = random.randint(0, 9_999)

        base_env = OBELIX(
            scaling_factor=SCALING_FACTOR, arena_size=ARENA_SIZE,
            max_steps=MAX_STEPS, wall_obstacles=use_walls,
            difficulty=difficulty, box_speed=box_speed, seed=seed,
        )
        raw_obs = base_env.reset(seed=seed)

        prev_reward = 0.0;  prev_ir = False
        obs   = augment_obs(raw_obs, prev_reward, prev_ir)
        state = frame_stack.reset(obs)

        bot_x = float(base_env.bot_center_x)
        bot_y = float(base_env.bot_center_y)
        box_x = float(base_env.box_center_x)
        box_y = float(base_env.box_center_y)

        intercept = BoxInterceptTracker(history_len=6, lookahead=10)
        intercept.reset(box_x, box_y)

        osc = OscillationTracker(window=20, threshold=20.0)
        visited_grid = np.zeros((25, 25), dtype=bool)
        ghost_timer  = 0
        consec_turns = 0
        prev_action  = "FW"

        tx, ty   = route_through_gap(bot_x, bot_y, box_x, box_y, use_walls)
        prev_dist = math.hypot(bot_x - tx, bot_y - ty)
        prev_bound_dist = None
        prev_bot_x, prev_bot_y = bot_x, bot_y
        prev_box_x, prev_box_y = box_x, box_y

        diag = EpisodeDiagnostics();  done = False

        while not done:
            action_idx, q_vals = agent.select_action(state)
            raw_next, env_r, done = base_env.step(ACTIONS[action_idx], render=False)

            bot_x = float(base_env.bot_center_x)
            bot_y = float(base_env.bot_center_y)
            box_x = float(base_env.box_center_x)
            box_y = float(base_env.box_center_y)
            box_visible   = bool(base_env.box_visible or base_env.enable_push)
            sensors_clear = not np.any(base_env.sensor_feedback[:17])

            ghost_timer = GHOST_TIMER_RESET if not sensors_clear \
                          else max(0, ghost_timer - 1)

            if box_visible:
                intercept.observe(box_x, box_y)

            consec_turns = 0 if ACTIONS[action_idx] == "FW" else consec_turns + 1

            osc_det = osc.update(bot_x, bot_y)

            shaped_r, prev_dist, prev_bound_dist = shape_reward(
                env_r, base_env, use_walls,
                bot_x, bot_y, box_x, box_y,
                prev_bot_x, prev_bot_y, prev_box_x, prev_box_y,
                prev_dist, prev_bound_dist,
                ACTIONS[action_idx], prev_action,
                consec_turns, ghost_timer,
                sensors_clear, box_visible,
                visited_grid, intercept, osc_det,
            )

            next_obs   = augment_obs(raw_next, env_r, bool(raw_next[16] > 0))
            next_state = frame_stack.push(next_obs)

            agent.replay_buffer.push(state, action_idx, shaped_r, next_state, done)
            agent.update()

            diag.update(raw_next, env_r, base_env,
                        action_idx=action_idx, q_vals=q_vals)

            state = next_state;  prev_reward = env_r
            prev_ir = bool(raw_next[16] > 0)
            prev_bot_x, prev_bot_y = bot_x, bot_y
            prev_box_x, prev_box_y = box_x, box_y
            prev_action = ACTIONS[action_idx]

        window_stats.add(diag)
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if (ep + 1) % 100 == 0:
            stats = window_stats.report()
            print_diagnostics(tag, ep + 1, n_episodes, stats, agent.epsilon)
            if save_best_as and stats["success_pct"] > best_success:
                best_success = stats["success_pct"]
                agent.save(save_best_as)
                print(f"  [best] success={best_success:.1f}% → {save_best_as}\n")

    print(f"[{tag}] Done. Best success: {best_success:.1f}%\n")
    return best_success


def main(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_mod", args.obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    agent       = DDQNAgent(INPUT_DIM, N_ACTIONS)
    frame_stack = FrameStack(FRAME_STACK, OBS_DIM)

    print("=" * 70)
    print(f"  OBELIX Phase 3  |  K={FRAME_STACK}  INPUT_DIM={INPUT_DIM}")
    print(f"  MAX_STEPS={MAX_STEPS}  BOX_SPEED={BOX_SPEED}  wall_prob={args.wall_prob}")
    print(f"  Loading: {args.weights_file}")
    print(f"  ε: {EPSILON_START}→{EPSILON_END}")
    print(f"  Submit: weights_phase3_best.pth")
    print("=" * 70 + "\n")

    if os.path.exists(args.weights_file):
        agent.load(args.weights_file)
        agent.epsilon = EPSILON_START
    else:
        print(f"  No weights found at {args.weights_file} — training from scratch.\n")

    if args.warmup_eps > 0:
        print(f"  Warmup: {args.warmup_eps} eps on Level 0 + walls")
        run_stage(agent, OBELIX, frame_stack,
                  args.warmup_eps, difficulty=0, wall_prob=args.wall_prob,
                  tag="L0-warmup")

    if args.blink_eps > 0:
        print(f"  Blink warmup: {args.blink_eps} eps on Level 2 + walls")
        agent.epsilon = max(EPSILON_END, agent.epsilon * 0.5)
        run_stage(agent, OBELIX, frame_stack,
                  args.blink_eps, difficulty=2, wall_prob=args.wall_prob,
                  tag="L2-warmup")

    print(f"  Level 3: {args.episodes} eps  (moving + blinking box)\n")
    agent.epsilon = EPSILON_START
    run_stage(agent, OBELIX, frame_stack,
              args.episodes, difficulty=3, wall_prob=args.wall_prob,
              tag="L3+wall", save_best_as="weights_phase3_best.pth",
              box_speed=args.box_speed)

    agent.save("weights_phase3_final.pth")
    print("\n  Done.")
    print("  Submit: weights_phase3_best.pth + agent.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",    type=str,   required=True)
    ap.add_argument("--weights_file", type=str,   default="weights_lvl2_v3_best.pth")
    ap.add_argument("--episodes",     type=int,   default=2000)
    ap.add_argument("--warmup_eps",   type=int,   default=0)
    ap.add_argument("--blink_eps",    type=int,   default=0)
    ap.add_argument("--wall_prob",    type=float, default=0.70)
    ap.add_argument("--box_speed",    type=int,   default=BOX_SPEED)
    args = ap.parse_args()
    main(args)