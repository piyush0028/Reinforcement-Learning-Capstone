from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX


SCALING_FACTOR    = 5
ARENA_SIZE        = 500
MAX_STEPS         = 1000
PHASE1_DIFFICULTY = 0
WALL_OBSTACLES    = True
BOX_SPEED         = 2

TOTAL_EPISODES  = 8_000
WARMUP_EPISODES = 300

BUFFER_SIZE = 100_000
BATCH_SIZE  = 256

GAMMA              = 0.99
LR                 = 3e-4
TAU                = 5e-3
TARGET_HARD_UPDATE = 500

EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY_EPISODES = 5_000

TRAIN_EVERY_BASE   = 4
TRAIN_EVERY_MATURE = 6

CURRICULUM_PHASE1_END = 1_500
CURRICULUM_PHASE2_END = 4_000

EVAL_EVERY = 250
EVAL_RUNS  = 10
SAVE_PATH  = "weights.pth"


OBS_DIM   = 18
N_ACTIONS = 5
ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int = OBS_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.value_head(z)
        a = self.adv_head(z)
        return v + a - a.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        if eps > 0 and random.random() < eps:
            return random.randrange(N_ACTIONS)
        t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(self(t).argmax(dim=-1).item())


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity  = capacity
        self.alpha     = alpha
        self._buf:  Deque[Transition] = collections.deque(maxlen=capacity)
        self._prio: Deque[float]      = collections.deque(maxlen=capacity)
        self._max_prio = 1.0

    def push(self, *transition: object, priority_boost: float = 1.0) -> None:
        prio = (self._max_prio * priority_boost) ** self.alpha
        self._buf.append(tuple(transition))
        self._prio.append(prio)

    def sample(
        self, n: int, beta: float = 0.4
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        prios = np.array(self._prio, dtype=np.float64)
        probs = prios / prios.sum()
        idxs  = np.random.choice(len(self._buf), size=n, replace=False, p=probs)

        weights = (len(self._buf) * probs[idxs]) ** (-beta)
        weights /= weights.max()

        batch = [self._buf[i] for i in idxs]
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        new_prios = (np.abs(td_errors) + 1e-6) ** self.alpha
        for i, p in zip(idxs, new_prios):
            self._prio[i] = float(p)
            self._max_prio = max(self._max_prio, float(p))

    def __len__(self) -> int:
        return len(self._buf)



def shape_reward(
    raw_reward: float,
    obs: np.ndarray,
    next_obs: np.ndarray,
    done: bool,
    enable_push: bool,
    env: OBELIX,
) -> float:
    shaped = raw_reward

    if not enable_push:
        forward_near = float(
            next_obs[5] + next_obs[7] + next_obs[9] + next_obs[11]
        )
        shaped += 0.5 * forward_near
        if next_obs[16] > 0:
            shaped += 2.0
        if next_obs[17] > 0:
            shaped -= 5.0
    else:
        cx = env.arena_size / 2.0
        cy = env.arena_size / 2.0
        dx = env.box_center_x - cx
        dy = env.box_center_y - cy
        dist     = math.sqrt(dx * dx + dy * dy)
        max_dist = cx * math.sqrt(2.0)
        progress = dist / max_dist
        shaped += 3.0 * progress
        if next_obs[17] > 0:
            shaped -= 10.0

    return shaped



def linear_eps(episode: int) -> float:
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def curriculum_seed(episode: int, rng: random.Random) -> int:
    if episode < CURRICULUM_PHASE1_END:
        return rng.randint(0, 19)
    elif episode < CURRICULUM_PHASE2_END:
        return rng.randint(0, 99)
    else:
        return rng.randint(0, 9_999)


def soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


def hard_update(online: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


def batch_to_tensors(
    batch: List[Transition], weights: np.ndarray
) -> Tuple[torch.Tensor, ...]:
    obs   = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
    acts  = torch.tensor([t[1] for t in batch],           dtype=torch.long)
    rews  = torch.tensor([t[2] for t in batch],           dtype=torch.float32)
    nobs  = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
    dones = torch.tensor([t[4] for t in batch],           dtype=torch.float32)
    ws    = torch.tensor(weights,                          dtype=torch.float32)
    return obs, acts, rews, nobs, dones, ws


def compute_ddqn_loss(
    online: DuelingDQN,
    target: DuelingDQN,
    batch: List[Transition],
    weights: np.ndarray,
) -> Tuple[torch.Tensor, np.ndarray]:
    obs, acts, rews, nobs, dones, ws = batch_to_tensors(batch, weights)

    with torch.no_grad():
        next_acts = online(nobs).argmax(dim=-1)
        next_q    = target(nobs).gather(1, next_acts.unsqueeze(1)).squeeze(1)
        td_target = rews + GAMMA * next_q * (1.0 - dones)

    current_q = online(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
    td_errors  = (td_target - current_q).detach().cpu().numpy()

    loss = (ws * F.smooth_l1_loss(current_q, td_target, reduction="none")).mean()
    return loss, td_errors


def make_env(seed: int = 0) -> OBELIX:
    return OBELIX(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        wall_obstacles=WALL_OBSTACLES,
        difficulty=PHASE1_DIFFICULTY,
        box_speed=BOX_SPEED,
        seed=seed,
    )


def evaluate(eval_env: OBELIX, online: DuelingDQN, n_runs: int = EVAL_RUNS) -> float:
    scores = []
    for i in range(n_runs):
        obs   = eval_env.reset(seed=1000 + i)
        total = 0.0
        done  = False
        while not done:
            action_idx = online.act(obs, eps=0.0)
            obs, reward, done = eval_env.step(ACTIONS[action_idx], render=False)
            total += reward
        scores.append(total)
    return float(np.mean(scores))



def main(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    py_rng = random.Random(42)

    env      = make_env(seed=0)
    eval_env = make_env(seed=0)

    online = DuelingDQN()
    target = DuelingDQN()
    hard_update(online, target)
    target.eval()

    if args.resume and os.path.exists(SAVE_PATH):
        online.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
        hard_update(online, target)
        print(f"[resume] Loaded weights from {SAVE_PATH}")

    optimizer = optim.Adam(online.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.episodes, eta_min=LR * 0.05
    )

    buffer = ReplayBuffer(BUFFER_SIZE)

    best_eval  = -float("inf")
    beta_start = 0.4
    beta_end   = 1.0
    ep_rewards = collections.deque(maxlen=100)
    t0         = time.time()

    print(
        f"\nOBELIX Dueling DDQN + PER  |  Phase-1"
        f"\nConfig : difficulty={PHASE1_DIFFICULTY}  wall_obstacles={WALL_OBSTACLES}"
        f"  sf={SCALING_FACTOR}  arena={ARENA_SIZE}  max_steps={MAX_STEPS}"
        f"\nPlan   : {args.episodes} episodes  |  warmup={WARMUP_EPISODES}"
        f"  |  batch={BATCH_SIZE}  |  buf={BUFFER_SIZE}"
        f"\nLR     : {LR} (cosine -> {LR*0.05:.2e})  |  gamma={GAMMA}"
        f"  |  tau={TAU}"
        f"\n{'─' * 72}"
    )

    for ep in range(1, args.episodes + 1):

        seed  = curriculum_seed(ep, py_rng)
        obs   = env.reset(seed=seed)
        eps   = linear_eps(ep)
        beta  = beta_start + (beta_end - beta_start) * min(1.0, ep / args.episodes)

        ep_raw    = 0.0
        done      = False
        push_seen = False

        while not done:
            if ep <= WARMUP_EPISODES:
                action_idx = random.randrange(N_ACTIONS)
            else:
                action_idx = online.act(obs, eps=eps)

            action_str             = ACTIONS[action_idx]
            next_obs, raw_reward, done = env.step(action_str, render=False)

            shaped = shape_reward(
                raw_reward, obs, next_obs, done, env.enable_push, env
            )

            boost = 1.0
            if done and env.enable_push and raw_reward >= 2000:
                boost = 12.0
            elif env.enable_push and not push_seen:
                boost = 4.0
                push_seen = True

            buffer.push(obs, action_idx, shaped, next_obs, done,
                        priority_boost=boost)
            obs     = next_obs
            ep_raw += raw_reward

        ep_rewards.append(ep_raw)

        if len(buffer) >= BATCH_SIZE and ep > WARMUP_EPISODES:
            n_updates = (
                TRAIN_EVERY_MATURE if len(buffer) >= 20_000
                else TRAIN_EVERY_BASE
            )
            for _ in range(n_updates):
                batch, idxs, w = buffer.sample(BATCH_SIZE, beta=beta)
                loss, td_err   = compute_ddqn_loss(online, target, batch, w)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                buffer.update_priorities(idxs, td_err)

            soft_update(online, target, TAU)
            scheduler.step()

        if ep % TARGET_HARD_UPDATE == 0:
            hard_update(online, target)

        if ep % EVAL_EVERY == 0:
            eval_score = evaluate(eval_env, online)
            elapsed    = (time.time() - t0) / 60.0
            avg100     = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            lr_now     = optimizer.param_groups[0]["lr"]

            print(
                f"Ep {ep:>5}/{args.episodes}  "
                f"eps={eps:.3f}  "
                f"avg100={avg100:>8.1f}  "
                f"eval={eval_score:>8.1f}  "
                f"buf={len(buffer):>7}  "
                f"lr={lr_now:.2e}  "
                f"time={elapsed:.1f}m"
            )

            if eval_score > best_eval:
                best_eval = eval_score
                torch.save(online.state_dict(), SAVE_PATH)
                print(f"  [*] New best {best_eval:.1f}  ->  {SAVE_PATH}")

    final_path = "weights_final.pth"
    torch.save(online.state_dict(), final_path)
    elapsed = (time.time() - t0) / 60.0
    print(
        f"\n{'─' * 72}"
        f"\nTraining complete in {elapsed:.1f} min"
        f"\nBest eval score : {best_eval:.1f}  ->  {SAVE_PATH}"
        f"\nFinal weights   : {final_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN trainer for OBELIX Phase-1")
    parser.add_argument(
        "--episodes", type=int, default=TOTAL_EPISODES,
        help=f"Training episodes (default: {TOTAL_EPISODES})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from existing weights.pth"
    )
    args = parser.parse_args()
    main(args)