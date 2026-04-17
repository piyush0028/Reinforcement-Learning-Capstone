"""
evaluate_local.py — Local evaluation across difficulty levels
=============================================================
Mirrors Codabench evaluation logic exactly.

Usage:
    python evaluate_local.py                       # uses agent.py
    python evaluate_local.py --agent my_agent.py   # any agent file
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX

# ── Must match Codabench ──────────────────────────────────────────────────────
SEEDS          = list(range(10))
MAX_STEPS      = 2000    # ← 1000, not 2000
SCALING_FACTOR = 5       # ← 5, not 1
ARENA_SIZE     = 500


def load_policy(agent_path: str):
    spec   = importlib.util.spec_from_file_location("submitted_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy, module


def evaluate(
    policy_fn,
    difficulty: int,
    wall_obstacles: bool,
    n_episodes: int = 10,
) -> tuple[float, float]:
    rewards = []

    for seed in SEEDS[:n_episodes]:
        # Reset agent frame buffer between episodes the same way Codabench does
        # (Codabench doesn't reset the module — we simulate that by calling
        #  policy with a zero obs first, which triggers the internal reset)
        rng = np.random.default_rng(seed)

        env = OBELIX(
            scaling_factor=SCALING_FACTOR,
            arena_size=ARENA_SIZE,
            max_steps=MAX_STEPS,
            difficulty=difficulty,
            wall_obstacles=wall_obstacles,
            seed=seed,
        )

        obs        = env.reset(seed=seed)
        done       = False
        cum_reward = 0.0

        while not done:
            action     = policy_fn(obs, rng)
            obs, reward, done = env.step(action, render=False)
            cum_reward += reward

        rewards.append(cum_reward)

    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="agent.py")
    args = parser.parse_args()

    agent_path = os.path.abspath(args.agent)
    if not os.path.exists(agent_path):
        print(f"Agent file not found: {agent_path}")
        sys.exit(1)

    policy_fn, _ = load_policy(agent_path)

    configs = [
        (0, True,  "Level 0 (static, wall)   "),
        (2, True,  "Level 2 (blink, wall)     "),
    ]

    print(f"\nEvaluating: {args.agent}")
    print(f"Config: sf={SCALING_FACTOR}  arena={ARENA_SIZE}  max_steps={MAX_STEPS}")
    print("─" * 58)
    print(f"{'Setting':<35} {'Mean Reward':>12} {'± Std':>8}")
    print("─" * 58)

    total_mean = 0.0
    for diff, walls, label in configs:
        mean, std = evaluate(policy_fn, diff, walls)
        total_mean += mean
        print(f"{label:<35} {mean:>12.1f} {std:>8.1f}")

    print("─" * 58)
    print(f"{'Average':<35} {total_mean / len(configs):>12.1f}")
    print()


if __name__ == "__main__":
    main()