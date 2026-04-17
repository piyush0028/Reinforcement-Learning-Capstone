"""
watch.py — Watch and record OBELIX agent performance
=====================================================
Opens a live window AND saves a .mp4 video file.

Usage:
    python watch.py                          # 3 episodes, difficulty 3
    python watch.py --difficulty 0 --runs 5 # Level 0, 5 episodes
    python watch.py --no_record              # watch only, no video saved

Output:
    agent_performance.mp4   — recorded video of all episodes
"""

import argparse
import importlib.util
import sys
import os
import numpy as np
import cv2

# ── Load agent ────────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("agent", "agent.py")
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)

# ── Load env ──────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from obelix import OBELIX


def run(args):
    rng    = np.random.default_rng(42)
    writer = None

    for ep in range(args.runs):
        env = OBELIX(
            scaling_factor = 5,
            arena_size     = 500,
            max_steps      = 2000,
            wall_obstacles = True,
            difficulty     = args.difficulty,
            box_speed      = 2,
            seed           = ep,
        )
        obs   = env.reset(seed=ep)
        done  = False
        total = 0.0
        step  = 0

        print(f"Episode {ep+1}/{args.runs} — difficulty={args.difficulty}")

        while not done:
            action = agent_mod.policy(obs, rng)
            obs, reward, done = env.step(action, render=True)
            total += reward
            step  += 1

            # Capture the rendered frame from the OpenCV window
            if not args.no_record:
                # env.frame is the rendered numpy array (H x W x 3)
                frame = env.frame.copy()

                # Add episode info overlay
                cv2.putText(
                    frame,
                    f"Ep {ep+1}  Step {step}  Reward {total:.0f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"Difficulty {args.difficulty}  "
                    f"{'PUSH' if env.enable_push else 'FIND'}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 100), 1, cv2.LINE_AA
                )

                # Initialise writer on first frame
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        args.output, fourcc, args.fps, (w, h)
                    )

                writer.write(frame)

            # Allow window to refresh
            key = cv2.waitKey(1)
            if key == ord("q"):
                print("Quit.")
                done = True

        status = "SUCCESS" if total > 1999 else "failed"
        print(f"  → steps={step}  reward={total:.1f}  {status}")

    cv2.destroyAllWindows()

    if writer is not None:
        writer.release()
        print(f"\nVideo saved: {args.output}")
    else:
        print("\nNo video recorded (--no_record was set).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty",  type=int,   default=3)
    ap.add_argument("--runs",        type=int,   default=5)
    ap.add_argument("--fps",         type=int,   default=60)
    ap.add_argument("--output",      type=str,   default="agent_performance.mp4")
    ap.add_argument("--no_record",   action="store_true")
    args = ap.parse_args()
    run(args)