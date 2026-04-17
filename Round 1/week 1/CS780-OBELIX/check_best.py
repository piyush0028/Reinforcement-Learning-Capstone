import numpy as np
from obelix import OBELIX
from agent import policy

env = OBELIX(scaling_factor=5, arena_size=500, max_steps=1000,
             wall_obstacles=True, difficulty=0, box_speed=0, seed=0)

scores = []
for seed in range(10):
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0
    done = False
    while not done:
        action = policy(obs, rng)
        obs, reward, done = env.step(action, render=False)
        total += reward
    scores.append(total)
    print(f"Seed {seed}: {total:.1f}")

print(f"\nMean: {np.mean(scores):.2f}")