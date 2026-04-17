# train_sarsa_lambda.py (fixed)
import numpy as np
import random
from collections import defaultdict
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


def obs_to_state(obs):
    bits = np.array(obs, dtype=int)
    powers = 1 << np.arange(len(bits))
    return int(np.sum(bits * powers))


def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, N_ACTIONS)
    q_vals = [Q[(state, a)] for a in range(N_ACTIONS)]
    return int(np.argmax(q_vals))


def train():
    EPISODES  = 5000       
    ALPHA     = 0.1
    GAMMA     = 0.99
    LAMBDA    = 0.9
    EPS_START = 1.0
    EPS_END   = 0.05
    EPS_DECAY = 0.9985      

    rng = np.random.default_rng(0)

    env_wall = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=True,  difficulty=0, box_speed=0, seed=0,
    )
    env_open = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=False, difficulty=0, box_speed=0, seed=0,
    )

    Q = defaultdict(float)
    epsilon = EPS_START
    reward_history = []
    best_avg = -float("inf")

    for episode in range(EPISODES):
        seed = random.randint(0, 100_000)
        env  = env_wall if episode % 2 == 0 else env_open
        obs  = env.reset(seed=seed)

        state  = obs_to_state(obs)
        action = epsilon_greedy(Q, state, epsilon, rng)
        E      = defaultdict(float)
        total_reward = 0.0

        while True:
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            next_state  = obs_to_state(next_obs)
            next_action = epsilon_greedy(Q, next_state, epsilon, rng)

            delta = (
                reward
                + GAMMA * Q[(next_state, next_action)] * (1 - done)
                - Q[(state, action)]
            )

            E[(state, action)] += 1.0

            for key in E:
                Q[key] += ALPHA * delta * E[key]
                E[key] *= GAMMA * LAMBDA

            state  = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        reward_history.append(total_reward)

        if len(reward_history) >= 100:
            avg = float(np.mean(reward_history[-100:]))
            if avg > best_avg:
                best_avg = avg
                np.save("weights_sarsa.npy", dict(Q))
                print(f"  *** New best avg={best_avg:.2f} — weights_sarsa.npy saved")

        if episode % 100 == 0:
            avg = float(np.mean(reward_history[-100:])) if len(reward_history) >= 100 else total_reward
            print(f"Ep {episode:5d} | reward={total_reward:8.1f} | avg100={avg:7.2f} | "
                  f"eps={epsilon:.3f} | Q-size={len(Q)}")

    np.save("weights_sarsa_final.npy", dict(Q))
    print(f"Done. Best avg={best_avg:.2f} | Q-table size={len(Q)}")


if __name__ == "__main__":
    train()