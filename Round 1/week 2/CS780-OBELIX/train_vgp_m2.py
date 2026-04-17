import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, in_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def normalize_reward(r):
    return r / 2000.0

def train():
    env_open = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=False, difficulty=0, box_speed=0, seed=0,
    )
    env_wall = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=True, difficulty=0, box_speed=0, seed=0,
    )

    device = torch.device("cpu")
    policy   = PolicyNetwork().to(device)
    value_fn = ValueNetwork().to(device)

    opt_p = optim.Adam(policy.parameters(),   lr=3e-3)
    opt_v = optim.Adam(value_fn.parameters(), lr=3e-3)

    EPISODES      = 5000
    GAMMA         = 0.99
    BETA_START    = 0.05
    BETA_END      = 0.01
    WALL_START_EP = 2000

    reward_history = deque(maxlen=100)
    best_avg = -float("inf")

    for episode in range(EPISODES):
        seed = random.randint(0, 100_000)

        env = env_wall if episode >= WALL_START_EP else env_open
        obs = env.reset(seed=seed)

        BETA = BETA_START + (BETA_END - BETA_START) * (episode / EPISODES)

        states, log_probs, entropies, rewards = [], [], [], []
        total_reward = 0.0

        while True:
            s_t   = torch.FloatTensor(obs).unsqueeze(0)
            probs = policy(s_t)
            m     = torch.distributions.Categorical(probs)
            a     = m.sample()

            next_obs, reward, done = env.step(ACTIONS[a.item()], render=False)
            reward_norm = normalize_reward(reward)

            states.append(obs)
            log_probs.append(m.log_prob(a))
            entropies.append(m.entropy())
            rewards.append(reward_norm)

            obs           = next_obs
            total_reward += reward
            if done:
                break

        # Compute returns and normalize
        returns   = torch.FloatTensor(compute_returns(rewards, GAMMA))
        returns   = (returns - returns.mean()) / (returns.std() + 1e-8)

        states_t  = torch.FloatTensor(np.array(states))
        values    = value_fn(states_t)
        advantage = returns - values.detach()

        policy_loss = sum(
            -(lp * adv + BETA * ent)
            for lp, adv, ent in zip(log_probs, advantage, entropies)
        )
        opt_p.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_p.step()

        value_loss = F.mse_loss(values, returns)
        opt_v.zero_grad()
        value_loss.backward()
        opt_v.step()

        reward_history.append(total_reward)
        avg = float(np.mean(reward_history))

        if avg > best_avg and len(reward_history) == 100:
            best_avg = avg
            torch.save(policy.state_dict(), "weights.pth")
            print(f"  *** New best avg={best_avg:.2f} — weights.pth saved")

        if episode % 100 == 0:
            phase = "wall" if episode >= WALL_START_EP else "open"
            print(f"Ep {episode:5d} [{phase}] | reward={total_reward:8.1f} | "
                  f"avg100={avg:7.2f} | beta={BETA:.3f} | buf={len(reward_history)}")

    if not os.path.exists("weights.pth"):
        torch.save(policy.state_dict(), "weights.pth")
        print("Fallback save: weights.pth")

if __name__ == "__main__":
    train()