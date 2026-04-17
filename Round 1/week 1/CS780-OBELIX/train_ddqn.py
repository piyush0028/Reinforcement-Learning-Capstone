import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM = 18


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


class QNetwork(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train():
    EPISODES = 3000
    GAMMA = 0.99
    LR = 5e-4
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 50000
    MIN_BUFFER = 1000
    TARGET_UPDATE = 150
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.9985
    GRAD_CLIP = 1.0

    device = torch.device("cpu")

    env_wall = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=True,  difficulty=0, box_speed=0, seed=0,
    )
    env_open = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=False, difficulty=0, box_speed=0, seed=0,
    )

    online_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    epsilon = EPS_START
    total_steps = 0
    reward_history = deque(maxlen=100)
    best_avg = -float("inf")

    for episode in range(EPISODES):
        seed = random.randint(0, 100_000)
        env  = env_wall if episode % 2 == 0 else env_open
        state = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        while not done:
            total_steps += 1

            if random.random() < epsilon:
                action_idx = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    s_t        = torch.FloatTensor(state).unsqueeze(0)
                    action_idx = int(online_net(s_t).argmax(dim=1).item())

            next_state, reward, done = env.step(ACTIONS[action_idx], render=False)
            reward = np.clip(reward, -10, 200)
            buffer.push(state, action_idx, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(buffer) >= MIN_BUFFER:
                s, a, r, ns, d = buffer.sample(BATCH_SIZE)

                q_values = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    best_actions = online_net(ns).argmax(dim=1, keepdim=True)
                    q_next = target_net(ns).gather(1, best_actions).squeeze(1)
                    td_target = r + GAMMA * q_next * (1.0 - d)

                loss = F.smooth_l1_loss(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
                optimizer.step()

            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        reward_history.append(total_reward)
        avg_reward = float(np.mean(reward_history))

        if avg_reward > best_avg and len(reward_history) == 100:
            best_avg = avg_reward
            torch.save(online_net.state_dict(), "weights.pth")
            print(f"  *** New best avg={best_avg:.2f} — weights.pth saved")

        if episode % 100 == 0:
            tag = "wall" if episode % 2 == 0 else "open"
            print(f"Ep {episode:5d} [{tag}] | reward={total_reward:8.1f} | "
                  f"avg100={avg_reward:7.2f} | eps={epsilon:.3f} | buf={len(buffer)}")


if __name__ == "__main__":
    train()