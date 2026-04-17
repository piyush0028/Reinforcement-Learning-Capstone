# train_vgp.py
import numpy as np
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train():
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=True,
        difficulty=0,
        box_speed=0,
        seed=50
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    episodes = 2000
    gamma = 0.99
    
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0
        
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = policy(state_t)
            
       
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            
    
            next_state, reward, done = env.step(ACTIONS[action.item()], render=False)
            
           
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        print(f"Episode {episode}, Reward: {total_reward:.2f}")
    
    torch.save(policy.state_dict(), "weights_vgp_m3.pth")
    print("Training complete! Saved weights to weights_vgp_m3.pth")

if __name__ == "__main__":
    train()

  