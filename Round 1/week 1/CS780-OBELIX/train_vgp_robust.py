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

class StateValueNetwork(nn.Module):
    def __init__(self, in_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1), 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
    value_net = StateValueNetwork().to(device) 
    
    optimizer_policy = optim.Adam(policy.parameters(), lr=0.001)
    optimizer_value = optim.Adam(value_net.parameters(), lr=0.001) 
    
    episodes = 2000
    gamma = 0.99
    beta = 0.01 
    
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        states = []      
        log_probs = []
        entropies = []   
        rewards = []
        total_reward = 0
        
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = policy(state_t)
            
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            log_prob = m.log_prob(action)
            entropy = m.entropy()
            
            next_state, reward, done = env.step(ACTIONS[action.item()], render=False)
            
            states.append(state)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        states_t = torch.FloatTensor(np.array(states)).to(device)
        values = value_net(states_t).squeeze()
        
        deltas = returns - values.detach() 
        
        policy_loss = []
        for log_prob, delta, entropy in zip(log_probs, deltas, entropies):
            step_loss = -(log_prob * delta + beta * entropy)
            policy_loss.append(step_loss)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        value_loss = F.mse_loss(values, returns)
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
        
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history)
        
        if total_reward > best_reward:
            best_reward = total_reward
            
        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | 100-Ep Avg: {avg_reward:.2f} | Best: {best_reward:.2f}")
    
    torch.save(policy.state_dict(), "weights_vgp_robust.pth")
    print(f"Training complete. Best reward: {best_reward:.2f}")

if __name__ == "__main__":
    train()

