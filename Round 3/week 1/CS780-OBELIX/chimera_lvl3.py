"""
CHIMERA TRAINER v4.1: ULTIMATE REFINEMENT
DDQN + PER + Ghost Tracking + Central Repulsion + Boundary Cushion + Sensor Reflexes
"""

from __future__ import annotations
import argparse
import random
import os
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(os.cpu_count() if os.cpu_count() else 4)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
FRAME_STACK = 32
OBS_DIM_RAW = 18
OBS_DIM = 20        
STATE_DIM = OBS_DIM * FRAME_STACK 

def augment_obs(raw_obs: np.ndarray, prev_reward: float, prev_ir: bool) -> np.ndarray:
    was_stuck  = 1.0 if prev_reward <= -199.0 else 0.0
    was_ir_hit = 1.0 if prev_ir else 0.0
    return np.append(raw_obs.astype(np.float32), [was_stuck, was_ir_hit])

class FrameStackWrapper:
    def __init__(self, env, k=32):
        self.env = env
        self.k = k
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_ir = False
        
    def reset(self, seed=None):
        raw_obs = self.env.reset(seed=seed)
        self.prev_reward = 0.0
        self.prev_ir = False
        obs = augment_obs(raw_obs, self.prev_reward, self.prev_ir)
        self.state[:] = np.tile(obs, self.k)
        return self.state.copy()
        
    def step(self, action, render=False):
        raw_obs, reward, done = self.env.step(action, render=render)
        obs = augment_obs(raw_obs, self.prev_reward, self.prev_ir)
        self.prev_reward = reward
        self.prev_ir = bool(raw_obs[16] > 0)
        
        self.state[:-OBS_DIM] = self.state[OBS_DIM:]
        self.state[-OBS_DIM:] = obs
        return self.state.copy(), reward, done

class DuelingDDQN(nn.Module):
    def __init__(self, input_dim=STATE_DIM, n_actions=5, hidden_dim=256):
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
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, p):
        idx = self.write + self.capacity - 1
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_batch(self, s_array):
        idx = np.zeros(len(s_array), dtype=np.int32)
        while True:
            active = idx < (self.capacity - 1)
            if not np.any(active): break
            left = 2 * idx + 1
            right = left + 1
            safe_left = np.where(active, left, 0)
            left_vals = self.tree[safe_left]
            mask = s_array > left_vals
            s_array = np.where(active & mask, s_array - left_vals, s_array)
            idx = np.where(active, np.where(mask, right, left), idx)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    @property
    def total_p(self): return self.tree[0]

class PrioritizedReplay:
    def __init__(self, cap: int = 500_000, alpha: float = 0.6):
        self.cap = cap
        self.alpha = alpha
        self.tree = SumTree(cap)
        self.max_priority = 1.0
        self.s_buf = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.a_buf = np.zeros(cap, dtype=np.int64)
        self.r_buf = np.zeros(cap, dtype=np.float32)
        self.s2_buf = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.d_buf = np.zeros(cap, dtype=np.float32)

    def add(self, s, a, r, s2, d):
        tree_idx = self.tree.write + self.tree.capacity - 1
        idx = self.tree.write
        self.s_buf[idx] = s
        self.a_buf[idx] = a
        self.r_buf[idx] = r
        self.s2_buf[idx] = s2
        self.d_buf[idx] = d
        p = self.max_priority ** self.alpha
        self.tree.add(p)
        return tree_idx

    def sample(self, batch: int, beta: float = 0.4):
        segment = self.tree.total_p / batch
        points = np.random.uniform(segment * np.arange(batch), segment * (np.arange(batch) + 1))
        batch_idx, priorities, data_idxs = self.tree.get_batch(points)
        probs = priorities / self.tree.total_p
        total = self.tree.n_entries
        weights = (total * probs) ** (-beta)
        weights /= weights.max()
        batch_weights = weights.astype(np.float32)
        s = self.s_buf[data_idxs]
        act = self.a_buf[data_idxs]
        r = self.r_buf[data_idxs]
        s2 = self.s2_buf[data_idxs]
        d = self.d_buf[data_idxs]
        return s, act, r, s2, d, batch_idx, batch_weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.max_priority = max(self.max_priority, prio)
            current_prio_alpha = self.tree.tree[idx]
            new_prio_alpha = prio ** self.alpha
            final_prio_alpha = max(new_prio_alpha, current_prio_alpha * 0.90)
            self.tree.update(idx, final_prio_alpha)

    def __len__(self): return self.tree.n_entries

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    # Default to Level 2 refinement settings
    ap.add_argument("--weights_file", type=str, default="weights_lvl2_v3.pth", help="File to load weights FROM") 
    ap.add_argument("--save_file", type=str, default="weights_lvl2_ULTIMATE.pth", help="File to save weights TO") 
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=2000) 
    ap.add_argument("--difficulty", type=int, default=2) # STAY ON LEVEL 2
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--wall_prob", type=float, default=0.70)
    ap.add_argument("--resume", action="store_true", help="Load existing weights")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4) # Lower LR for fine-tuning
    ap.add_argument("--batch", type=int, default=128) 
    ap.add_argument("--replay", type=int, default=500_000) 
    ap.add_argument("--warmup", type=int, default=5000) 
    ap.add_argument("--target_sync", type=int, default=1000) 
    ap.add_argument("--eps_start", type=float, default=0.20) # 20% EXPLORATION RESTART
    ap.add_argument("--eps_end", type=float, default=0.01) 
    ap.add_argument("--eps_decay_episodes", type=int, default=1000) 
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"🔥 CHIMERA PROTOCOL v4.1 (Ultimate Lvl 2 Refinement) | Optimized for: {device}")
    print(f"📥 Loading from: {args.weights_file}")
    print(f"💾 Saving to: {args.save_file}")

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDDQN(STATE_DIM, len(ACTIONS), 256).to(device)
    tgt = DuelingDDQN(STATE_DIM, len(ACTIONS), 256).to(device)

    steps = 0
    best_avg_score = -float('inf')

    if args.resume and os.path.exists(args.weights_file):
        print(f"Injecting Neural Network from {args.weights_file}...")
        ckpt = torch.load(args.weights_file, map_location=device, weights_only=True)
        state_dict = ckpt.get("online_net", ckpt)
        q.load_state_dict(state_dict)
            
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplay(args.replay, alpha=0.6)

    def eps_by_episode(current_ep):
        if current_ep >= args.eps_decay_episodes:
            return args.eps_end
        frac = current_ep / float(args.eps_decay_episodes)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    cumulative_total_reward = 0.0
    recent_returns = deque(maxlen=50) 
    recent_successes = deque(maxlen=50) 

    TRAIN_FREQ = 4

    def get_target_coords(bx, by, tx, ty, walls_on):
        if not walls_on:
            return tx, ty
        if tx > 260 and bx < 240:
            if bx < 210: return 200.0, 250.0  
            return 300.0, 250.0      
        elif tx < 240 and bx > 260:
            if bx > 290: return 300.0, 250.0  
            return 200.0, 250.0  
        return tx, ty 

    for ep in range(args.episodes):
        
        use_walls_this_episode = np.random.rand() < args.wall_prob
        current_difficulty = args.difficulty 

        is_eval = (ep > 0 and ep % 20 == 0)
        current_eps = 0.02 if is_eval else eps_by_episode(ep)

        base_env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=use_walls_this_episode,
            difficulty=current_difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        env = FrameStackWrapper(base_env, k=FRAME_STACK)
        
        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        ep_steps = 0
        
        # --- NEW LOGIC INITIALIZATIONS ---
        visited_grid = np.zeros((25, 25), dtype=bool) 
        ir_claimed = False
        ghost_tracking_timer = 0
        
        # --- TELEMETRY TRACKERS ---
        stuck_frames = 0
        steps_near_wall = 0
        ghost_active_frames = 0  
        
        bot_x, bot_y = base_env.bot_center_x, base_env.bot_center_y
        box_x, box_y = base_env.box_center_x, base_env.box_center_y
        
        start_bot_x = bot_x
        started_opposite = ((start_bot_x < 250 and box_x > 250) or (start_bot_x > 250 and box_x < 250))
        crossed_the_wall = False
        steps_to_cross = -1
        
        target_x, target_y = get_target_coords(bot_x, bot_y, box_x, box_y, use_walls_this_episode)
        prev_dist = math.hypot(bot_x - target_x, bot_y - target_y)
        prev_bound_dist = None 
        
        consecutive_turns = 0
        prev_box_x, prev_box_y = box_x, box_y
        prev_bot_x, prev_bot_y = bot_x, bot_y
        prev_action = "FW"

        for _ in range(args.max_steps):
            
            if np.random.rand() < current_eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                    a = q(s_t).argmax(dim=1).item()

            action_str = ACTIONS[a]
            s2, r, done = env.step(action_str, render=False)
            custom_r = float(r)
            
            bot_x, bot_y = base_env.bot_center_x, base_env.bot_center_y
            box_x, box_y = base_env.box_center_x, base_env.box_center_y
            
            is_native_stuck = bool(base_env.sensor_feedback[17] > 0)
            sensors_clear = not np.any(base_env.sensor_feedback[:17])
            current_ir = bool(base_env.sensor_feedback[16] > 0)
            recent_stuck = bool(s[-2] > 0) # augmented 'was_stuck' bit from previous frame
            
            # --- TELEMETRY UPDATES ---
            if is_native_stuck:
                stuck_frames += 1
            if use_walls_this_episode and 220 < bot_x < 280:
                steps_near_wall += 1
            if use_walls_this_episode and started_opposite and not crossed_the_wall:
                if (start_bot_x < 250 and bot_x > 290) or (start_bot_x > 250 and bot_x < 210):
                    crossed_the_wall = True
                    steps_to_cross = ep_steps 
                    
            # ---------------------------------------------------------
            # FIND PHASE EXCLUSIVE LOGIC
            # ---------------------------------------------------------
            if not base_env.enable_push:
                agent_moved = (abs(bot_x - prev_bot_x) > 0 or abs(bot_y - prev_bot_y) > 0)
                
                # 1. Stronger Grid Exploration (Bumped to 1.5)
                grid_x, grid_y = min(24, int(bot_x // 20)), min(24, int(bot_y // 20))
                if not visited_grid[grid_y, grid_x] and not is_native_stuck:
                    visited_grid[grid_y, grid_x] = True
                    custom_r += 1.5  
                
                # 2. Context-Aware Outer Boundary Repulsion (50px Cushion)
                edge_dist_curr = min(bot_x, base_env.frame_size[1] - bot_x, bot_y, base_env.frame_size[0] - bot_y)
                edge_dist_prev = min(prev_bot_x, base_env.frame_size[1] - prev_bot_x, prev_bot_y, base_env.frame_size[0] - prev_bot_y)
                
                if edge_dist_curr < 50.0:
                    if edge_dist_curr < edge_dist_prev:
                        custom_r -= 5.0  # Fear the outer wall while searching
                    elif edge_dist_curr > edge_dist_prev:
                        custom_r += 2.0  # Reward pulling back to the center
                        
                # 3. Reactive Sensor Steering (Snap-to-Target Reflexes)
                if not recent_stuck and not sensors_clear:
                    left_side_active = np.any(base_env.sensor_feedback[0:8])
                    right_side_active = np.any(base_env.sensor_feedback[8:16])
                    
                    if left_side_active and not right_side_active:
                        if action_str in ["L22", "L45"]: custom_r += 3.0
                        elif action_str in ["R22", "R45"]: custom_r -= 5.0
                    elif right_side_active and not left_side_active:
                        if action_str in ["R22", "R45"]: custom_r += 3.0
                        elif action_str in ["L22", "L45"]: custom_r -= 5.0

                # Basic distance shaping
                if sensors_clear:
                    if agent_moved:
                        progress = prev_dist - current_dist
                        custom_r += (progress * 1.5)  
                    else:
                        custom_r -= 2.0 
                prev_dist = current_dist
            
            # ---------------------------------------------------------
            # GLOBAL LOGIC (Applies everywhere)
            # ---------------------------------------------------------
            
            # 4. Ghost Tracking
            if not sensors_clear:
                ghost_tracking_timer = 30  
            else:
                if ghost_tracking_timer > 0:
                    ghost_tracking_timer -= 1  
                    ghost_active_frames += 1 

            # Anti-Wiggle Penalties
            if prev_action in ["L22", "L45"] and action_str in ["R22", "R45"]:
                custom_r -= 15.0  
            elif prev_action in ["R22", "R45"] and action_str in ["L22", "L45"]:
                custom_r -= 15.0

            # 5. Conditional Momentum
            if action_str == "FW" and prev_action == "FW":
                if not sensors_clear or ghost_tracking_timer > 0:
                    custom_r += 2.0   

            prev_action = action_str
            
            # Tamed Nuclear Turn Tracker 
            if action_str != "FW":
                consecutive_turns += 1
                if is_native_stuck:
                    if consecutive_turns > 3:
                        custom_r -= 50.0
                else:
                    if consecutive_turns >= 4:
                        custom_r -= 20.0
            else:
                consecutive_turns = 0 
                
            if is_native_stuck:
                custom_r += 150.0  
                if action_str != "FW" and consecutive_turns == 1:
                    custom_r += 30.0  

            # 6. Anti-Farming IR Lock
            if current_ir and not is_native_stuck and not ir_claimed:
                custom_r += 100.0  
                ir_claimed = True  

            target_x, target_y = get_target_coords(bot_x, bot_y, box_x, box_y, use_walls_this_episode)
            current_dist = math.hypot(bot_x - target_x, bot_y - target_y)

            # ---------------------------------------------------------
            # PUSH PHASE EXCLUSIVE LOGIC
            # ---------------------------------------------------------
            if base_env.enable_push:
                if action_str in ["L45", "R45"]:
                    custom_r -= 10.0  
                
                dist_x = min(box_x - 10, base_env.frame_size[1] - 10 - box_x)
                dist_y = min(box_y - 10, base_env.frame_size[0] - 10 - box_y)
                current_bound_dist = min(dist_x, dist_y)
                
                if prev_bound_dist is not None:
                    progress = prev_bound_dist - current_bound_dist
                    box_moved = (abs(box_x - prev_box_x) > 0 or abs(box_y - prev_box_y) > 0)
                    
                    if box_moved:
                        custom_r += (progress * 2.0)  
                        
                        # 7. Central Wall Repulsion Field
                        if use_walls_this_episode:
                            center_wall_x = base_env.frame_size[1] / 2.0
                            curr_dist_center = abs(box_x - center_wall_x)
                            prev_dist_center = abs(prev_box_x - center_wall_x)
                            
                            if curr_dist_center < prev_dist_center:
                                custom_r -= 4.0 
                            elif curr_dist_center > prev_dist_center:
                                custom_r += 2.0 
                    else:
                        custom_r -= 1.0 
                prev_bound_dist = current_bound_dist
                
            prev_bot_x, prev_bot_y = bot_x, bot_y
            prev_box_x, prev_box_y = box_x, box_y

            ep_ret += custom_r
            
            # --- PURE PER BUFFER LOGIC ---
            if not is_eval:
                replay.add(s, a, custom_r, s2, float(done))
                steps += 1
                
                if steps % TRAIN_FREQ == 0 and len(replay) >= max(args.warmup, args.batch):
                    beta = min(1.0, 0.4 + steps * (1.0 - 0.4) / 3_000_000)
                    sb, ab, rb, s2b, db, idx_b, weights_b = replay.sample(args.batch, beta=beta)
                    
                    sb_t = torch.as_tensor(sb, dtype=torch.float32)
                    ab_t = torch.as_tensor(ab, dtype=torch.int64)
                    rb_t = torch.as_tensor(rb, dtype=torch.float32)
                    s2b_t = torch.as_tensor(s2b, dtype=torch.float32)
                    db_t = torch.as_tensor(db, dtype=torch.float32)
                    weights_t = torch.as_tensor(weights_b, dtype=torch.float32)

                    with torch.no_grad():
                        next_q = q(s2b_t)
                        next_a = torch.argmax(next_q, dim=1)
                        next_q_tgt = tgt(s2b_t)
                        next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                        y = rb_t + args.gamma * (1.0 - db_t) * next_val

                    pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                    td_errors = torch.abs(pred - y).detach().numpy()
                    replay.update_priorities(idx_b, td_errors + 1e-5)

                    loss = (weights_t * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()

                    if steps % args.target_sync == 0:
                        tgt.load_state_dict(q.state_dict())
            
            s = s2
            ep_steps += 1
            if done: break

        is_success = 1 if ep_steps < args.max_steps else 0
        
        # --- VITALS DASHBOARD CALCULATIONS ---
        stuck_percent = (stuck_frames / max(1, ep_steps)) * 100.0
        ghost_percent = (ghost_active_frames / max(1, ep_steps)) * 100.0
        explored_percent = (np.sum(visited_grid) / 625.0) * 100.0

        lvl_str = f"Lvl {current_difficulty} | Wall {'ON' if use_walls_this_episode else 'OFF'}"
        if use_walls_this_episode:
            if started_opposite:
                cross_str = f"Crossed: YES ({steps_to_cross}s)" if crossed_the_wall else "Crossed: NO"
            else:
                cross_str = "Crossed: N/A"
        else:
            cross_str = "Wall: OFF"

        if is_eval:
            print(f"   >>> 🧪 [EVALUATION] {lvl_str} | Ret: {ep_ret:.1f} | Steps: {ep_steps} | Stuck: {stuck_percent:.1f}% | Exp: {explored_percent:.1f}% | {cross_str} <<<")
        else:
            recent_successes.append(is_success)
            cumulative_total_reward += ep_ret
            recent_returns.append(ep_ret)
            avg_50 = np.mean(recent_returns)
            win_rate_50 = (sum(recent_successes) / len(recent_successes)) * 100.0

            if (ep + 1) % 1 == 0:  
                print(f"Ep {ep+1:04d}/{args.episodes} | Ret: {ep_ret:7.1f} | Avg: {avg_50:7.1f} | Win: {win_rate_50:4.1f}% | eps: {current_eps:.3f} | Stuck: {stuck_percent:4.1f}% | WallTime: {steps_near_wall:3d} | Exp: {explored_percent:4.1f}% | Ghost: {ghost_percent:4.1f}% | {cross_str}")

            if ep >= 50 and avg_50 > best_avg_score:
                best_avg_score = avg_50
                best_file_name = args.save_file.replace(".pth", "_best.pth")
                torch.save(q.state_dict(), best_file_name)
                print(f"   --> 🌟 NEW HIGH SCORE: {best_avg_score:.1f}! Saved to {best_file_name}")

            if (ep + 1) % 50 == 0:
                torch.save(q.state_dict(), args.save_file)
                print(f"   --> 💾 Saved regular checkpoint to {args.save_file}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------
# EVALUATION POLICY (Codabench Ready)
# ---------------------------------------------------------
_MODEL = None
_EVAL_STATE = None 
_EVAL_STEP = 0 

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _MODEL, _EVAL_STATE, _EVAL_STEP
    
    if _MODEL is None:
        _MODEL = DuelingDDQN()
        submission_dir = os.path.dirname(__file__)
        wpath = os.path.join(submission_dir, "weights_lvl2_ULTIMATE_best.pth") 
        
        if os.path.exists(wpath):
            _MODEL.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        else:
            fallback_wpath = os.path.join(submission_dir, "weights_lvl2_ULTIMATE.pth")
            if os.path.exists(fallback_wpath):
                _MODEL.load_state_dict(torch.load(fallback_wpath, map_location="cpu", weights_only=True))
        _MODEL.eval()

    if _EVAL_STEP >= 2000:
        _EVAL_STATE = None
        _EVAL_STEP = 0
        
    _EVAL_STEP += 1

    aug_obs = augment_obs(obs, 0.0, bool(obs[16]>0))

    if _EVAL_STATE is None:
        _EVAL_STATE = np.tile(aug_obs, FRAME_STACK).astype(np.float32)
    else:
        _EVAL_STATE[:-OBS_DIM] = _EVAL_STATE[OBS_DIM:]
        _EVAL_STATE[-OBS_DIM:] = aug_obs
        
    x = torch.from_numpy(_EVAL_STATE).unsqueeze(0)
    with torch.no_grad():
        qs = _MODEL(x).squeeze(0).numpy()
    return ACTIONS[int(np.argmax(qs))]