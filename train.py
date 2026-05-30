import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import os
from src.agent import Agent, AgentTD3
from src.utils import get_state, to_tensor
import time
from collections import deque
import json
import imageio

def train(num_steps: int, env_name: str):

    env = gym.make(id=env_name)
    goal_dim = 29
    agent = Agent(env=env, goal_dim=goal_dim)
    device = agent.worker.device
    c = 10

    eval_every = 50000
    eval_num = 50
    reward_log = []
    step_log = []
    success_log = []
    step = 0
    best_reward = -torch.inf

    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path, f"checkpoints/hiro/{env_name}.pt")
    metric_path = os.path.join(root_path, f"metrics/hiro/{env_name}.json")
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)

    if os.path.exists(save_path):
        agent.load(save_path)
        print(f"Resumed from {save_path}")
    else:
        print(f"Starting warmup") # gather some random data for buffer 
        warmup_hiro(agent, env, goal_dim, c, num_steps = 10000)
        print(f"Finished warmup\n")

    if os.path.exists(metric_path):
        print(f"Resuming data from {metric_path}")
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        step_log = metrics['steps']
        reward_log = metrics['rewards']
        success_log = metrics["success_rate"]
        step = step_log[-1] # run from last eval
        best_reward = max(reward_log)
        print(f"Resumed from {save_path} on step {step}")

    while step < num_steps:

        obs, _ = env.reset()
        state = to_tensor(get_state(obs), device)
        goal = agent.manager.choose_action(state, training=True)

        state_seq, goal_seq, action_seq, reward_seq = [], [], [], []
        t = 0
        done = False
        exit_loop = False

        while not exit_loop and step < num_steps:

            action = agent.worker.choose_action(state, goal, training=True)
            next_obs, raw_reward, terminated, truncated, info = env.step(
                action.detach().cpu().numpy()
            )

            # using -norm from goal for manager rather than raw value
            nav_reward = -np.linalg.norm(
                next_obs['achieved_goal'] - next_obs['desired_goal']
            )

            next_state = to_tensor(get_state(next_obs), device)
            done = terminated
            exit_loop = terminated or truncated
            step += 1
            t += 1

            state_seq.append(state.cpu().numpy())
            goal_seq.append(goal.cpu().numpy())
            action_seq.append(action.detach().cpu().numpy())
            reward_seq.append(nav_reward)

            # h and intrinsic reward
            next_goal = state[:goal_dim] + goal - next_state[:goal_dim]
            intrinsic_reward = -torch.linalg.norm(next_goal)
            blended_reward = intrinsic_reward + 0.1 * torch.tensor(raw_reward)

            # worker store and update
            agent.worker.buffer.store(
                state, goal, action, blended_reward, next_state, next_goal, float(done)
            )
            agent.worker.update()

            # manager store and update every c steps or episode end
            if t % c == 0 and t > 0 or exit_loop:
                reward_sum = torch.tensor(reward_seq, dtype=torch.float32).sum()
                agent.manager.buffer.store(
                    state_seq=np.array(state_seq),
                    goal_seq=np.array(goal_seq),
                    action_seq=np.array(action_seq),
                    reward=reward_sum,
                    next_state=next_state,
                    done=float(done)
                )
                agent.manager.update(agent.worker)
                state_seq, goal_seq, action_seq, reward_seq = [], [], [], []

                if not done:
                    goal = agent.manager.choose_action(next_state, training=True)
            else:
                goal = next_goal

            if step % eval_every == 0:
                mean_reward, success_rate = eval_hiro(agent, eval_num, env_name, goal_dim)
                step_log.append(step)
                reward_log.append(mean_reward)
                success_log.append(success_rate)
                print(f"Step: {step} | reward: {mean_reward:.2f} | success rate: {success_rate:.2f}")

                metrics = {
                    "env_name" : env_name,
                    "steps" : step_log,
                    "rewards" : reward_log,
                    "success_rate" : success_log
                }

                with open(metric_path, 'w') as f:
                    json.dump(metrics, f)

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    videos_path = os.path.join(root_path, f"videos/hiro/{env_name}_{step}.gif")
                    os.makedirs(os.path.dirname(videos_path), exist_ok=True)
                    record_episode_hiro(agent, env_name, videos_path, goal_dim)
                    agent.save(save_path)

            state = next_state

def eval_hiro(agent, num_episodes: int, env_name: str, goal_dim: int):
    env = gym.make(id=env_name)
    device = agent.worker.device
    reward_log = []
    success_log = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = to_tensor(get_state(obs), device)
        goal = agent.manager.choose_action(state, training=False)
        terminated = truncated = False
        episode_reward = 0
        t = 0
        success = False

        while not (terminated or truncated):
            action = agent.worker.choose_action(state, goal, training=False)
            next_obs, _, terminated, truncated, info = env.step(
                action.detach().cpu().numpy()
            )
            next_state = to_tensor(get_state(next_obs), device)

            nav_reward = -np.linalg.norm(
                next_obs['achieved_goal'] - next_obs['desired_goal']
            )

            episode_reward += nav_reward
            t += 1

            if info.get('success', False):
                success = True

            next_goal = state[:goal_dim] + goal - next_state[:goal_dim]

            if t % 10 == 0 and not (terminated or truncated):
                goal = agent.manager.choose_action(next_state, training=False)
            else:
                goal = next_goal

            state = next_state

        reward_log.append(episode_reward)
        success_log.append(1 if success else 0)

    env.close()
    return np.mean(reward_log), np.mean(success_log)

def record_episode_hiro(agent, env_name: str, save_path: str, goal_dim: int):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = gym.make(id=env_name, render_mode="rgb_array")
    device = agent.worker.device
    frames = []

    obs, _ = env.reset()
    state = to_tensor(get_state(obs), device)
    goal = agent.manager.choose_action(state, training=False)
    terminated = truncated = False
    t = 0

    while not (terminated or truncated):
        action = agent.worker.choose_action(state, goal, training=False)
        next_obs, _, terminated, truncated, _ = env.step(
            action.detach().cpu().numpy()
        )
        next_state = to_tensor(get_state(next_obs), device)
        frames.append(env.render())
        t += 1

        next_goal = state[:goal_dim] + goal - next_state[:goal_dim]

        if t % 10 == 0 and not (terminated or truncated):
            goal = agent.manager.choose_action(next_state, training=False)
        else:
            goal = next_goal

        state = next_state

    imageio.mimsave(save_path, frames, fps=30)
    env.close()

def train_td3(num_steps: int, env_name: str):

    env = gym.make(id=env_name) 
    agent = AgentTD3(env)

    eval_every = int(50000)
    num_eval = 50
    
    step = 0
    best_reward = -torch.inf

    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path, f"checkpoints/td3/{env_name}.pt")
    metric_path = os.path.join(root_path, f"metrics/td3/{env_name}.json")

    reward_log = []
    step_log = []

    if os.path.exists(save_path):
        agent.load(save_path)
    
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    if os.path.exists(metric_path):
        print(f"Resuming data from {metric_path}")
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        step_log = metrics['steps']
        reward_log = metrics['rewards']
        step = step_log[-1] # run from last eval
        best_reward = max(reward_log)
        print(f"Resumed from {save_path} on step {step}")

    while step < num_steps:

        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated) and step < num_steps:

            action = agent.choose_action(state, training=True)

            next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            step += 1

            agent.buffer.store(state, action, reward, next_state, terminated)    

            agent.update()

            if step % eval_every == 0:
                mean_reward = eval_td3(agent, num_eval, env_name)
                step_log.append(step)
                reward_log.append(mean_reward)
                metrics = {
                    "env_name" : env_name,
                    "steps" : step_log,
                    "rewards" : reward_log
                }

                with open(metric_path, 'w') as f:
                    json.dump(metrics, f)
                print(f"Step: {step} | mean reward: {mean_reward:.3f}")
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    video_path = os.path.join(root_path, f"videos/td3/{env_name}_{step}.gif")
                    record_episode_td3(agent, env_name, video_path)

                    agent.save(save_path)

            state = next_state   

def eval_td3(agent, num_episodes: int, env_name: str):

    env = gym.make(id=env_name) 
    reward_log = []
    
    for i in range(num_episodes):

        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            episode_reward += reward
        
        reward_log.append(episode_reward)
    
    return np.mean(reward_log)

def record_episode_td3(agent, env_name, save_path, max_frames = 200):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    env = gym.make(id=env_name, render_mode="rgb_array")
    frames = []
    state, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated) and len(frames) < max_frames:
        action = agent.choose_action(state, training=False)
        state, _, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
        frames.append(env.render())
    imageio.mimsave(save_path, frames, fps=30)
    env.close()

def warmup_hiro(agent, env, goal_dim, c, num_steps=5000):
    obs, _ = env.reset()
    state = to_tensor(get_state(obs), agent.worker.device)
    goal = torch.zeros(goal_dim).to(agent.worker.device)  # dummy goal
    
    state_seq, goal_seq, action_seq, reward_seq = [], [], [], []
    t = 0

    for _ in range(num_steps):
        action = torch.tensor(env.action_space.sample(), dtype=torch.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        next_state = to_tensor(get_state(next_obs), agent.worker.device)
        done = terminated
        exit_loop = terminated or truncated

        next_goal = state[:goal_dim] + goal - next_state[:goal_dim]
        intrinsic_reward = -torch.linalg.norm(next_goal)

        agent.worker.buffer.store(state, goal, action, intrinsic_reward, next_state, next_goal, float(done))

        state_seq.append(state.cpu().numpy())
        goal_seq.append(goal.cpu().numpy())
        action_seq.append(action.numpy())
        reward_seq.append(reward)
        t += 1

        if t % c == 0:
            reward_sum = torch.tensor(reward_seq, dtype=torch.float32).sum()
            agent.manager.buffer.store(
                state_seq=np.array(state_seq),
                goal_seq=np.array(goal_seq),
                action_seq=np.array(action_seq),
                reward=reward_sum,
                next_state=next_state,
                done=float(done)
            )
            state_seq, goal_seq, action_seq, reward_seq = []  , [], [], []

        if exit_loop:
            obs, _ = env.reset()
            state = to_tensor(get_state(obs), agent.worker.device)
            goal = torch.zeros(goal_dim).to(agent.worker.device)
            state_seq, goal_seq, action_seq, reward_seq = [], [], [], []
            t = 0
        else:
            goal = next_goal
            state = next_state

def test_antmaze():
    """
    Used to debug received dense ant maze information
    """
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    
    env = gym.make("AntMaze_UMazeDense-v5")
    obs, _ = env.reset()
    
    print(f"unrwapped reward type: {env.unwrapped.reward_type}")
    print(f"Observation keys: {obs.keys()}")
    print(f"Desired goal: {obs['desired_goal']}")
    print(f"Achieved goal: {obs['achieved_goal']}")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Distance to goal: {np.linalg.norm(obs['desired_goal'] - obs['achieved_goal']):.3f}")
    
    print(f"\nTaking 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward: {reward}")
        print(f"expected nav reward: {-np.linalg.norm(next_obs['desired_goal'] - next_obs['achieved_goal']):.4f}")
        dist = np.linalg.norm(next_obs['desired_goal'] - next_obs['achieved_goal'])
        print(f"  Step {i+1} | reward: {reward:.4f} | distance: {dist:.3f} | info: {info}")
    
    env.close()

if __name__ == "__main__":
   
    # test_antmaze()

    num_steps = int(5e6)
    env_name = "AntMaze_UMazeDense-v5"
    start = time.perf_counter()
    train(num_steps, env_name)
    end = time.perf_counter()
    print(f"Time to train: {end-start:.2f}\n")
   
    """num_steps = int(2e6)

    td3_envs = [
       "Hopper-v5",
       "HalfCheetah-v5",
       "Ant-v5"
    ]

    for i, env in enumerate(td3_envs):
        print(f"Starting TD3 on {env}")
        start = time.perf_counter()
        train_td3(num_steps=num_steps, env_name=env)
        end = time.perf_counter()
        print(f"Finished TD3 on {env} in {end-start:.2f} seconds\n")"""