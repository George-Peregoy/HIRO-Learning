import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import os
from src.agent import Agent
from src.utils import get_state, to_tensor
import time
from collections import deque

def train(num_steps: int, env_name: str):

    env = gym.make(id=env_name)
    goal_dim = 29
    agent = Agent(env=env, goal_dim=goal_dim)
    device = agent.worker.device
    c = 10

    print_every = 5000
    reward_list = []
    step = 0
    best_success_rate = 0

    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path, f"checkpoints/{env_name}.pt")

    if os.path.exists(save_path):
        agent.load(save_path)
        print(f"Resumed from {save_path}")

    episode_successes = deque(maxlen=100)

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
            next_obs, reward, terminated, truncated, info = env.step(
                action.detach().cpu().numpy()
            )

            next_state = to_tensor(get_state(next_obs), device)
            done = terminated
            exit_loop = terminated or truncated
            step += 1
            t += 1

            if exit_loop:
                episode_successes.append(1 if info.get('success', False) else 0)

            # accumulate sequences — store numpy for efficiency
            state_seq.append(state.cpu().numpy())
            goal_seq.append(goal.cpu().numpy())
            action_seq.append(action.detach().cpu().numpy())
            reward_seq.append(reward)
            reward_list.append(reward)

            # h and intrinsic reward
            next_goal = state[:goal_dim] + goal - next_state[:goal_dim]
            intrinsic_reward = -torch.linalg.norm(next_goal)

            # worker store and update
            agent.worker.buffer.store(
                state, goal, action, intrinsic_reward, next_state, next_goal, float(done)
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

            # logging
            if step % print_every == 0:
                success_rate = np.mean(episode_successes) if episode_successes else 0
                print(f"Step: {step} | Success rate: {success_rate:.3f}")
                reward_list = []
                if success_rate >= best_success_rate:
                    best_success_rate = success_rate
                    agent.save(save_path)

            state = next_state

if __name__ == "__main__":

    num_steps = int(5e6)
    env_name = "AntMaze_UMazeDense-v5"
    start = time.perf_counter()
    train(num_steps, env_name)
    end = time.perf_counter()
    print(f"Time to train: {end-start}\n")