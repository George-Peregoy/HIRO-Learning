import torch 
import torch.nn as nn
import os
from src.buffer import ManagerBuffer, WorkerBuffer
from src.network import Network
from src.utils import get_state_dim

class Agent:

    def __init__(self, env, goal_dim):
        self.state_dim = get_state_dim(env)
        self.action_dim = env.action_space.shape[0]
        self.goal_dim = goal_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"MOVING AGENT TO CUDA")
        
        self.worker = Worker(env, goal_dim)
        self.manager = Manager(env, goal_dim)

    def save(self, path: str = "checkpoints/hiro.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            # worker
            'worker_actor_live': self.worker.actor_live.state_dict(),
            'worker_actor_target': self.worker.actor_target.state_dict(),
            'worker_critic_live_1': self.worker.critic_live_1.state_dict(),
            'worker_critic_live_2': self.worker.critic_live_2.state_dict(),
            'worker_critic_target_1': self.worker.critic_target_1.state_dict(),
            'worker_critic_target_2': self.worker.critic_target_2.state_dict(),
            # manager
            'manager_actor_live': self.manager.actor_live.state_dict(),
            'manager_actor_target': self.manager.actor_target.state_dict(),
            'manager_critic_live_1': self.manager.critic_live_1.state_dict(),
            'manager_critic_live_2': self.manager.critic_live_2.state_dict(),
            'manager_critic_target_1': self.manager.critic_target_1.state_dict(),
            'manager_critic_target_2': self.manager.critic_target_2.state_dict(),
        }, path)
        print(f"Saved to {path}\n")

    def load(self, path: str = "checkpoints/hiro.pt"):
        checkpoint = torch.load(path, map_location=self.device)

        # worker
        self.worker.actor_live.load_state_dict(checkpoint['worker_actor_live'])
        self.worker.actor_target.load_state_dict(checkpoint['worker_actor_target'])
        self.worker.critic_live_1.load_state_dict(checkpoint['worker_critic_live_1'])
        self.worker.critic_live_2.load_state_dict(checkpoint['worker_critic_live_2'])
        self.worker.critic_target_1.load_state_dict(checkpoint['worker_critic_target_1'])
        self.worker.critic_target_2.load_state_dict(checkpoint['worker_critic_target_2'])
        # manager
        self.manager.actor_live.load_state_dict(checkpoint['manager_actor_live'])
        self.manager.actor_target.load_state_dict(checkpoint['manager_actor_target'])
        self.manager.critic_live_1.load_state_dict(checkpoint['manager_critic_live_1'])
        self.manager.critic_live_2.load_state_dict(checkpoint['manager_critic_live_2'])
        self.manager.critic_target_1.load_state_dict(checkpoint['manager_critic_target_1'])
        self.manager.critic_target_2.load_state_dict(checkpoint['manager_critic_target_2'])
        print(f"Loaded from {path}\n")

class Manager:

    def __init__(self, env, goal_dim):
        self.state_dim = get_state_dim(env)
        self.action_dim = env.action_space.shape[0]
        self.goal_dim = goal_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"MOVING AGENT TO CUDA")
        
        self.goal_scale_output = torch.tensor([
            10, 10,        # x, y
            0.5,           # z
            1, 1, 1, 1,    # quaternion
            *([1] * 8),    # joint angles
            *([1] * 14)    # remaining
        ], dtype=torch.float32)[:self.goal_dim].to(self.device)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.sigma = 1
        self.c = 10
        self.reward_scale = 0.1

        # define policy networks
        self.actor_live = Network(
            layer_sizes=[self.state_dim, 300, 300, self.goal_dim],
            lr = 1e-4,
            output_activation=nn.Tanh
        )

        self.critic_live_1 = Network(
            layer_sizes=[self.state_dim + self.goal_dim, 300, 300, 1],
            lr = 1e-3,
        )
        
        self.critic_live_2 = Network(
            layer_sizes=[self.state_dim + self.goal_dim, 300, 300, 1],
            lr = 1e-3,
        )

        # define target networks
        self.actor_target = Network(
            layer_sizes=[self.state_dim, 300, 300, self.goal_dim],
            lr = 1e-4,
            output_activation=nn.Tanh
        )

        self.critic_target_1 = Network(
            layer_sizes=[self.state_dim + self.goal_dim, 300, 300, 1],
            lr = 1e-3,
        )
        
        self.critic_target_2 = Network(
            layer_sizes=[self.state_dim + self.goal_dim, 300, 300, 1],
            lr = 1e-3,
        )
        self.actor_target.load_state_dict(self.actor_live.state_dict())
        self.critic_target_1.load_state_dict(self.critic_live_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_live_2.state_dict())

        # init buffer 
        self.buffer = ManagerBuffer(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim, device=self.device, c=self.c)

        # move to device 
        self.actor_live.to(self.device)
        self.critic_live_1.to(self.device)
        self.critic_live_2.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target_1.to(self.device)
        self.critic_target_2.to(self.device)

        self.policy_delay = 2
        self.update_count = 0 
        self.policy_noise = 0.2
        self.noise_clip = 0.5

    def choose_action(self, state, training: bool = True):
        
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            goal = self.actor_live(state)
            if training:
                goal += torch.randn_like(goal) * self.sigma
            goal = goal.clamp(-1, 1) 
            return goal[:self.goal_dim] * self.goal_scale_output
        
    def update(self, worker):

        if self.buffer.max_idx < self.batch_size:
            return
        
        state_seq, goal_seq, action_seqs, rewards, next_states, dones = self.buffer.get(batch_size=self.batch_size)

        # relabel g_t
        std = 0.5 * 0.5 * self.goal_scale_output # as in paper
        candidates = torch.zeros((self.batch_size, 10, self.goal_dim)).to(self.device)
        candidates[:, 0, :] = goal_seq[:, 0, :]
        candidates[:, 1, :] = (next_states[:, :self.goal_dim] - state_seq[:, 0, :self.goal_dim])
        candidates[:, 2:, :] = (torch.randn(self.batch_size, 8, self.goal_dim) * \
            std + (next_states[:, :self.goal_dim] - state_seq[:, 0, :self.goal_dim]).unsqueeze(1))
        
        # shape (batch_size, c, c, state_dim) -> 10 candidate sequences per candidate relabeled g
        candidate_seqs = torch.zeros((self.batch_size, 10, self.c, self.goal_dim)).to(self.device)
        candidate_seqs[:, :, 0, :] = candidates # shape (batch, 10, state_dim) 

        # prop init relabeled g using h
        for i in range(1, self.c): # prop through sequence
            candidate_seqs[:, :, i, :] = state_seq[:, i-1, :self.goal_dim].unsqueeze(1) \
                + candidate_seqs[:, :, i-1, :] - state_seq[:, i, :self.goal_dim].unsqueeze(1)

        # (batch, c, state_dim) ->  (batch, 1, c, state_dim) ->(batch, 10, c, state_dim) -> (batch*10*c, state_dim)
        s = state_seq.unsqueeze(1).expand(-1, 10, -1, -1).reshape(-1, self.state_dim)
        # (batch, 10, c, state_dim) -> (batch*10*c, state_dim)
        g = candidate_seqs.reshape(-1, self.goal_dim) 
        
        # (batch*10*c, action_dim) -> (batch, 10, c, action_dim)
        with torch.no_grad():
            pred_actions = worker.actor_live(torch.cat([s, g], dim=-1)).reshape(self.batch_size, 10, self.c, self.action_dim)
        # (batch, c, action_dim) -> (batch, 10, c, action_dim)
        a = action_seqs.unsqueeze(1).expand(-1, 10, -1, -1)
        
        # (batch, 10, c, action_dim) -> (batch, 10)
        score = ((a - pred_actions)**2).mean(dim=(2,3)) 
        # (batch,)
        best_idx = score.argmin(dim=1)  
        # (batch, c, state_dim) -> (batch, state_dim)
        goals = candidates[torch.arange(self.batch_size), best_idx, :]  

        # normal TD3 update 
        
        states = state_seq[:, 0, :]

        # get target values for bellman 
        new_actions = self.actor_live(states) * self.goal_scale_output
        next_target_actions = self.actor_target(next_states) * self.goal_scale_output
        noise = torch.randn_like(next_target_actions) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_target_actions = (next_target_actions + noise).clamp(-self.goal_scale_output, self.goal_scale_output)
        
        Q1_target = self.critic_target_1(torch.cat([next_states, next_target_actions], dim=-1))
        Q2_target = self.critic_target_2(torch.cat([next_states, next_target_actions], dim=-1))

        y = (self.reward_scale * rewards + self.gamma * (1 - dones) * torch.minimum(Q1_target, Q2_target)).detach()

        q1 = self.critic_live_1(torch.cat([states, goals], dim=-1))
        q2 = self.critic_live_2(torch.cat([states, goals], dim=-1))

        # get loss
        critic_loss_1 = nn.functional.mse_loss(q1, y)
        critic_loss_2 = nn.functional.mse_loss(q2, y)

        self.update_count += 1

        # update critics always
        self.critic_live_1.update(loss=critic_loss_1)
        self.critic_live_2.update(loss=critic_loss_2)

        if self.update_count % self.policy_delay == 0:
            # freeze critics
            for p in self.critic_live_1.parameters():
                p.requires_grad = False
            for p in self.critic_live_2.parameters():
                p.requires_grad = False

            actor_loss = self.critic_live_1(torch.cat([states, new_actions], dim=-1)).mean()  # manager

            for p in self.critic_live_1.parameters():
                p.requires_grad = True
            for p in self.critic_live_2.parameters():
                p.requires_grad = True

            self.actor_live.update(loss=-actor_loss)

            # soft update all targets
            for target_param, live_param in zip(self.actor_target.parameters(), self.actor_live.parameters()):
                target_param.data.copy_(self.tau * live_param.data + (1 - self.tau) * target_param.data)

        # soft update critic targets always
        for target_param, live_param in zip(self.critic_target_1.parameters(), self.critic_live_1.parameters()):
            target_param.data.copy_(self.tau * live_param.data + (1 - self.tau) * target_param.data)
        for target_param, live_param in zip(self.critic_target_2.parameters(), self.critic_live_2.parameters()):
            target_param.data.copy_(self.tau * live_param.data + (1 - self.tau) * target_param.data)

class Worker:

    def __init__(self, env, goal_dim):
        self.state_dim = get_state_dim(env)  
        self.action_dim = env.action_space.shape[0]
        self.goal_dim = goal_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256 # didn't see in paper -> defaulting to 256
        self.sigma = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"MOVING AGENT TO CUDA")

        # define policy networks
        self.actor_live = Network(
            layer_sizes=[self.state_dim + self.goal_dim, 300, 300, self.action_dim],
            lr = 1e-4,
            output_activation=nn.Tanh
        )

        self.critic_live_1 = Network(
            layer_sizes=[self.state_dim + self.goal_dim + self.action_dim, 300, 300, 1],
            lr = 1e-3,
        )
        
        self.critic_live_2 = Network(
            layer_sizes=[self.state_dim + self.goal_dim + self.action_dim, 300, 300, 1],
            lr = 1e-3,
        )

        # define target networks
        self.actor_target = Network(
            layer_sizes=[self.state_dim + self.goal_dim, 300, 300, self.action_dim],
            lr = 1e-4,
            output_activation=nn.Tanh
        )

        self.critic_target_1 = Network(
            layer_sizes=[self.state_dim + self.goal_dim + self.action_dim, 300, 300, 1],
            lr = 1e-3,
        )
        
        self.critic_target_2 = Network(
            layer_sizes=[self.state_dim + self.goal_dim + self.action_dim, 300, 300, 1],
            lr = 1e-3,
        )

        self.actor_target.load_state_dict(self.actor_live.state_dict())
        self.critic_target_1.load_state_dict(self.critic_live_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_live_2.state_dict())

        # init buffer 
        self.buffer = WorkerBuffer(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim, device=self.device)

        # move to device 
        self.actor_live.to(self.device)
        self.critic_live_1.to(self.device)
        self.critic_live_2.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target_1.to(self.device)
        self.critic_target_2.to(self.device)

        self.policy_delay = 2
        self.update_count = 0 
        self.policy_noise = 0.2
        self.noise_clip = 0.5

    def choose_action(self, state, goal, grad: bool = False, training: bool = True):
        
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not isinstance(goal, torch.Tensor):
            goal = torch.tensor(goal, dtype=torch.float32).to(self.device)

        with torch.enable_grad() if grad else torch.no_grad():

            input = torch.cat([state, goal], dim=-1)
            action = self.actor_live(input)

            if training:
                action = (action + torch.randn_like(action) * self.sigma).clamp(-1, 1)
            else:
                action = action.clamp(-1, 1)
            
            return action

    def update(self):
        
        if self.buffer.max_idx < self.batch_size:
            return
        
        # pull from buffer
        states, goals, actions, rewards, next_states, next_goals, dones = self.buffer.get(self.batch_size)
        
        # get target values for bellman
        new_actions = self.actor_live(torch.cat([states, goals], dim=-1))
        next_target_actions = self.actor_target(torch.cat([next_states, next_goals], dim=-1))
        noise = torch.randn_like(next_target_actions) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_target_actions = (next_target_actions + noise).clamp(-1, 1)

        Q1_target = self.critic_target_1(torch.cat([next_states, next_goals, next_target_actions], dim=-1))
        Q2_target = self.critic_target_2(torch.cat([next_states, next_goals, next_target_actions], dim=-1))

        y = (rewards + self.gamma * (1 - dones) * torch.minimum(Q1_target, Q2_target)).detach()

        q1 = self.critic_live_1(torch.cat([states, goals, actions], dim=-1))
        q2 = self.critic_live_2(torch.cat([states, goals, actions], dim=-1))

        # get loss
        critic_loss_1 = nn.functional.mse_loss(q1, y)
        critic_loss_2 = nn.functional.mse_loss(q2, y)

        self.update_count += 1

        # update critics always
        self.critic_live_1.update(loss=critic_loss_1)
        self.critic_live_2.update(loss=critic_loss_2)

        if self.update_count % self.policy_delay == 0:
            # freeze critics
            for p in self.critic_live_1.parameters():
                p.requires_grad = False
            for p in self.critic_live_2.parameters():
                p.requires_grad = False

            actor_loss = self.critic_live_1(torch.cat([states, goals, new_actions], dim=-1)).mean()  # worker

            for p in self.critic_live_1.parameters():
                p.requires_grad = True
            for p in self.critic_live_2.parameters():
                p.requires_grad = True

            self.actor_live.update(loss=-actor_loss)

            # soft update all targets
            for target_param, live_param in zip(self.actor_target.parameters(), self.actor_live.parameters()):
                target_param.data.copy_(self.tau * live_param.data + (1 - self.tau) * target_param.data)

        # soft update critic targets always
        for target_param, live_param in zip(self.critic_target_1.parameters(), self.critic_live_1.parameters()):
            target_param.data.copy_(self.tau * live_param.data + (1 - self.tau) * target_param.data)
        for target_param, live_param in zip(self.critic_target_2.parameters(), self.critic_live_2.parameters()):
            target_param.data.copy_(self.tau * live_param.data + (1 - self.tau) * target_param.data)