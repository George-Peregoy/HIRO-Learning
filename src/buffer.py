import torch
import torch.nn as nn
import numpy as np

class WorkerBuffer:
    """
    Replay buffer class.

    Parameters
    ----------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    device : torch.Device
        Device tensor operations are running on.

    Methods
    -------
    store(state, action, reward, next_state, done)
        Stores current step information.
    get(batch_size)
        Returns a random sample of step information.
    """

    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # store stuff
        self.buffer_size = int(200000)
        self.states = torch.zeros((self.buffer_size, state_dim))
        self.goals = torch.zeros((self.buffer_size, goal_dim))
        self.actions = torch.zeros((self.buffer_size, action_dim))
        self.rewards = torch.zeros((self.buffer_size, 1))
        self.next_states = torch.zeros((self.buffer_size, state_dim))
        self.next_goals = torch.zeros((self.buffer_size, goal_dim))
        self.dones = torch.zeros((self.buffer_size, 1))

        self.max_idx = 0 # num of stored states
        self.device = device

    def store(self, 
              state: torch.Tensor, 
              goal: torch.Tensor, 
              action: torch.Tensor, 
              reward: torch.Tensor,
              next_state: torch.Tensor, 
              next_goal: torch.Tensor,
              done):
        """
        Stores relavent data in respective list.

        Parameters
        ----------
        state : torch.Tensor
            State vector.
        action : torch.Tensor
            Action vector.
        reward : float
            Reward for action.
        next_state : torch.Tensor
            Next state vector.
        done : float
            1 if episode finished else 0.
        """
        idx = self.max_idx % self.buffer_size # wraps -> replaces old

        self.states[idx] = state.detach().clone()
        self.goals[idx] = goal.detach().clone()
        self.actions[idx] = action.detach().clone()
        self.rewards[idx] = reward.detach().clone()
        self.next_states[idx] = next_state.detach().clone()
        self.next_goals[idx] = next_goal.detach().clone()
        self.dones[idx] = torch.tensor(done, dtype=torch.float32)

        # increment max_idx
        self.max_idx += 1

    def get(self, batch_size: int):
        """
        Returns random sample from stored Tensors.

        Parameters
        ----------
        batch_size : int
            Size being sampled.

        Returns
        -------
        states : torch.Tensor
            Random sample of states. Size (batch_size, state_dim)
        actions : torch.Tensor
            Random sample of actions. Size (batch_size, action_dim)
        rewards : torch.Tensor
            Random sample of rewards. Size (batch_size, 1)
        next_states : torch.Tensor
            Random sample of rewards. Size (batch_size, state_dim)
        dones : torch.Tensor
            Random sample of dones. Size (batch_size, 1)
        """
        idx = np.random.randint(0, min(self.max_idx, self.buffer_size), batch_size)
        return(
            self.states[idx].to(self.device),
            self.goals[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.next_goals[idx].to(self.device),
            self.dones[idx].to(self.device)
        )
    
class ManagerBuffer:
    """
    Replay buffer class.

    Parameters
    ----------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    device : torch.Device
        Device tensor operations are running on.

    Methods
    -------
    store(state, action, reward, next_state, done)
        Stores current step information.
    get(batch_size)
        Returns a random sample of step information.
    """

    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, device, c: int):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.c = c
        
        # store stuff
        self.buffer_size = int(200000)
        self.state_seqs = torch.zeros((self.buffer_size, c, state_dim))
        self.goal_seqs = torch.zeros((self.buffer_size, c, goal_dim))
        self.action_seqs = torch.zeros((self.buffer_size, c, action_dim))
        self.rewards = torch.zeros((self.buffer_size, 1))
        self.next_states = torch.zeros((self.buffer_size, state_dim))
        self.dones = torch.zeros((self.buffer_size, 1))

        self.max_idx = 0 # num of stored states
        self.device = device

    def store(self, 
              state_seq: list, 
              goal_seq: list, 
              action_seq: list, 
              reward: torch.Tensor,
              next_state: torch.Tensor, 
              done
        ):
        """
        Stores relavent data in respective list.

        Parameters
        ----------
        state : torch.Tensor
            State vector.
        action : torch.Tensor
            Action vector.
        reward : float
            Reward for action.
        next_state : torch.Tensor
            Next state vector.
        done : float
            1 if episode finished else 0.
        """
        idx = self.max_idx % self.buffer_size # wraps -> replaces old

        self.state_seqs[idx] = torch.tensor(np.array(state_seq), dtype=torch.float32)
        self.goal_seqs[idx] = torch.tensor(np.array(goal_seq), dtype=torch.float32)
        self.action_seqs[idx] = torch.tensor(np.array(action_seq), dtype=torch.float32)
        self.rewards[idx] = reward.detach().clone()
        self.next_states[idx] = next_state.detach().clone()
        self.dones[idx] = torch.tensor(done, dtype=torch.float32)

       
        # increment max_idx
        self.max_idx += 1

    def get(self, batch_size: int):
        """
        Returns random sample from stored Tensors.

        Parameters
        ----------
        batch_size : int
            Size being sampled.

        Returns
        -------
        states : torch.Tensor
            Random sample of states. Size (batch_size, state_dim)
        actions : torch.Tensor
            Random sample of actions. Size (batch_size, action_dim)
        rewards : torch.Tensor
            Random sample of rewards. Size (batch_size, 1)
        next_states : torch.Tensor
            Random sample of rewards. Size (batch_size, state_dim)
        dones : torch.Tensor
            Random sample of dones. Size (batch_size, 1)
        """
        idx = np.random.randint(0, min(self.max_idx, self.buffer_size), batch_size)
        return(
            self.state_seqs[idx].to(self.device),
            self.goal_seqs[idx].to(self.device),
            self.action_seqs[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.dones[idx].to(self.device)
            )
    
class ReplayBuffer:
    """
    Replay buffer class. Used in TD3.

    Parameters
    ----------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    device : torch.Device
        Device tensor operations are running on.

    Attributes
    ----------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    buffer_size : int
        Max amount of stored steps.
    states : torch.Tensor
        Stored state vectors. Size (buffer_size, state_dim).
    actions : torch.Tensor
        Stored action vectors. Size (buffer_size, action_dim).
    rewards : torch.Tensor
        Stored rewards. Size (buffer_size, 1).
    next_states : torch.Tensor
        Stored next state vectors. Size (bufffer_size, state_dim).
    dones : torch.Tensor
        Stored dones. Size (buffer_size, 1).
    max_idx : int
        Max idx of stored information.
    device : torch.Device
        Device tensor operations are running on.

    Methods
    -------
    store(state, action, reward, next_state, done)
        Stores current step information.
    get(batch_size)
        Returns a random sample of step information.
    """

    def __init__(self, state_dim: int, action_dim: int, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # store stuff
        self.buffer_size = int(1e6)
        self.states = torch.zeros((self.buffer_size, state_dim))
        self.actions = torch.zeros((self.buffer_size, action_dim))
        self.rewards = torch.zeros((self.buffer_size, 1))
        self.next_states = torch.zeros((self.buffer_size, state_dim))
        self.dones = torch.zeros((self.buffer_size, 1))

        self.max_idx = 0 # num of stored states
        self.device = device

    def store(self, 
              state: np.ndarray | torch.Tensor, 
              action: np.ndarray | torch.Tensor, 
              reward: float, 
              next_state: np.ndarray | torch.Tensor, 
              done: float):
        """
        Stores relavent data in respective list.

        Parameters
        ----------
        state : torch.Tensor
            State vector.
        action : torch.Tensor
            Action vector.
        reward : float
            Reward for action.
        next_state : torch.Tensor
            Next state vector.
        done : float
            1 if episode finished else 0.
        """
        idx = self.max_idx % self.buffer_size # wraps - replaces old

        self.states[idx] = torch.tensor(state, dtype=torch.float32)
        self.actions[idx] = action.detach().clone().to(torch.float32)
        self.rewards[idx] = torch.tensor(reward, dtype=torch.float32)
        self.next_states[idx] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[idx] = torch.tensor(done, dtype=torch.float32)

        # increment max_idx
        self.max_idx += 1

    def get(self, batch_size: int):
        """
        Returns random sample from stored Tensors.

        Parameters
        ----------
        batch_size : int
            Size being sampled.

        Returns
        -------
        states : torch.Tensor
            Random sample of states. Size (batch_size, state_dim)
        actions : torch.Tensor
            Random sample of actions. Size (batch_size, action_dim)
        rewards : torch.Tensor
            Random sample of rewards. Size (batch_size, 1)
        next_states : torch.Tensor
            Random sample of rewards. Size (batch_size, state_dim)
        dones : torch.Tensor
            Random sample of dones. Size (batch_size, 1)
        """
        idx = np.random.randint(0, min(self.max_idx, self.buffer_size), batch_size)
        return(
            self.states[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.dones[idx].to(self.device)
        )