import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class SuANNSurrogate(nn.Module):
    """
    SuANN (Surrogate Artificial Neural Network) Dynamics Model.
    
    [Paper 4 — TQC+SuANN, Wiley AIS 2025]
    Learns environment transition dynamics (s_t, a_t) -> (delta_s_t, r_t)
    where state s_t = [agent_x, agent_y, target_x, target_y] normalized [0, 1].
    Action a_t = one-hot discrete action vector of shape (8,).
    
    Allows 1000x faster simulation during TQC offline training.
    """
    def __init__(self, state_dim=4, action_dim=8, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        
        # State delta network (predicts change in agent position)
        self.state_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # delta_agent_x, delta_agent_y
        )
        
        # Reward prediction network
        self.reward_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # predicted reward
        )

    def forward(self, state, action):
        """
        Forward pass predicting state transition and reward.
        state: Tensor (batch, 4) -> [agent_x, agent_y, target_x, target_y]
        action: Tensor (batch, 8) -> one-hot action
        returns: next_state (batch, 4), reward (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        delta_pos = self.state_net(x)
        reward = self.reward_net(x)
        
        # Next agent position = current agent position + delta_pos
        next_agent_pos = torch.clamp(state[:, :2] + delta_pos, 0.0, 1.0)
        # Target position remains unchanged in a transition step
        next_target_pos = state[:, 2:]
        
        next_state = torch.cat([next_agent_pos, next_target_pos], dim=-1)
        return next_state, reward

class SuANNDynamicsTrainer:
    """Trainer for SuANN Surrogate Model on PyGame Environment Transitions."""
    def __init__(self, surrogate_model, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = surrogate_model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.state_criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for states, actions, next_states, rewards in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)
            rewards = rewards.to(self.device)

            pred_next_states, pred_rewards = self.model(states, actions)
            
            loss_state = self.state_criterion(pred_next_states[:, :2], next_states[:, :2])
            loss_reward = self.reward_criterion(pred_rewards, rewards)
            loss = loss_state + loss_reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / max(1, len(dataloader))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
