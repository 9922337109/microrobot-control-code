import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class CNNVectorEncoder(nn.Module):
    """
    Option B Encoder:
    Combines a CNN for 64x64 RGB visual observations with a vector MLP
    for agent and target positions [agent_x, agent_y, target_x, target_y].
    """
    def __init__(self, img_channels=3, vector_dim=4, embedding_dim=320):
        super().__init__()
        # CNN Encoder for 64x64 RGB images
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),  # -> (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),             # -> (64, 6, 6)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),             # -> (64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                           # -> 1024
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        
        # Vector MLP Encoder for [agent_pos (2), target_pos (2)]
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fusion = nn.Linear(256 + 64, embedding_dim)

    def forward(self, image, vector):
        """
        image: (batch, 3, 64, 64) normalized [0, 1]
        vector: (batch, 4) [agent_x, agent_y, target_x, target_y]
        """
        cnn_feat = self.cnn(image)
        vec_feat = self.vector_mlp(vector)
        combined = torch.cat([cnn_feat, vec_feat], dim=-1)
        embedding = F.relu(self.fusion(combined))
        return embedding


class QuantileCritic(nn.Module):
    """
    Quantile Critic outputting N_QUANTILES predictions per action.
    [Paper 4 — Salehi et al., Wiley AIS 2025]
    """
    def __init__(self, embedding_dim=320, action_dim=8, n_quantiles=25):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * n_quantiles)
        )

    def forward(self, embedding):
        """
        returns: (batch_size, action_dim, n_quantiles)
        """
        batch_size = embedding.shape[0]
        out = self.net(embedding)
        return out.view(batch_size, self.action_dim, self.n_quantiles)


class Actor(nn.Module):
    """Discrete Actor network outputting action logits for 8 directional movements."""
    def __init__(self, embedding_dim=320, action_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, embedding):
        logits = self.net(embedding)
        return F.softmax(logits, dim=-1)

    def get_action(self, embedding, deterministic=False):
        probs = self.forward(embedding)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), probs


class TQCAgent(nn.Module):
    """
    Truncated Quantile Critics (TQC) Agent with Option B (CNN + Vector state encoder).
    
    Features:
    - 5 Critic Networks (N_CRITICS=5) each outputting 25 Quantiles (N_QUANTILES=25).
    - Truncates top TOP_QUANTILE_DROP (e.g. 5) quantiles across ensemble to prevent overestimation.
    - Epistemic Uncertainty Estimation (quantile spread) for Adaptive MPC Shield scaling.
    """
    def __init__(
        self,
        img_channels=3,
        vector_dim=4,
        action_dim=8,
        n_critics=5,
        n_quantiles=25,
        top_quantile_drop=5,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.top_quantile_drop = top_quantile_drop
        self.total_target_quantiles = (n_critics * n_quantiles) - top_quantile_drop
        self.gamma = gamma
        self.tau = tau

        # Shared Encoder & Actor
        self.encoder = CNNVectorEncoder(img_channels, vector_dim).to(device)
        self.actor = Actor(action_dim=action_dim).to(device)

        # Ensemble of Critics & Target Critics
        self.critics = nn.ModuleList([
            QuantileCritic(action_dim=action_dim, n_quantiles=n_quantiles)
            for _ in range(n_critics)
        ]).to(device)
        
        self.target_critics = nn.ModuleList([
            QuantileCritic(action_dim=action_dim, n_quantiles=n_quantiles)
            for _ in range(n_critics)
        ]).to(device)
        
        # Copy initial weights to target critics
        for c, tc in zip(self.critics, self.target_critics):
            tc.load_state_dict(c.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.critics.parameters()), lr=lr
        )

        # Cumulative quantile midpoints for loss computation: shape (1, n_quantiles, 1)
        quantiles = (torch.arange(n_quantiles, device=device).float() + 0.5) / n_quantiles
        self.quantiles = quantiles.view(1, n_quantiles, 1)

    def select_action(self, image, vector, deterministic=False):
        """
        Select action given image observation (64x64x3) and vector state [agent_x, agent_y, target_x, target_y].
        """
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            img_t = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            vec_t = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
            embed = self.encoder(img_t, vec_t)
            action, _, _ = self.actor.get_action(embed, deterministic=deterministic)
        return action.item()

    def estimate_uncertainty(self, image, vector):
        """
        Computes epistemic uncertainty (quantile spread across critics)
        used by MPCSafetyShield for adaptive safety margin scaling [Paper 4].
        """
        self.encoder.eval()
        for c in self.critics: c.eval()
        with torch.no_grad():
            img_t = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            vec_t = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
            embed = self.encoder(img_t, vec_t)
            
            # Gather all quantile predictions for all critics across all actions
            all_quantiles = []
            for c in self.critics:
                q = c(embed)  # (1, action_dim, n_quantiles)
                all_quantiles.append(q)
            
            # Concatenate along quantile dimension -> (1, action_dim, n_critics * n_quantiles)
            stacked = torch.cat(all_quantiles, dim=-1)
            uncertainty = torch.std(stacked, dim=-1).mean().item()
        return uncertainty

    def train_step(self, images, vectors, actions, rewards, next_images, next_vectors, dones):
        """
        Performs one gradient update step for Critics and Actor.
        """
        self.encoder.train()
        self.actor.train()
        for c in self.critics: c.train()

        img_t = images.to(self.device) / 255.0
        vec_t = vectors.to(self.device)
        act_t = actions.to(self.device).long()
        rew_t = rewards.to(self.device).unsqueeze(-1)
        next_img_t = next_images.to(self.device) / 255.0
        next_vec_t = next_vectors.to(self.device)
        done_t = dones.to(self.device).unsqueeze(-1)

        embed = self.encoder(img_t, vec_t)
        with torch.no_grad():
            next_embed = self.encoder(next_img_t, next_vec_t)
            next_probs = self.actor(next_embed)
            next_actions = torch.argmax(next_probs, dim=-1)

            # Target quantiles calculation across all ensemble critics
            target_quantiles = []
            for tc in self.target_critics:
                q_val = tc(next_embed)  # (batch, action_dim, n_quantiles)
                # Select target action quantiles
                q_act = q_val[torch.arange(len(q_val)), next_actions]  # (batch, n_quantiles)
                target_quantiles.append(q_act)
            
            # Sort all quantiles across ensemble: (batch, n_critics * n_quantiles)
            stacked_targets = torch.cat(target_quantiles, dim=1)
            sorted_targets, _ = torch.sort(stacked_targets, dim=1)
            
            # Truncate top-k highest quantiles (risk-averse truncation)
            truncated_targets = sorted_targets[:, :self.total_target_quantiles]
            
            # Bellman update target: Y = reward + gamma * (1 - done) * quantile
            y_target = rew_t + (1.0 - done_t) * self.gamma * truncated_targets  # (batch, total_target_quantiles)

        # --- Train Critics ---
        # Current predictions for chosen actions: (batch, n_quantiles)
        critic_losses = []
        for critic in self.critics:
            current_q = critic(embed)  # (batch, action_dim, n_quantiles)
            current_q_act = current_q[torch.arange(len(current_q)), act_t]  # (batch, n_quantiles)
            
            # Pairwise diff between target quantiles and predicted quantiles
            # y_target: (batch, 1, total_target_quantiles), current_q_act: (batch, n_quantiles, 1)
            diff = y_target.unsqueeze(1) - current_q_act.unsqueeze(2)  # (batch, n_quantiles, total_target_quantiles)
            
            # Quantile Huber Loss
            huber_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none')
            weight = torch.abs(self.quantiles - (diff < 0).float())
            critic_loss = (weight * huber_loss).mean()
            critic_losses.append(critic_loss)

        total_critic_loss = sum(critic_losses)
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        # --- Train Actor ---
        embed_detached = embed.detach()
        probs = self.actor(embed_detached)
        
        # Soft Q-expectation actor loss
        with torch.no_grad():
            q_means = []
            for critic in self.critics:
                q_vals = critic(embed_detached).mean(dim=-1)  # (batch, action_dim)
                q_means.append(q_vals)
            avg_q = torch.stack(q_means, dim=0).mean(dim=0)   # (batch, action_dim)

        actor_loss = -(probs * avg_q).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critics
        for c, tc in zip(self.critics, self.target_critics):
            for param, target_param in zip(c.parameters(), tc.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return total_critic_loss.item(), actor_loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
