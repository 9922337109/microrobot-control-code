import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.suann_surrogate import SuANNSurrogate, SuANNDynamicsTrainer
from utils.tqc_agent import TQCAgent

DATASET_PATH = "results/tqc_data/suann_dataset.npz"
SUANN_CKP_PATH = "results/logs/tqc_model/suann_model.pth"
TQC_CKP_PATH = "results/logs/tqc_model/tqc_checkpoint.pth"

def train_tqc_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print(f"  TRAINING TQC + SUANN PIPELINE (Device: {device.upper()})")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at {DATASET_PATH}. Run collect_tqc_data.py first.")
        return

    # Load collected data
    data = np.load(DATASET_PATH)
    images = torch.from_numpy(data['images']).permute(0, 3, 1, 2)  # (N, 3, 64, 64)
    vectors = torch.from_numpy(data['vectors'])                     # (N, 4)
    actions = torch.from_numpy(data['actions'])                     # (N,)
    rewards = torch.from_numpy(data['rewards'])                     # (N,)
    next_images = torch.from_numpy(data['next_images']).permute(0, 3, 1, 2)
    next_vectors = torch.from_numpy(data['next_vectors'])
    dones = torch.from_numpy(data['dones'])

    # Convert discrete actions to one-hot for SuANN
    actions_one_hot = torch.zeros(len(actions), 8)
    actions_one_hot.scatter_(1, actions.unsqueeze(1), 1.0)

    # ── 1. TRAIN SUANN SURROGATE DYNAMICS MODEL ────────────────────────────────
    print("\n[STEP 1/2] Training SuANN Surrogate Model...")
    suann_dataset = TensorDataset(vectors, actions_one_hot, next_vectors, rewards.unsqueeze(1))
    suann_loader = DataLoader(suann_dataset, batch_size=256, shuffle=True)
    
    suann_model = SuANNSurrogate().to(device)
    suann_trainer = SuANNDynamicsTrainer(suann_model, lr=1e-3, device=device)
    
    for epoch in range(1, 11):
        loss = suann_trainer.train_epoch(suann_loader)
        if epoch % 2 == 0:
            print(f"  SuANN Epoch {epoch:2d}/10 | Loss: {loss:.6f}")
            
    suann_trainer.save(SUANN_CKP_PATH)
    print(f"  [OK] Saved SuANN Model: {SUANN_CKP_PATH}")

    # ── 2. TRAIN TQC AGENT (Option B: CNN + Vector State) ──────────────────────
    print("\n[STEP 2/2] Training Truncated Quantile Critics (TQC) Agent...")
    tqc_dataset = TensorDataset(images, vectors, actions, rewards, next_images, next_vectors, dones)
    tqc_loader = DataLoader(tqc_dataset, batch_size=64, shuffle=True)
    
    agent = TQCAgent(
        img_channels=3,
        vector_dim=4,
        action_dim=8,
        n_critics=5,
        n_quantiles=25,
        top_quantile_drop=5,
        lr=3e-4,
        device=device
    )

    epochs = 15
    for epoch in range(1, epochs + 1):
        critic_losses = []
        actor_losses = []
        for img, vec, act, rew, n_img, n_vec, dn in tqc_loader:
            c_loss, a_loss = agent.train_step(img, vec, act, rew, n_img, n_vec, dn)
            critic_losses.append(c_loss)
            actor_losses.append(a_loss)
            
        avg_c = np.mean(critic_losses)
        avg_a = np.mean(actor_losses)
        print(f"  TQC Epoch {epoch:2d}/{epochs} | Critic Loss: {avg_c:8.4f} | Actor Loss: {avg_a:8.4f}")

    agent.save(TQC_CKP_PATH)
    print("\n" + "=" * 60)
    print(f"  [SUCCESS] TQC Agent Checkpoint Saved: {TQC_CKP_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    train_tqc_pipeline()
