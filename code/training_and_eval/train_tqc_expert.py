import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.suann_surrogate import SuANNSurrogate, SuANNDynamicsTrainer
from utils.tqc_agent import TQCAgent

DATASET_PATH = "results/tqc_data/tqc_expert_dataset.npz"
LOG_DIR = "results/logs/tqc_v2_expert"
SUANN_CKP_PATH = os.path.join(LOG_DIR, "suann_model.pth")
TQC_FINAL_CKP_PATH = os.path.join(LOG_DIR, "tqc_expert_checkpoint.pth")

def train_tqc_expert_deep(epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("=" * 75)
    print(f"  DEEP HIGH-CAPACITY TQC EXPERT MODEL TRAINING (Device: {device.upper()})")
    print(f"  Target Epochs: {epochs} | Log Directory: {LOG_DIR}")
    print("=" * 75)

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Expert dataset not found at {DATASET_PATH}. Run collect_tqc_expert_data.py first.")
        return

    # Load 100k expert dataset
    data = np.load(DATASET_PATH)
    images = torch.from_numpy(data['images']).permute(0, 3, 1, 2)  # (N, 3, 64, 64)
    vectors = torch.from_numpy(data['vectors'])                     # (N, 4)
    actions = torch.from_numpy(data['actions'])                     # (N,)
    rewards = torch.from_numpy(data['rewards'])                     # (N,)
    next_images = torch.from_numpy(data['next_images']).permute(0, 3, 1, 2)
    next_vectors = torch.from_numpy(data['next_vectors'])
    dones = torch.from_numpy(data['dones'])

    print(f"[INFO] Loaded {len(images)} transition samples from expert dataset.")

    # Convert discrete actions to one-hot for SuANN
    actions_one_hot = torch.zeros(len(actions), 8)
    actions_one_hot.scatter_(1, actions.unsqueeze(1), 1.0)

    # ── 1. TRAIN SUANN SURROGATE DYNAMICS MODEL (30 EPOCHS) ───────────────────
    print("\n[STEP 1/2] Training SuANN Surrogate Model (30 Epochs)...")
    suann_dataset = TensorDataset(vectors, actions_one_hot, next_vectors, rewards.unsqueeze(1))
    suann_loader = DataLoader(suann_dataset, batch_size=256, shuffle=True)
    
    suann_model = SuANNSurrogate().to(device)
    suann_trainer = SuANNDynamicsTrainer(suann_model, lr=1e-3, device=device)
    
    for epoch in range(1, 31):
        loss = suann_trainer.train_epoch(suann_loader)
        if epoch % 5 == 0 or epoch == 30:
            print(f"  SuANN Epoch {epoch:2d}/30 | Loss: {loss:.6f}")
            
    suann_trainer.save(SUANN_CKP_PATH)
    print(f"  [OK] Saved SuANN Model: {SUANN_CKP_PATH}")

    # ── 2. DEEP TQC AGENT TRAINING (200 EPOCHS) ──────────────────────────────
    print(f"\n[STEP 2/2] Deep Training Truncated Quantile Critics (TQC) Agent ({epochs} Epochs)...")
    tqc_dataset = TensorDataset(images, vectors, actions, rewards, next_images, next_vectors, dones)
    tqc_loader = DataLoader(tqc_dataset, batch_size=128, shuffle=True)
    
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

    # Learning rate schedulers for smooth convergence
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.critic_optimizer, T_max=epochs)
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.actor_optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        critic_losses = []
        actor_losses = []
        for img, vec, act, rew, n_img, n_vec, dn in tqc_loader:
            c_loss, a_loss = agent.train_step(img, vec, act, rew, n_img, n_vec, dn)
            critic_losses.append(c_loss)
            actor_losses.append(a_loss)
            
        critic_scheduler.step()
        actor_scheduler.step()

        avg_c = np.mean(critic_losses)
        avg_a = np.mean(actor_losses)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            current_lr = critic_scheduler.get_last_lr()[0]
            print(f"  TQC Epoch {epoch:3d}/{epochs} | Critic Loss: {avg_c:8.5f} | Actor Loss: {avg_a:8.5f} | LR: {current_lr:.6f}")
            
            # Save periodic checkpoint
            epoch_ckp_path = os.path.join(LOG_DIR, f"tqc_expert_epoch_{epoch}.pth")
            agent.save(epoch_ckp_path)

    # Save final model
    agent.save(TQC_FINAL_CKP_PATH)
    print("\n" + "=" * 75)
    print(f"  [SUCCESS] Deep TQC Expert Agent Saved to: {TQC_FINAL_CKP_PATH}")
    print("=" * 75)

if __name__ == "__main__":
    train_tqc_expert_deep(epochs=200)
