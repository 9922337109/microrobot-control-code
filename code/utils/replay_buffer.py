import numpy as np
import torch
import os


class ReplayBuffer:
    """
    GPU-friendly circular replay buffer for online TQC training.

    Images are stored as uint8 (HWC) to save 4x memory vs float32.
    Sampling returns normalised float tensors (images /255) directly
    on the requested device, minimising CPU-GPU data transfers.

    Capacity: number of (s, a, r, s', done) transitions stored.
    """

    def __init__(
        self,
        capacity: int = 300_000,
        img_shape: tuple = (64, 64, 3),
        vector_dim: int = 4,
        device: str = 'cuda',
    ):
        self.capacity   = capacity
        self.device     = device
        self.img_shape  = img_shape
        self.vector_dim = vector_dim
        self.ptr        = 0
        self.size       = 0

        # Pre-allocate CPU arrays (uint8 images → 300K × 64×64×3 ≈ 3.7 GB max)
        self.images       = np.zeros((capacity, *img_shape),  dtype=np.uint8)
        self.vectors      = np.zeros((capacity, vector_dim),   dtype=np.float32)
        self.actions      = np.zeros((capacity,),              dtype=np.int64)
        self.rewards      = np.zeros((capacity,),              dtype=np.float32)
        self.next_images  = np.zeros((capacity, *img_shape),   dtype=np.uint8)
        self.next_vectors = np.zeros((capacity, vector_dim),   dtype=np.float32)
        self.dones        = np.zeros((capacity,),              dtype=np.float32)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        image:      np.ndarray,   # (H, W, C) uint8
        vector:     np.ndarray,   # (4,)  float32
        action:     int,
        reward:     float,
        next_image: np.ndarray,   # (H, W, C) uint8
        next_vector: np.ndarray,  # (4,)  float32
        done:       bool,
    ):
        idx = self.ptr
        self.images[idx]       = image
        self.vectors[idx]      = vector
        self.actions[idx]      = action
        self.rewards[idx]      = reward
        self.next_images[idx]  = next_image
        self.next_vectors[idx] = next_vector
        self.dones[idx]        = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ── Read ──────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch.
        Returns tensors on self.device; images are float32 normalised to [0, 1].
        """
        assert self.size >= batch_size, (
            f"Buffer has only {self.size} samples, requested {batch_size}."
        )
        idx = np.random.randint(0, self.size, size=batch_size)

        imgs      = (torch.from_numpy(self.images[idx])
                     .permute(0, 3, 1, 2)            # HWC → CHW
                     .to(self.device, dtype=torch.float32))    # already /255 in train_step
        vecs      = torch.from_numpy(self.vectors[idx]).to(self.device)
        acts      = torch.from_numpy(self.actions[idx]).to(self.device)
        rews      = torch.from_numpy(self.rewards[idx]).to(self.device)
        next_imgs = (torch.from_numpy(self.next_images[idx])
                     .permute(0, 3, 1, 2)
                     .to(self.device, dtype=torch.float32))
        next_vecs = torch.from_numpy(self.next_vectors[idx]).to(self.device)
        dones_t   = torch.from_numpy(self.dones[idx]).to(self.device)

        return imgs, vecs, acts, rews, next_imgs, next_vecs, dones_t

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save current buffer contents to a compressed .npz file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            images       = self.images[:self.size],
            vectors      = self.vectors[:self.size],
            actions      = self.actions[:self.size],
            rewards      = self.rewards[:self.size],
            next_images  = self.next_images[:self.size],
            next_vectors = self.next_vectors[:self.size],
            dones        = self.dones[:self.size],
            meta         = np.array([self.ptr, self.size]),
        )

    def load(self, path: str):
        """Restore buffer contents from a previously saved .npz file."""
        data = np.load(path)
        n = min(int(data['meta'][1]), self.capacity)
        self.images[:n]       = data['images'][:n]
        self.vectors[:n]      = data['vectors'][:n]
        self.actions[:n]      = data['actions'][:n]
        self.rewards[:n]      = data['rewards'][:n]
        self.next_images[:n]  = data['next_images'][:n]
        self.next_vectors[:n] = data['next_vectors'][:n]
        self.dones[:n]        = data['dones'][:n]
        self.size = n
        self.ptr  = int(data['meta'][0]) % self.capacity
        print(f"[ReplayBuffer] Loaded {n:,} transitions from {path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"ReplayBuffer(size={self.size:,}/{self.capacity:,}, "
                f"ptr={self.ptr}, device={self.device})")
