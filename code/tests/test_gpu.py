import torch
import jax
print(f"PyTorch GPU Available: {torch.cuda.is_available()}")
print(f"PyTorch Device Name: {torch.cuda.get_device_name(0)}")
print(f"JAX Devices: {jax.devices()}")