import torch

from models.mlp_mixer import MLPMixer


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create dummy input
x = torch.randn(32, 3, 32, 32).to(device)

# Initialize model
model = MLPMixer().to(device)

# Forward pass
output = model(x)

# Display results
print("Input Shape:", x.shape)

print("Output Shape:", output.shape)

print("Device:", output.device)

print(
    "Total Parameters:",
    sum(p.numel() for p in model.parameters())
)

print("MLP-Mixer successfully implemented!")