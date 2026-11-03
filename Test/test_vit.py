import torch
from models.vit import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT().to(device)

x = torch.randn(32, 3, 32, 32).to(device)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

total_params = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_params)
print("Device:", device)