import torch

from models.cnn import CNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

x = torch.randn(32, 3, 32, 32).to(device)

output = model(x)

print("Input Shape:", x.shape)

print("Output Shape:", output.shape)

print(
    "Total Parameters:",
    sum(p.numel() for p in model.parameters())
)

print("Device:", device)