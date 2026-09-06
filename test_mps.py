import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

import torch
from ai.models.unet import UNet

device = torch.device("mps")
model = UNet(in_channels=5, num_classes=20).to(device)
model.eval()

# Dummy input
x = torch.randn(1, 5, 256, 256).to(device)
print("Running forward pass on MPS...")
try:
    with torch.no_grad():
        out = model(x)
    print("Success! Output shape:", out.shape)
except Exception as e:
    print("Error:", e)
