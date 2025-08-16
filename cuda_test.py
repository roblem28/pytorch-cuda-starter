import torch, torch.nn as nn, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device, torch.cuda.get_device_name(0) if device=="cuda" else "")

X = torch.randn(1000, 20, device=device)
y = torch.randint(0, 2, (1000,), device=device)

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    opt.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

print("✅ CUDA training ran")
