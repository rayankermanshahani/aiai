# vision transformer (ViT) paper: https://arxiv.org/abs/2010.11929
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

torch.set_default_dtype(torch.bfloat16)

# image patch embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super(PatchEmbedding, self).__init__()
        assert image_size % patch_size == 0, "PatchEmbedding() initialization error: image dimensions must be divisible by patch size"
        self.image_size = image_size
        self.patch_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        return x

# multihead attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "MultiHeadAttention() initialization error: latent (embedding) dimension must be divisble by number of attention heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        return x

# transformer encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super(TransformerEncoder, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

# complete vision transformer module
class ViT(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, num_classes: int, embed_dim: int, num_heads: int, num_layers: int, mlp_dim: int) -> None:
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, dtype=torch.bfloat16))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim, dtype=torch.bfloat16))
        self.layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # positional embedding
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # forward pass through encoder blocks
        for layer in self.layers:
            x = layer(x)

        # forward pass through final mlp head
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x

# prepare mnist dataset
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x.to(torch.bfloat16)
])

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT(
    image_size=32,
    patch_size=4,
    in_channels=1,
    num_classes=10,
    embed_dim=128,
    num_heads=8,
    num_layers=6,
    mlp_dim=256
).to(device)

# initialize loss function and optimizer
criterion  = nn.CrossEntropyLoss()
# optimizer = optim.adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs,targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'Loss': f"{train_loss/total:.4f}",
            'Acc': f"{100.*correct/total:.2f}%"
        })

    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%")

# model evaluation
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Evaluating")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update the progress bar with current loss and accuracy
        pbar.set_postfix({
            'Loss': f"{test_loss/total:.4f}",
            'Acc': f"{100.*correct/total:.2f}%"
        })

print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {100.*correct/total:.2f}%")


