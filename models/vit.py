import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128
    ):

        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        # (B, 3, 32, 32) -> (B, 128, 8, 8)
        x = self.projection(x)

        # (B, 128, 8, 8) -> (B, 128, 64)
        x = x.flatten(2)

        # (B, 128, 64) -> (B, 64, 128)
        x = x.transpose(1, 2)

        return x
    
    
    
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens=65, embed_dim=128):
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )

    def forward(self, x):
        return x + self.pos_embedding
    
    
    
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_dim=256):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):

        # Self-attention
        z = self.norm1(x)
        attention_output, _ = self.attention(z, z, z)
        x = x + attention_output

        # Feed-forward network
        z = self.norm2(x)
        x = x + self.mlp(z)

        return x
    
    
    
    

class CLSToken(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        cls = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls, x), dim=1)

        return x
    
    
    
    
    
    
class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        num_layers=2,
        num_classes=10
    ):
        super().__init__()

        # Convert image into patch embeddings
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # Add CLS token
        self.cls_token = CLSToken(embed_dim)

        # Add positional embeddings
        num_tokens = self.patch_embedding.num_patches + 1

        self.pos_embedding = PositionalEmbedding(
            num_tokens, embed_dim
        )

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_dim)
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # Step 1: Patch embedding
        x = self.patch_embedding(x)

        # Step 2: Add CLS token
        x = self.cls_token(x)

        # Step 3: Add positional embedding
        x = self.pos_embedding(x)

        # Step 4: Pass through Transformer blocks
        x = self.transformer(x)

        # Step 5: Normalize
        x = self.norm(x)

        # Step 6: Extract CLS token
        x = x[:, 0]

        # Step 7: Classification
        x = self.head(x)

        return x