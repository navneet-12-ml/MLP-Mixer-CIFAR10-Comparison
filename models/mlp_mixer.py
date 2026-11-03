
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, img_size=32, patch_size=4,
                 in_channels=3, embed_dim=128):

        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_size = patch_size

        self.projection = nn.Linear(
            in_channels * patch_size * patch_size,
            embed_dim
        )

    def forward(self, x):

        B, C, H, W = x.shape

        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)

        x = x.contiguous().view(B, C, -1,
                                self.patch_size * self.patch_size)

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B, self.num_patches, -1)

        x = self.projection(x)

        return x




class TokenMixing(nn.Module):

    def __init__(self, num_patches=64, token_dim=32):

        super().__init__()

        self.norm = nn.LayerNorm(128)

        self.mlp = nn.Sequential(

            nn.Linear(num_patches, token_dim),

            nn.GELU(),

            nn.Linear(token_dim, num_patches)

        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = x.transpose(1, 2)

        x = self.mlp(x)

        x = x.transpose(1, 2)

        return x + residual




class ChannelMixing(nn.Module):

    def __init__(self, embed_dim=128, channel_dim=256):

        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(

            nn.Linear(embed_dim, channel_dim),

            nn.GELU(),

            nn.Linear(channel_dim, embed_dim)

        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.mlp(x)

        return x + residual
    
    
    
    
class MixerBlock(nn.Module):

    def __init__(self, num_patches=64,
                 embed_dim=128,
                 token_dim=32,
                 channel_dim=256):

        super().__init__()

        self.token_mixing = TokenMixing(
            num_patches, token_dim
        )

        self.channel_mixing = ChannelMixing(
            embed_dim, channel_dim
        )

    def forward(self, x):

        x = self.token_mixing(x)

        x = self.channel_mixing(x)

        return x





class MLPMixer(nn.Module):

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        token_dim=32,
        channel_dim=256,
        num_blocks=4,
        num_classes=10
    ):

        super().__init__()

        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            embed_dim
        )

        num_patches = (img_size // patch_size) ** 2

        # 2. Stack multiple Mixer Blocks
        self.blocks = nn.Sequential(

            *[
                MixerBlock(
                    num_patches,
                    embed_dim,
                    token_dim,
                    channel_dim
                )
                for _ in range(num_blocks)
            ]

        )

        # 3. Final Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Classification Head
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):

        # Patch Embedding
        x = self.patch_embedding(x)

        # Mixer Blocks
        x = self.blocks(x)

        # Final Normalization
        x = self.norm(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        # Classification
        x = self.head(x)

        return x