import torch
import matplotlib.pyplot as plt
import numpy as np

from models.mlp_mixer import MLPMixer


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained MLP-Mixer
    model = MLPMixer().to(device)

    model.load_state_dict(
        torch.load(
            "best_mlp_mixer.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Extract weights from the first Mixer block
    token_weights_1 = (
        model.blocks[0].token_mixing.mlp[0].weight
        .detach().cpu().numpy()
    )

    token_weights_2 = (
        model.blocks[0].token_mixing.mlp[2].weight
        .detach().cpu().numpy()
    )

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(token_weights_1, cmap="coolwarm", aspect="auto")
    axes[0].set_title("Token Mixing: First Linear Layer")
    axes[0].set_xlabel("Input Patches (64)")
    axes[0].set_ylabel("Hidden Units (32)")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(token_weights_2, cmap="coolwarm", aspect="auto")
    axes[1].set_title("Token Mixing: Second Linear Layer")
    axes[1].set_xlabel("Hidden Units (32)")
    axes[1].set_ylabel("Output Patches (64)")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("token_mixing_weights.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()