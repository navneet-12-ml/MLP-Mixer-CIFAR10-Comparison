import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.mlp_mixer import MLPMixer
from models.cnn import CNN
from models.vit import ViT

from data_loader import get_dataloaders


def generate_confusion_matrix(model, test_loader, device, model_name):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    # Normalize each row to percentages
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized *= 100

    classes = [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck"
    ]

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=classes
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".1f",
        colorbar=True
    )

    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    filename = f"{model_name.lower()}_confusion_matrix.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    # Save raw confusion matrix too
    np.savetxt(
        f"{model_name.lower()}_confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    print(f"\n{model_name} Confusion Matrix:")
    print(cm)

    print(f"Saved: {filename}")


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    _, _, test_loader = get_dataloaders(batch_size=128)

    models = {
        "MLP_Mixer": (MLPMixer(), "best_mlp_mixer.pth"),
        "CNN": (CNN(), "best_cnn.pth"),
        "ViT": (ViT(), "best_vit.pth")
    }

    for name, (model, checkpoint) in models.items():

        print(f"\nEvaluating {name}...")

        model = model.to(device)

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
                weights_only=True
            )
        )

        generate_confusion_matrix(
            model,
            test_loader,
            device,
            name
        )


if __name__ == "__main__":
    main()