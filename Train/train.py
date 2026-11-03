import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from models.mlp_mixer import MLPMixer
from data_loader import get_dataloaders


def main():

    # 1. Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


    # 2. Load dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=128
    )


    # 3. Initialize model
    model = MLPMixer().to(device)

    print("\nTotal Parameters:",
          sum(p.numel() for p in model.parameters()))


    # 4. Loss function
    criterion = nn.CrossEntropyLoss()


    # 5. Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )


    # 6. Learning rate scheduler
    epochs = 50
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs if False else 50
    )


    # 7. Training configuration

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_accuracy = 0.0


    # 8. Training loop
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)

        epoch_accuracy = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)


        # 9. Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)

        val_epoch_accuracy = 100 * val_correct / val_total

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)


        # 10. Update scheduler
        scheduler.step()


        # 11. Save best model
        if val_epoch_accuracy > best_accuracy:

            best_accuracy = val_epoch_accuracy

            torch.save(
                model.state_dict(),
                "best_mlp_mixer.pth"
            )


        # 12. Print results
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"Train Acc: {epoch_accuracy:.2f}% "
            f"Val Loss: {val_epoch_loss:.4f} "
            f"Val Acc: {val_epoch_accuracy:.2f}%"
        )


    # 13. Save training history
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        },
        "training_history.pth"
    )


    # 14. Plot loss
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("MLP-Mixer Loss")

    plt.legend()
    plt.grid()

    plt.savefig("loss_curve.png")

    plt.show()


    # 15. Plot accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title("MLP-Mixer Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig("accuracy_curve.png")

    plt.show()


    print("\nTraining Completed!")

    print("Best Validation Accuracy:", best_accuracy)


if __name__ == "__main__":
    main()