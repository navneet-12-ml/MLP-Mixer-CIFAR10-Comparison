import torch
import torch.nn as nn

from models.cnn import CNN
from data_loader import get_dataloaders


def main():

    # 1. Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device:", device)

    # 2. Load test dataset
    _, _, test_loader = get_dataloaders(batch_size=128)

    # 3. Initialize CNN
    model = CNN().to(device)

    # 4. Load best trained weights
    model.load_state_dict(
        torch.load(
            "best_cnn.pth",
            map_location=device,
            weights_only=True
        )
    )

    print("Trained CNN loaded successfully!")

    # 5. Loss function
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    classes = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    class_correct = [0] * 10
    class_total = [0] * 10

    # 6. Evaluation
    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):

                label = labels[i].item()

                class_total[label] += 1

                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    # 7. Overall results
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)

    print("\n========== CNN TEST RESULTS ==========")

    print("Total Test Images:", total)

    print(f"Test Loss: {avg_loss:.4f}")

    print(f"Test Accuracy: {accuracy:.2f}%")

    print("\n========== CLASS-WISE ACCURACY ==========")

    for i in range(10):

        class_accuracy = (
            100 * class_correct[i] / class_total[i]
        )

        print(
            f"{classes[i]:12s}: {class_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()