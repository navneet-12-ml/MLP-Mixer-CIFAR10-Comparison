import torch
import torch.nn as nn
from models.vit import ViT
from data_loader import get_dataloaders


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=128)

    model = ViT().to(device)

    model.load_state_dict(
        torch.load("best_vit.pth", map_location=device, weights_only=True)
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

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
                prediction = predicted[i].item()

                if label == prediction:
                    class_correct[label] += 1

                class_total[label] += 1

    accuracy = 100 * correct / total
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    print("\nClass-wise Accuracy:")

    for i in range(10):

        acc = 100 * class_correct[i] / class_total[i]

        print(f"{classes[i]}: {acc:.2f}%")


if __name__ == "__main__":
    main()