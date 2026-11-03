import torch

from data_loader import get_dataloaders


def main():

    train_loader, val_loader, test_loader = get_dataloaders()

    print("Training batches:", len(train_loader))

    print("Validation batches:", len(val_loader))

    print("Testing batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("\nImage Shape:", images.shape)

    print("Label Shape:", labels.shape)

    print("Image Device:", images.device)

    print("\nSample Labels:", labels[:10])

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    print("\nSample Classes:")

    for label in labels[:10]:

        print(classes[label.item()])


if __name__ == "__main__":
    main()