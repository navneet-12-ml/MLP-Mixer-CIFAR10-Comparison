import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def get_dataloaders(batch_size=128):

    # 1. Training transformations
    train_transform = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )

    ])

    # 2. Validation and testing transformations
    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )

    ])

    # 3. Load training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./dataset",
        train=True,
        download=True,
        transform=train_transform
    )

    # 4. Load another copy for validation
    val_dataset = torchvision.datasets.CIFAR10(
        root="./dataset",
        train=True,
        download=True,
        transform=test_transform
    )

    # 5. Generate indices
    indices = list(range(50000))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=5000,
        random_state=42,
        stratify=train_dataset.targets
    )

    # 6. Create subsets
    train_subset = Subset(
        train_dataset,
        train_indices
    )

    val_subset = Subset(
        val_dataset,
        val_indices
    )

    # 7. Load testing dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="./dataset",
        train=False,
        download=True,
        transform=test_transform
    )

    # 8. Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader