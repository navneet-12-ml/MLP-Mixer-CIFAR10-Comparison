import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import csv
import os

from models.mlp_mixer import MLPMixer
from data_loader import get_dataloaders



def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy



def train_experiment(
    experiment_name,
    config,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=50
):

    print("\n" + "=" * 60)
    print("EXPERIMENT:", experiment_name)
    print("=" * 60)

    set_seed(42)

    # Initialize model
    model = MLPMixer(**config).to(device)

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", parameters)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_val_accuracy = 0

    best_model_state = None


    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

        train_accuracy = 100 * correct / total

        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Save best validation model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }


    model.load_state_dict(best_model_state)

    model.to(device)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # Save checkpoint
    os.makedirs("ablation_checkpoints", exist_ok=True)

    torch.save(
        best_model_state,
        f"ablation_checkpoints/{experiment_name}.pth"
    )

    print("\nRESULTS:", experiment_name)

    print("Parameters:", parameters)

    print("Best Validation Accuracy:", best_val_accuracy)

    print("Test Accuracy:", test_accuracy)

    print("Test Loss:", test_loss)

    return {
        "Experiment": experiment_name,
        "Blocks": config["num_blocks"],
        "Token Dimension": config["token_dim"],
        "Parameters": parameters,
        "Validation Accuracy": best_val_accuracy,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss
    }



def main():

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Load existing CIFAR-10 dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=128
    )


    experiments = {

        "Baseline": {
            "patch_size": 4,
            "embed_dim": 128,
            "token_dim": 32,
            "channel_dim": 256,
            "num_blocks": 4
        },

        "Shallow_Mixer": {
            "patch_size": 4,
            "embed_dim": 128,
            "token_dim": 32,
            "channel_dim": 256,
            "num_blocks": 2
        },

        "Wide_Token_MLP": {
            "patch_size": 4,
            "embed_dim": 128,
            "token_dim": 64,
            "channel_dim": 256,
            "num_blocks": 4
        },

        "Narrow_Token_MLP": {
            "patch_size": 4,
            "embed_dim": 128,
            "token_dim": 16,
            "channel_dim": 256,
            "num_blocks": 4
        }

    }

    results = []


    for name, config in experiments.items():

        result = train_experiment(

            experiment_name=name,

            config=config,

            train_loader=train_loader,

            val_loader=val_loader,

            test_loader=test_loader,

            device=device,

            epochs=50

        )

        results.append(result)



    with open(
        "ablation_results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(results)

    print("\n" + "=" * 60)

    print("ABLATION STUDY COMPLETED")

    print("=" * 60)

    for result in results:

        print(
            result["Experiment"],
            "->",
            result["Test Accuracy"],
            "%"
        )

    print("\nResults saved to ablation_results.csv")


if __name__ == "__main__":

    main()