import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from preprocess import (
    get_train_transform,
    get_eval_transform
)

from dataset import (
    PlantVillageDataset
)

from model import (
    create_model,
    freeze_backbone,
    unfreeze_last_blocks
)

from config import (
    REPORT_DIR,
    MODEL_DIR,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    SPLIT_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    DEVICE,
    HEAD_EPOCHS,
    FINETUNE_EPOCHS,
    HEAD_LEARNING_RATE,
    FINETUNE_LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE
)


def set_seed():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


def calculate_accuracy(
    outputs,
    labels
):

    predictions = (
        outputs.argmax(
            dim=1
        )
    )

    correct = (
        predictions == labels
    ).sum().item()

    return (
        correct /
        labels.size(0)
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    training = (
        optimizer is not None
    )

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        if training:

            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()

                optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            batch_size
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def train_phase(
    model,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs,
    history,
    phase_name
):

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0

    for epoch in range(
        1,
        epochs + 1
    ):

        start_time = time.time()

        train_loss, train_acc = (
            run_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_acc = (
            run_epoch(
                model,
                validation_loader,
                criterion
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_acc
        )

        history["val_accuracy"].append(
            val_acc
        )

        print(
            f"\n{phase_name} | "
            f"Epoch {epoch}/{epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        print(
            f"Time: {elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_val_loss":
                        best_val_loss,

                    "model_name":
                        "efficientnet_b0"
                },
                BEST_MODEL_PATH
            )

            print(
                "✓ Best checkpoint saved."
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping."
            )

            break


def main():

    set_seed()

    print("=" * 70)
    print("PLANT DISEASE MODEL TRAINING")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    if not SPLIT_PATH.exists():

        raise FileNotFoundError(
            f"""
Split file not found:

{SPLIT_PATH}

Run first:

python src\\split_dataset.py
"""
        )

    dataframe = pd.read_csv(
        SPLIT_PATH
    )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(
            file
        )

    class_to_index = {
        name: index
        for index, name
        in enumerate(class_names)
    }

    # --------------------------------------------------------
    # Dataframes
    # --------------------------------------------------------

    train_df = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_df = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        PlantVillageDataset(
            train_df,
            class_to_index,
            get_train_transform()
        )
    )

    validation_dataset = (
        PlantVillageDataset(
            validation_df,
            class_to_index,
            get_eval_transform()
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        num_classes=len(class_names),
        pretrained=True
    )

    model = model.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # ========================================================
    # PHASE 1
    # Frozen EfficientNet backbone
    # ========================================================

    print(
        "\nPHASE 1: Training classifier head"
    )

    freeze_backbone(
        model
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters()
        ),
        lr=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_phase(
        model,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        HEAD_EPOCHS,
        history,
        "HEAD"
    )

    # ========================================================
    # PHASE 2
    # Fine tuning
    # ========================================================

    print(
        "\nPHASE 2: Fine-tuning upper layers"
    )

    unfreeze_last_blocks(
        model,
        number_of_blocks=2
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters()
        ),
        lr=FINETUNE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_phase(
        model,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        FINETUNE_EPOCHS,
        history,
        "FINE-TUNE"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        REPORT_DIR /
        "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    print(
        f"\nTraining history saved: "
        f"{history_path}"
    )

    print(
        f"Best model: "
        f"{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":

    main()