import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from efficientnet_model import (
    EfficientNetB0PlantDisease
)

from plant_dataset import (
    PlantVillageDataset
)

from transforms import (
    get_train_transforms,
    get_eval_transforms
)

from config import (
    PROJECT_ROOT,
    REPORT_DIR,
    MODEL_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 4

EPOCHS = 5

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

PATIENCE = 2

NUM_CLASSES = 38


# ============================================================
# FILES
# ============================================================

SPLIT_FILE = (
    REPORT_DIR /
    "dataset_split.csv"
)

CLASS_NAMES_FILE = (
    PROJECT_ROOT /
    "class_names.json"
)

BEST_MODEL_FILE = (
    MODEL_DIR /
    "efficientnet_b0_phase1_best.pth"
)

HISTORY_FILE = (
    REPORT_DIR /
    "efficientnet_phase1_history.csv"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if not torch.cuda.is_available():

        raise RuntimeError(
            """
CUDA is not available.

Your project is configured for GPU training.

Check:

python -c "import torch; print(torch.cuda.is_available())"
"""
        )

    device = torch.device(
        "cuda"
    )

    print(
        "=" * 70
    )

    print(
        "DEVICE"
    )

    print(
        "=" * 70
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory: {memory:.2f} GB"
    )

    print()

    return device


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    if not CLASS_NAMES_FILE.exists():

        raise FileNotFoundError(
            f"""
Missing:

{CLASS_NAMES_FILE}

Run:

python src\\prepare_dataset.py
"""
        )

    with open(
        CLASS_NAMES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(
            file
        )

    if len(class_names) != NUM_CLASSES:

        raise RuntimeError(
            f"""
Expected {NUM_CLASSES} classes.

Found:
{len(class_names)}
"""
        )

    return class_names


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"""
Missing:

{SPLIT_FILE}

Run:

python src\\split_dataset.py
"""
        )

    df = pd.read_csv(
        SPLIT_FILE
    )

    required_columns = {
        "path",
        "class_name",
        "split"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"""
Missing columns:

{missing}
"""
        )

    return df


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    train_df,
    class_names
):

    print(
        "\nCalculating class weights..."
    )

    counts = (
        train_df["class_name"]
        .value_counts()
    )

    class_counts = []

    for class_name in class_names:

        count = counts.get(
            class_name,
            0
        )

        if count == 0:

            raise RuntimeError(
                f"""
Class has no training images:

{class_name}
"""
            )

        class_counts.append(
            count
        )

    counts_tensor = torch.tensor(
        class_counts,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Inverse square-root weighting
    #
    # More stable than plain inverse frequency for
    # a highly imbalanced dataset.
    # --------------------------------------------------------

    weights = (
        1.0
        / torch.sqrt(
            counts_tensor
        )
    )

    # --------------------------------------------------------
    # Normalize mean weight to 1
    # --------------------------------------------------------

    weights = (
        weights
        / weights.mean()
    )

    print(
        "\nClass distribution:"
    )

    print(
        f"Smallest class: "
        f"{counts_tensor.min().item():.0f}"
    )

    print(
        f"Largest class: "
        f"{counts_tensor.max().item():.0f}"
    )

    print(
        f"Weight minimum: "
        f"{weights.min().item():.4f}"
    )

    print(
        f"Weight maximum: "
        f"{weights.max().item():.4f}"
    )

    return weights


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_df,
    validation_df,
    class_to_index,
    device
):

    print(
        "\nCreating datasets..."
    )

    train_dataset = PlantVillageDataset(
        dataframe=train_df,
        class_to_index=class_to_index,
        transform=get_train_transforms()
    )

    validation_dataset = PlantVillageDataset(
        dataframe=validation_df,
        class_to_index=class_to_index,
        transform=get_eval_transforms()
    )

    print(
        f"Training images: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset):,}"
    )

    pin_memory = (
        device.type == "cuda"
    )

    loader_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory
    }

    if NUM_WORKERS > 0:

        loader_args[
            "persistent_workers"
        ] = True

        loader_args[
            "prefetch_factor"
        ] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_args
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Workers: {NUM_WORKERS}"
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
    )

    return (
        train_loader,
        validation_loader
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    total_epochs
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    total_batches = len(loader)

    for batch_index, (
        images,
        labels
    ) in enumerate(
        loader,
        start=1
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda"
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += batch_size

        # ----------------------------------------------------
        # Progress every 100 batches
        # ----------------------------------------------------

        if (
            batch_index % 100 == 0
            or batch_index == total_batches
        ):

            current_loss = (
                running_loss / total
            )

            current_accuracy = (
                correct / total
            )

            print(
                f"\r"
                f"Epoch {epoch}/{total_epochs} "
                f"| Batch "
                f"{batch_index:,}/"
                f"{total_batches:,} "
                f"| Loss "
                f"{current_loss:.4f} "
                f"| Acc "
                f"{current_accuracy * 100:.2f}%",
                end="",
                flush=True
            )

    print()

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

            batch_size = (
                images.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += batch_size

    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct / total
    )

    return (
        validation_loss,
        validation_accuracy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EFFICIENTNET-B0 — PHASE 1"
    )

    print(
        "TRANSFER LEARNING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    class_names = (
        load_class_names()
    )

    class_to_index = {
        name: index
        for index, name
        in enumerate(class_names)
    }

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        "\nDataset:"
    )

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Validation: {len(validation_df):,}"
    )

    print(
        f"Test: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Dataloaders
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader
    ) = create_dataloaders(
        train_df,
        validation_df,
        class_to_index,
        device
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_df,
            class_names
        )
    )

    class_weights = (
        class_weights.to(
            device
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING EFFICIENTNET-B0"
    )

    print(
        "=" * 70
    )

    model = EfficientNetB0PlantDisease(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # FREEZE BACKBONE
    # --------------------------------------------------------

    model.freeze_backbone()

    model = model.to(
        device
    )

    print(
        f"Total parameters: "
        f"{model.total_parameter_count():,}"
    )

    print(
        f"Trainable parameters: "
        f"{model.trainable_parameter_count():,}"
    )

    print(
        "\n✓ Backbone frozen."
    )

    print(
        "✓ Classifier is trainable."
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # Only trainable parameters are passed.
    # --------------------------------------------------------

    trainable_parameters = filter(
        lambda parameter:
            parameter.requires_grad,
        model.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1
        )
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": []
    }

    # --------------------------------------------------------
    # Best values
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_validation_accuracy = 0.0

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"PHASE 1 — EPOCH "
            f"{epoch}/{EPOCHS}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                EPOCHS
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print(
            "Validation..."
        )

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion,
                device
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "validation_loss"
        ].append(
            validation_loss
        )

        history[
            "validation_accuracy"
        ].append(
            validation_accuracy
        )

        history[
            "learning_rate"
        ].append(
            current_lr
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"\nTrain Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_validation_accuracy = (
                validation_accuracy
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        validation_loss,

                    "validation_accuracy":
                        validation_accuracy,

                    "class_names":
                        class_names,

                    "phase":
                        "phase1_transfer_learning"
                },
                BEST_MODEL_FILE
            )

            print(
                "\n✓ BEST PHASE-1 MODEL SAVED"
            )

            print(
                BEST_MODEL_FILE
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_FILE,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 1 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_FILE
    )

    print(
        "\nHistory:"
    )

    print(
        HISTORY_FILE
    )

    print(
        "\nNEXT:"
    )

    print(
        "Load this model and begin Phase 2 fine-tuning."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()