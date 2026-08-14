import json

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

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 2

NUM_CLASSES = 38

UNFREEZE_BLOCKS = 2


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

PHASE1_MODEL = (
    MODEL_DIR /
    "efficientnet_b0_phase1_best.pth"
)

FINAL_MODEL = (
    MODEL_DIR /
    "efficientnet_b0_final.pth"
)

HISTORY_FILE = (
    REPORT_DIR /
    "efficientnet_finetune_history.csv"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 70)
    print("DEVICE")
    print("=" * 70)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print()

    return device


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

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
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(class_names)}"
        )

    return class_names


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{SPLIT_FILE}"
        )

    df = pd.read_csv(
        SPLIT_FILE
    )

    required = {
        "path",
        "class_name",
        "split"
    }

    missing = (
        required -
        set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    train_df,
    class_names
):

    counts = (
        train_df["class_name"]
        .value_counts()
    )

    values = []

    for name in class_names:

        count = counts.get(
            name,
            0
        )

        if count == 0:

            raise RuntimeError(
                f"No training images for {name}"
            )

        values.append(
            count
        )

    counts_tensor = torch.tensor(
        values,
        dtype=torch.float32
    )

    weights = (
        1.0 /
        torch.sqrt(
            counts_tensor
        )
    )

    weights = (
        weights /
        weights.mean()
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

    train_dataset = (
        PlantVillageDataset(
            dataframe=train_df,
            class_to_index=class_to_index,
            transform=get_train_transforms()
        )
    )

    validation_dataset = (
        PlantVillageDataset(
            dataframe=validation_df,
            class_to_index=class_to_index,
            transform=get_eval_transforms()
        )
    )

    pin_memory = (
        device.type == "cuda"
    )

    common_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory
    }

    if NUM_WORKERS > 0:

        common_args[
            "persistent_workers"
        ] = True

        common_args[
            "prefetch_factor"
        ] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_args
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_args
    )

    print(
        f"Training images: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset):,}"
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
# LOAD PHASE 1 MODEL
# ============================================================

def load_phase1_model(
    device,
    class_names
):

    if not PHASE1_MODEL.exists():

        raise FileNotFoundError(
            f"""
Phase 1 checkpoint not found:

{PHASE1_MODEL}

Run Phase 1 first.
"""
        )

    model = (
        EfficientNetB0PlantDisease(
            num_classes=NUM_CLASSES
        )
    )

    checkpoint = torch.load(
        PHASE1_MODEL,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "\n✓ Phase 1 checkpoint loaded."
    )

    # --------------------------------------------------------
    # Unfreeze final blocks
    # --------------------------------------------------------

    model.unfreeze_last_blocks(
        number_of_blocks=UNFREEZE_BLOCKS
    )

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
        f"Unfrozen blocks: "
        f"{UNFREEZE_BLOCKS}"
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch
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
            loss.item() *
            batch_size
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

        if (
            batch_index % 100 == 0
            or batch_index == total_batches
        ):

            loss_value = (
                running_loss /
                total
            )

            accuracy = (
                correct /
                total
            )

            print(
                f"\r"
                f"Epoch {epoch} "
                f"| Batch "
                f"{batch_index:,}/"
                f"{total_batches:,} "
                f"| Loss "
                f"{loss_value:.4f} "
                f"| Acc "
                f"{accuracy * 100:.2f}%",
                end="",
                flush=True
            )

    print()

    return (
        running_loss / total,
        correct / total
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
                loss.item() *
                batch_size
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

    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EFFICIENTNET-B0 — PHASE 2"
    )

    print(
        "FINE-TUNING"
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

    # Test is intentionally NOT loaded.
    #
    # It remains untouched until final evaluation.

    print(
        "\nDataset:"
    )

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Validation: {len(validation_df):,}"
    )

    # --------------------------------------------------------
    # DataLoader
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
    # Model
    # --------------------------------------------------------

    model = load_phase1_model(
        device,
        class_names
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

    criterion = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    # --------------------------------------------------------
    # Optimizer
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
    # Mixed precision
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

    best_loss = float(
        "inf"
    )

    best_accuracy = 0.0

    no_improvement = 0

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
            f"FINE-TUNING EPOCH "
            f"{epoch}/{EPOCHS}"
        )

        print(
            "=" * 70
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch
            )
        )

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

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

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
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if validation_loss < best_loss:

            best_loss = (
                validation_loss
            )

            best_accuracy = (
                validation_accuracy
            )

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        validation_loss,

                    "validation_accuracy":
                        validation_accuracy,

                    "class_names":
                        class_names,

                    "phase":
                        "phase2_fine_tuning"
                },
                FINAL_MODEL
            )

            print(
                "\n✓ BEST FINE-TUNED MODEL SAVED"
            )

            print(
                FINAL_MODEL
            )

        else:

            no_improvement += 1

            print(
                f"\nNo improvement: "
                f"{no_improvement}/"
                f"{PATIENCE}"
            )

        if no_improvement >= PATIENCE:

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
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 2 FINE-TUNING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best validation loss: "
        f"{best_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        "\nFinal model:"
    )

    print(
        FINAL_MODEL
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
        "Evaluate the final model on the untouched test set."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()