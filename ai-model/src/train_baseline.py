import json

import pandas as pd
import torch

from torch.utils.data import DataLoader

from config import (
    PROJECT_ROOT,
    REPORT_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)

from plant_dataset import (
    PlantVillageDataset
)

from transforms import (
    get_train_transforms,
    get_eval_transforms
)

from baseline_model import (
    BaselineCNN
)

from trainer import (
    Trainer
)


# ============================================================
# PATHS
# ============================================================

SPLIT_FILE = (
    REPORT_DIR /
    "dataset_split.csv"
)

CLASS_NAMES_FILE = (
    PROJECT_ROOT /
    "class_names.json"
)


# ============================================================
# BASELINE SETTINGS
# ============================================================

# Only a reference model.
# Full training split is used.

EPOCHS = 2

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

PATIENCE = 1


# ============================================================
# DEVICE
# ============================================================

def get_device():

    print(
        "\n" + "=" * 70
    )

    print(
        "DEVICE CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"CUDA version: "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU count: "
        f"{torch.cuda.device_count()}"
    )

    if not torch.cuda.is_available():

        print(
            "\nERROR: CUDA is not available."
        )

        print(
            "Do not start training on CPU."
        )

        raise RuntimeError(
            "CUDA GPU is required for this training configuration."
        )

    device = torch.device(
        "cuda"
    )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    gpu_memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU: {gpu_name}"
    )

    print(
        f"GPU memory: "
        f"{gpu_memory:.2f} GB"
    )

    torch.cuda.empty_cache()

    print(
        "\n✓ CUDA device selected."
    )

    return device


# ============================================================
# LOAD CLASSES
# ============================================================

def load_classes():

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

    if len(class_names) != 38:

        raise RuntimeError(
            f"Expected 38 classes, "
            f"found {len(class_names)}."
        )

    return class_names


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split():

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

    required = {
        "path",
        "class_name",
        "split"
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PLANTVILLAGE BASELINE CNN"
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

    print(
        "\nLoading class names..."
    )

    class_names = load_classes()

    num_classes = len(
        class_names
    )

    print(
        f"Number of classes: "
        f"{num_classes}"
    )

    class_to_index = {
        name: index
        for index, name
        in enumerate(class_names)
    }

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset split..."
    )

    df = load_split()

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
        f"Total: "
        f"{len(df):,}"
    )

    print(
        f"Train: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test: "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print(
        "\nCreating DataLoaders..."
    )

    pin_memory = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        prefetch_factor=2
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        prefetch_factor=2
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Workers: "
        f"{NUM_WORKERS}"
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
    )

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    print(
        "\nTesting GPU batch..."
    )

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"CPU batch: "
        f"{images.shape}"
    )

    images = images.to(
        device,
        non_blocking=True
    )

    labels = labels.to(
        device,
        non_blocking=True
    )

    print(
        f"GPU images: "
        f"{images.device}"
    )

    print(
        f"GPU labels: "
        f"{labels.device}"
    )

    torch.cuda.synchronize()

    print(
        "✓ GPU batch test passed."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING BASELINE CNN"
    )

    print(
        "=" * 70
    )

    model = BaselineCNN(
        num_classes=num_classes
    )

    model = model.to(
        device
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        "✓ Model moved to CUDA."
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        model_name="baseline_cnn",
        output_dir=MODEL_DIR,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = trainer.fit(
        epochs=EPOCHS
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        REPORT_DIR /
        "baseline_training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Model:"
    )

    print(
        MODEL_DIR /
        "baseline_cnn_best.pth"
    )

    print(
        f"\nHistory:"
    )

    print(
        history_path
    )

    print(
        "\nNext:"
    )

    print(
        "Evaluate baseline on the untouched test set."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()