import json

import pandas as pd
import torch

from torch.utils.data import DataLoader

from config import (
    PROJECT_ROOT,
    REPORT_DIR,
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
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "DATALOADER TEST"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    print(
        "\nLoading dataset split..."
    )

    df = pd.read_csv(
        SPLIT_FILE
    )

    print(
        f"Total rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    with open(
        CLASS_NAMES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(
            file
        )

    print(
        f"Number of classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(class_names)
    }

    print(
        "\nExample class mapping:"
    )

    for class_name in class_names[:5]:

        print(
            f"{class_name} "
            f"-> "
            f"{class_to_index[class_name]}"
        )

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

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
        "\nDataset split:"
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
    # Create datasets
    # --------------------------------------------------------

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

    test_dataset = PlantVillageDataset(
        dataframe=test_df,
        class_to_index=class_to_index,
        transform=get_eval_transforms()
    )

    # --------------------------------------------------------
    # Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Get ONE training batch
    # --------------------------------------------------------

    print(
        "\nLoading one training batch..."
    )

    images, labels = next(
        iter(train_loader)
    )

    print(
        "\nTraining batch:"
    )

    print(
        f"Images shape: "
        f"{images.shape}"
    )

    print(
        f"Labels shape: "
        f"{labels.shape}"
    )

    print(
        f"Image dtype: "
        f"{images.dtype}"
    )

    print(
        f"Label dtype: "
        f"{labels.dtype}"
    )

    print(
        f"Image minimum: "
        f"{images.min().item():.4f}"
    )

    print(
        f"Image maximum: "
        f"{images.max().item():.4f}"
    )

    # --------------------------------------------------------
    # Verify batch
    # --------------------------------------------------------

    expected_channels = 3

    expected_height = 224

    expected_width = 224

    if images.ndim != 4:

        raise RuntimeError(
            f"Expected 4D tensor, "
            f"got {images.ndim}D"
        )

    if images.shape[1] != expected_channels:

        raise RuntimeError(
            "Incorrect number of channels."
        )

    if images.shape[2] != expected_height:

        raise RuntimeError(
            "Incorrect image height."
        )

    if images.shape[3] != expected_width:

        raise RuntimeError(
            "Incorrect image width."
        )

    if labels.ndim != 1:

        raise RuntimeError(
            "Labels should be 1-dimensional."
        )

    # --------------------------------------------------------
    # Check label range
    # --------------------------------------------------------

    if labels.min() < 0:

        raise RuntimeError(
            "Negative label detected."
        )

    if labels.max() >= len(
        class_names
    ):

        raise RuntimeError(
            "Label exceeds class count."
        )

    print(
        "\n✓ Image shape is correct."
    )

    print(
        "✓ Labels are valid."
    )

    print(
        "✓ Training augmentation works."
    )

    # --------------------------------------------------------
    # Test validation batch
    # --------------------------------------------------------

    print(
        "\nTesting validation DataLoader..."
    )

    validation_images, validation_labels = next(
        iter(validation_loader)
    )

    print(
        f"Validation images: "
        f"{validation_images.shape}"
    )

    print(
        f"Validation labels: "
        f"{validation_labels.shape}"
    )

    # --------------------------------------------------------
    # Test test-loader
    # --------------------------------------------------------

    print(
        "\nTesting test DataLoader..."
    )

    test_images, test_labels = next(
        iter(test_loader)
    )

    print(
        f"Test images: "
        f"{test_images.shape}"
    )

    print(
        f"Test labels: "
        f"{test_labels.shape}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "DATALOADER TEST PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()