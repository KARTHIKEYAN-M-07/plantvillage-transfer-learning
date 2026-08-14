import json
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from efficientnet_model import (
    EfficientNetB0PlantDisease
)

from plant_dataset import (
    PlantVillageDataset
)

from transforms import (
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

MODEL_FILE = (
    MODEL_DIR /
    "efficientnet_b0_final.pth"
)

REPORT_FILE = (
    REPORT_DIR /
    "final_test_classification_report.txt"
)

METRICS_FILE = (
    REPORT_DIR /
    "final_test_metrics.txt"
)

CONFUSION_MATRIX_FILE = (
    REPORT_DIR /
    "final_confusion_matrix.png"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    return device


# ============================================================
# LOAD CLASSES
# ============================================================

def load_class_names():

    with open(
        CLASS_NAMES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_dataset(
    class_to_index
):

    df = pd.read_csv(
        SPLIT_FILE
    )

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"\nTest images: "
        f"{len(test_df):,}"
    )

    dataset = PlantVillageDataset(
        dataframe=test_df,
        class_to_index=class_to_index,
        transform=get_eval_transforms()
    )

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    device
):

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"""
Final model not found:

{MODEL_FILE}
"""
        )

    model = EfficientNetB0PlantDisease(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    model.eval()

    print(
        "\n✓ Final model loaded."
    )

    return model


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(
    model,
    loader,
    device
):

    all_predictions = []

    all_labels = []

    print(
        "\nRunning inference on test set..."
    )

    total_batches = len(loader)

    with torch.no_grad():

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

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    images
                )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

            if (
                batch_index % 100 == 0
                or batch_index == total_batches
            ):

                print(
                    f"\r"
                    f"Batch "
                    f"{batch_index:,}/"
                    f"{total_batches:,}",
                    end="",
                    flush=True
                )

    print()

    return (
        all_labels,
        all_predictions
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    y_true,
    y_pred,
    class_names
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall: "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{f1:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FINAL PLANTVILLAGE TEST RESULTS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Test Accuracy: "
            f"{accuracy:.6f}\n"
        )

        file.write(
            f"Weighted Precision: "
            f"{precision:.6f}\n"
        )

        file.write(
            f"Weighted Recall: "
            f"{recall:.6f}\n"
        )

        file.write(
            f"Weighted F1: "
            f"{f1:.6f}\n"
        )

        file.write(
            f"Macro Precision: "
            f"{macro_precision:.6f}\n"
        )

        file.write(
            f"Macro Recall: "
            f"{macro_recall:.6f}\n"
        )

        file.write(
            f"Macro F1: "
            f"{macro_f1:.6f}\n"
        )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    return accuracy


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    class_names
):

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    figure, axis = plt.subplots(
        figsize=(20, 20)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        ax=axis,
        xticks_rotation=90,
        colorbar=False
    )

    axis.set_title(
        "PlantVillage — EfficientNet-B0 Final Model"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        "\n✓ Confusion matrix saved:"
    )

    print(
        CONFUSION_MATRIX_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

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
    # Test dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset(
        class_to_index
    )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        prefetch_factor=2
    )

    print(
        f"Test batches: "
        f"{len(test_loader):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_true, y_pred = get_predictions(
        model,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    save_metrics(
        y_true,
        y_pred,
        class_names
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        y_true,
        y_pred,
        class_names
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved:"
    )

    print(
        METRICS_FILE
    )

    print(
        REPORT_FILE
    )

    print(
        CONFUSION_MATRIX_FILE
    )


if __name__ == "__main__":

    main()