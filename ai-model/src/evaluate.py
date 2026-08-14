import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from config import (
    DEVICE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    SPLIT_PATH,
    REPORT_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)

from dataset import (
    PlantVillageDataset
)

from preprocess import (
    get_eval_transform
)

from model import (
    create_model
)


def main():

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

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
    # Load test set
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        SPLIT_PATH
    )

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    dataset = (
        PlantVillageDataset(
            test_df,
            class_to_index,
            get_eval_transform()
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        num_classes=len(class_names),
        pretrained=False
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions
            )

            all_labels.extend(
                labels.numpy()
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1-score: {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(class_names)
            )
        ),
        target_names=class_names,
        zero_division=0
    )

    print(
        "\nClassification Report:\n"
    )

    print(report)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        REPORT_DIR /
        "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PLANT DISEASE MODEL EVALUATION\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"F1-score: {f1:.6f}\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n\n"
        )

        file.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(class_names)
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(18, 18)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d"
    )

    plt.tight_layout()

    confusion_path = (
        REPORT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200
    )

    plt.close()

    print(
        f"\nMetrics saved: {metrics_path}"
    )

    print(
        f"Confusion matrix saved: "
        f"{confusion_path}"
    )


if __name__ == "__main__":

    main()