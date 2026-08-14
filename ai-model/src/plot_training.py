import json

import matplotlib.pyplot as plt

from config import (
    REPORT_DIR
)


def main():

    history_path = (
        REPORT_DIR /
        "training_history.json"
    )

    with open(
        history_path,
        "r",
        encoding="utf-8"
    ) as file:

        history = json.load(
            file
        )

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_path = (
        REPORT_DIR /
        "training_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    loss_path = (
        REPORT_DIR /
        "training_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {accuracy_path}"
    )

    print(
        f"Saved: {loss_path}"
    )


if __name__ == "__main__":

    main()