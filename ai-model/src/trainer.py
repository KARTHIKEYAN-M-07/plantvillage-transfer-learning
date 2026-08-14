from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        device,
        model_name,
        output_dir,
        learning_rate=0.001,
        weight_decay=0.0001,
        patience=1
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = (
            validation_loader
        )

        self.device = device

        self.model_name = model_name

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.learning_rate = (
            learning_rate
        )

        self.weight_decay = (
            weight_decay
        )

        self.patience = patience

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        self.scheduler = (
            ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=1
            )
        )

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        self.use_amp = (
            self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        self.history = {

            "train_loss": [],

            "train_accuracy": [],

            "validation_loss": [],

            "validation_accuracy": [],

            "learning_rate": []
        }


    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_one_epoch(
        self,
        epoch,
        total_epochs
    ):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        total_batches = len(
            self.train_loader
        )

        for batch_index, (
            images,
            labels
        ) in enumerate(
            self.train_loader,
            start=1
        ):

            # ------------------------------------------------
            # Move data to GPU
            # ------------------------------------------------

            images = images.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):

                outputs = self.model(
                    images
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            self.scaler.scale(
                loss
            ).backward()

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_index == 1
                or batch_index % 100 == 0
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


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in (
                self.validation_loader
            ):

                images = images.to(
                    self.device,
                    non_blocking=True
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp
                ):

                    outputs = self.model(
                        images
                    )

                    loss = self.criterion(
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


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        epoch,
        validation_loss,
        validation_accuracy
    ):

        checkpoint_path = (
            self.output_dir
            / f"{self.model_name}_best.pth"
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "validation_loss":
                    validation_loss,

                "validation_accuracy":
                    validation_accuracy,

                "history":
                    self.history
            },
            checkpoint_path
        )

        return checkpoint_path


    # ========================================================
    # TRAIN
    # ========================================================

    def fit(
        self,
        epochs
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"TRAINING: {self.model_name}"
        )

        print(
            "=" * 70
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Epochs: {epochs}"
        )

        print(
            f"Learning rate: "
            f"{self.learning_rate}"
        )

        print(
            f"Training batches: "
            f"{len(self.train_loader):,}"
        )

        print(
            f"Validation batches: "
            f"{len(self.validation_loader):,}"
        )

        print(
            f"Mixed precision: "
            f"{self.use_amp}"
        )

        best_validation_loss = float(
            "inf"
        )

        best_validation_accuracy = 0.0

        epochs_without_improvement = 0

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(
            1,
            epochs + 1
        ):

            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            print(
                "-" * 70
            )

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            train_loss, train_accuracy = (
                self.train_one_epoch(
                    epoch,
                    epochs
                )
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            print(
                "Validation..."
            )

            validation_loss, validation_accuracy = (
                self.validate()
            )

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            self.scheduler.step(
                validation_loss
            )

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            self.history[
                "train_loss"
            ].append(
                train_loss
            )

            self.history[
                "train_accuracy"
            ].append(
                train_accuracy
            )

            self.history[
                "validation_loss"
            ].append(
                validation_loss
            )

            self.history[
                "validation_accuracy"
            ].append(
                validation_accuracy
            )

            self.history[
                "learning_rate"
            ].append(
                current_lr
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            print(
                f"Train Loss: "
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

            # ------------------------------------------------
            # Save best
            # ------------------------------------------------

            if validation_loss < best_validation_loss:

                best_validation_loss = (
                    validation_loss
                )

                best_validation_accuracy = (
                    validation_accuracy
                )

                epochs_without_improvement = 0

                checkpoint_path = (
                    self.save_checkpoint(
                        epoch,
                        validation_loss,
                        validation_accuracy
                    )
                )

                print(
                    "\n✓ Best model saved:"
                )

                print(
                    checkpoint_path
                )

            else:

                epochs_without_improvement += 1

                print(
                    f"\nNo improvement: "
                    f"{epochs_without_improvement}/"
                    f"{self.patience}"
                )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                epochs_without_improvement
                >= self.patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "TRAINING COMPLETE"
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

        return self.history