import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class EfficientNetB0PlantDisease(nn.Module):

    def __init__(self, num_classes=38):

        super().__init__()

        # ====================================================
        # LOAD IMAGENET PRETRAINED EFFICIENTNET-B0
        # ====================================================

        weights = EfficientNet_B0_Weights.DEFAULT

        self.model = efficientnet_b0(
            weights=weights
        )

        # ====================================================
        # REPLACE ORIGINAL CLASSIFIER
        # ====================================================

        input_features = (
            self.model.classifier[1].in_features
        )

        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)


    # ========================================================
    # FREEZE BACKBONE
    # ========================================================

    def freeze_backbone(self):

        for parameter in self.model.features.parameters():

            parameter.requires_grad = False

        # Classifier must remain trainable

        for parameter in self.model.classifier.parameters():

            parameter.requires_grad = True


    # ========================================================
    # UNFREEZE FINAL BLOCKS
    # ========================================================

    def unfreeze_last_blocks(
        self,
        number_of_blocks=2
    ):

        # First freeze everything

        for parameter in self.model.features.parameters():

            parameter.requires_grad = False

        # ----------------------------------------------------
        # EfficientNet-B0 features are sequential blocks
        # ----------------------------------------------------

        total_blocks = len(
            self.model.features
        )

        start_index = max(
            0,
            total_blocks - number_of_blocks
        )

        for block in self.model.features[
            start_index:
        ].parameters():

            block.requires_grad = True

        # ----------------------------------------------------
        # Classifier always trainable
        # ----------------------------------------------------

        for parameter in self.model.classifier.parameters():

            parameter.requires_grad = True


    # ========================================================
    # UNFREEZE EVERYTHING
    # ========================================================

    def unfreeze_all(self):

        for parameter in self.model.parameters():

            parameter.requires_grad = True


    # ========================================================
    # TRAINABLE PARAMETERS
    # ========================================================

    def trainable_parameter_count(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


    # ========================================================
    # TOTAL PARAMETERS
    # ========================================================

    def total_parameter_count(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )