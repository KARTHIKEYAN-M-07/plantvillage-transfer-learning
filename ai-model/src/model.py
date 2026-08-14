import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_model(
    num_classes,
    pretrained=True
):

    if pretrained:

        weights = (
            EfficientNet_B0_Weights.DEFAULT
        )

        model = efficientnet_b0(
            weights=weights
        )

    else:

        model = efficientnet_b0(
            weights=None
        )

    # --------------------------------------------------------
    # Replace classifier
    # --------------------------------------------------------

    input_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes
    )

    return model


def freeze_backbone(model):

    for parameter in model.features.parameters():

        parameter.requires_grad = False

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True


def unfreeze_last_blocks(
    model,
    number_of_blocks=2
):

    for parameter in model.features.parameters():

        parameter.requires_grad = False

    total_blocks = len(
        model.features
    )

    start_index = max(
        0,
        total_blocks - number_of_blocks
    )

    for index in range(
        start_index,
        total_blocks
    ):

        for parameter in (
            model.features[index].parameters()
        ):

            parameter.requires_grad = True

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True