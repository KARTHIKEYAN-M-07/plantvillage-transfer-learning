import torch

from efficientnet_model import (
    EfficientNetB0PlantDisease
)


def main():

    print("=" * 70)
    print("EFFICIENTNET-B0 TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nCreating EfficientNet-B0..."
    )

    model = EfficientNetB0PlantDisease(
        num_classes=38
    )

    model = model.to(
        device
    )

    print(
        "✓ Model created."
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    print(
        f"\nTotal parameters: "
        f"{model.total_parameter_count():,}"
    )

    # --------------------------------------------------------
    # Phase 1
    # --------------------------------------------------------

    model.freeze_backbone()

    print(
        f"Phase 1 trainable parameters: "
        f"{model.trainable_parameter_count():,}"
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        with torch.amp.autocast(
            device_type="cuda"
        ):

            output = model(x)

    print(
        f"\nInput shape: "
        f"{x.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if tuple(output.shape) != (2, 38):

        raise RuntimeError(
            f"""
Incorrect output shape.

Expected:
(2, 38)

Received:
{tuple(output.shape)}
"""
        )

    # --------------------------------------------------------
    # GPU synchronization
    # --------------------------------------------------------

    torch.cuda.synchronize()

    print(
        "\n✓ Forward pass successful."
    )

    print(
        "✓ Output contains 38 class scores."
    )

    # --------------------------------------------------------
    # Phase 2 test
    # --------------------------------------------------------

    model.unfreeze_last_blocks(
        number_of_blocks=2
    )

    print(
        f"\nPhase 2 trainable parameters: "
        f"{model.trainable_parameter_count():,}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EFFICIENTNET-B0 TEST PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()