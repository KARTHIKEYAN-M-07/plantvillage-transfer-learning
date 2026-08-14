import torch

from baseline_model import BaselineCNN


def main():

    print("=" * 70)
    print("BASELINE CNN TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    num_classes = 38

    batch_size = 4

    image_size = 224

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BaselineCNN(
        num_classes=num_classes
    )

    print(
        "\nModel created successfully."
    )

    # --------------------------------------------------------
    # Create fake input
    # --------------------------------------------------------

    x = torch.randn(
        batch_size,
        3,
        image_size,
        image_size
    )

    print(
        f"Input shape: {x.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

    print(
        f"Output shape: {output.shape}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        num_classes
    )

    if tuple(output.shape) != expected_shape:

        raise RuntimeError(
            f"""
Incorrect output shape.

Expected:
{expected_shape}

Received:
{tuple(output.shape)}
"""
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        "\n✓ Forward pass successful."
    )

    print(
        "✓ Output has 38 class scores."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE CNN TEST PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()