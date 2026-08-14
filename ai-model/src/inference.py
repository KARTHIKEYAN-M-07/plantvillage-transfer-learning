import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from efficientnet_model import EfficientNetB0PlantDisease


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 38

IMAGE_SIZE = 224

TOP_K = 5


# ============================================================
# PROJECT PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SRC_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = (
    MODEL_DIR /
    "efficientnet_b0_final.pth"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# IMPORTANT:
# This preprocessing must match the preprocessing used
# during model evaluation.

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"""
FINAL MODEL NOT FOUND

Expected:

{MODEL_FILE}

Make sure Phase 2 fine-tuning was completed.
"""
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = (
        EfficientNetB0PlantDisease(
            num_classes=NUM_CLASSES
        )
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

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

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    if "class_names" in checkpoint:

        class_names = checkpoint[
            "class_names"
        ]

    else:

        raise RuntimeError(
            """
Class names were not stored in the
model checkpoint.
"""
        )

    if len(class_names) != NUM_CLASSES:

        raise RuntimeError(
            f"""
Expected {NUM_CLASSES} classes.

Found:
{len(class_names)}
"""
        )

    print(
        "✓ Model loaded."
    )

    print(
        f"Classes: {len(class_names)}"
    )

    return (
        model,
        class_names,
        device
    )


# ============================================================
# CLEAN CLASS NAME
# ============================================================

def clean_class_name(
    class_name
):

    # PlantVillage format:
    #
    # Tomato___Late_blight
    #
    # Convert to:
    #
    # Tomato - Late blight

    parts = class_name.split(
        "___"
    )

    if len(parts) == 2:

        plant = (
            parts[0]
            .replace("_", " ")
            .strip()
        )

        disease = (
            parts[1]
            .replace("_", " ")
            .strip()
        )

        return (
            plant,
            disease
        )

    return (
        class_name.replace(
            "_",
            " "
        ),
        "Unknown"
    )


# ============================================================
# PREDICT
# ============================================================

def predict(
    image_path,
    model,
    class_names,
    device
):

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"""
Image not found:

{image_path}
"""
        )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        )

    except Exception as error:

        raise RuntimeError(
            f"""
Unable to open image:

{image_path}

Error:
{error}
"""
        )

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    original_size = image.size

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = IMAGE_TRANSFORM(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        if device.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    tensor
                )

        else:

            outputs = model(
                tensor
            )

    # --------------------------------------------------------
    # Convert logits → probabilities
    # --------------------------------------------------------

    probabilities = F.softmax(
        outputs,
        dim=1
    )

    probabilities = (
        probabilities[0]
        .cpu()
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    values, indices = torch.topk(
        probabilities,
        k=min(
            TOP_K,
            len(class_names)
        )
    )

    predictions = []

    for value, index in zip(
        values,
        indices
    ):

        class_name = (
            class_names[
                index.item()
            ]
        )

        plant, disease = (
            clean_class_name(
                class_name
            )
        )

        predictions.append(
            {
                "class_name":
                    class_name,

                "plant":
                    plant,

                "disease":
                    disease,

                "confidence":
                    round(
                        value.item() * 100,
                        2
                    )
            }
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    best = predictions[0]

    result = {

        "success": True,

        "image": str(
            image_path
        ),

        "image_size": {
            "width":
                original_size[0],

            "height":
                original_size[1]
        },

        "prediction": {
            "class_name":
                best["class_name"],

            "plant":
                best["plant"],

            "disease":
                best["disease"],

            "confidence":
                best["confidence"]
        },

        "top_5":
            predictions
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print(
        "\n" + "=" * 70
    )

    print(
        "PLANT DISEASE PREDICTION"
    )

    print(
        "=" * 70
    )

    prediction = (
        result["prediction"]
    )

    print(
        f"\nPlant:"
        f" {prediction['plant']}"
    )

    print(
        f"Disease:"
        f" {prediction['disease']}"
    )

    print(
        f"Confidence:"
        f" {prediction['confidence']:.2f}%"
    )

    print(
        "\nTop predictions:"
    )

    print(
        "-" * 70
    )

    for rank, item in enumerate(
        result["top_5"],
        start=1
    ):

        print(
            f"{rank}. "
            f"{item['plant']} - "
            f"{item['disease']} "
            f"({item['confidence']:.2f}%)"
        )

    print(
        "=" * 70
    )

    print(
        "\nJSON:"
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Image argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            """
Usage:

python src\\inference.py <image_path>

Example:

python src\\inference.py data\\test_leaf.jpg
"""
        )

        sys.exit(1)

    image_path = sys.argv[1]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, class_names, device = (
        load_model()
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result = predict(
        image_path,
        model,
        class_names,
        device
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_result(
        result
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()