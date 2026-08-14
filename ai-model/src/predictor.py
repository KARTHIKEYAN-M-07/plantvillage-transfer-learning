from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from efficientnet_model import EfficientNetB0PlantDisease


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 38
IMAGE_SIZE = 224
TOP_K = 5

# These are APPLICATION thresholds, not model accuracy.
# They should be treated as heuristics.
HIGH_CONFIDENCE = 80.0
MODERATE_CONFIDENCE = 50.0

MIN_IMAGE_WIDTH = 128
MIN_IMAGE_HEIGHT = 128

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SRC_DIR.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_final.pth"
)


# ============================================================
# TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

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
# PREDICTOR
# ============================================================

class PlantDiseasePredictor:

    def __init__(self):

        print("=" * 70)
        print("LOADING PLANT DISEASE PREDICTOR")
        print("=" * 70)

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if torch.cuda.is_available():

            self.device = torch.device(
                "cuda"
            )

        else:

            self.device = torch.device(
                "cpu"
            )

        print(
            f"Device: {self.device}"
        )

        if self.device.type == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                f"""
Final model not found:

{MODEL_FILE}

Make sure:
models/efficientnet_b0_final.pth

exists.
"""
            )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = (
            EfficientNetB0PlantDisease(
                num_classes=NUM_CLASSES
            )
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            MODEL_FILE,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------

        self.class_names = checkpoint.get(
            "class_names"
        )

        if not self.class_names:

            raise RuntimeError(
                "Class names are missing "
                "from the model checkpoint."
            )

        if len(self.class_names) != NUM_CLASSES:

            raise RuntimeError(
                f"""
Expected {NUM_CLASSES} classes.

Found:
{len(self.class_names)}
"""
            )

        print(
            f"Classes: {len(self.class_names)}"
        )

        print(
            "✓ Predictor ready."
        )

        print(
            "=" * 70
        )


    # ========================================================
    # CLASS NAME
    # ========================================================

    @staticmethod
    def parse_class_name(
        class_name
    ):

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

            return plant, disease

        return (
            class_name.replace(
                "_",
                " "
            ),
            "Unknown"
        )


    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    @staticmethod
    def validate_image(
        image_path
    ):

        path = Path(
            image_path
        )

        # ----------------------------------------------------
        # Exists
        # ----------------------------------------------------

        if not path.exists():

            raise FileNotFoundError(
                "Image file does not exist."
            )

        if not path.is_file():

            raise ValueError(
                "Provided path is not a file."
            )

        # ----------------------------------------------------
        # Extension
        # ----------------------------------------------------

        if (
            path.suffix.lower()
            not in ALLOWED_EXTENSIONS
        ):

            raise ValueError(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, WEBP or BMP."
            )

        # ----------------------------------------------------
        # Open
        # ----------------------------------------------------

        try:

            image = Image.open(
                path
            )

            image.verify()

        except (
            UnidentifiedImageError,
            OSError
        ):

            raise ValueError(
                "The uploaded file is not "
                "a valid image."
            )

        # Re-open after verify()
        image = Image.open(
            path
        )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        width, height = image.size

        if (
            width < MIN_IMAGE_WIDTH
            or height < MIN_IMAGE_HEIGHT
        ):

            raise ValueError(
                f"""
Image is too small.

Minimum:
{MIN_IMAGE_WIDTH} x {MIN_IMAGE_HEIGHT}

Received:
{width} x {height}
"""
            )

        return image


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    @staticmethod
    def confidence_level(
        confidence
    ):

        if confidence >= HIGH_CONFIDENCE:

            return "high"

        if confidence >= MODERATE_CONFIDENCE:

            return "moderate"

        return "low"


    # ========================================================
    # PREDICT IMAGE
    # ========================================================

    def predict(
        self,
        image_path
    ):

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        image = self.validate_image(
            image_path
        )

        original_width, original_height = (
            image.size
        )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        tensor = IMAGE_TRANSFORM(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            if self.device.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = self.model(
                        tensor
                    )

            else:

                outputs = self.model(
                    tensor
                )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        k = min(
            TOP_K,
            len(self.class_names)
        )

        values, indices = torch.topk(
            probabilities,
            k=k
        )

        top_predictions = []

        for value, index in zip(
            values,
            indices
        ):

            class_name = (
                self.class_names[
                    index.item()
                ]
            )

            plant, disease = (
                self.parse_class_name(
                    class_name
                )
            )

            confidence = (
                value.item() * 100
            )

            top_predictions.append({
                "class_name":
                    class_name,

                "plant":
                    plant,

                "disease":
                    disease,

                "confidence":
                    round(
                        confidence,
                        2
                    )
            })

        # ----------------------------------------------------
        # Best prediction
        # ----------------------------------------------------

        best = top_predictions[0]

        confidence = (
            best["confidence"]
        )

        level = (
            self.confidence_level(
                confidence
            )
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = {

            "success": True,

            "model": {
                "name":
                    "EfficientNet-B0",

                "classes":
                    NUM_CLASSES
            },

            "image": {
                "width":
                    original_width,

                "height":
                    original_height
            },

            "prediction": {

                "class_name":
                    best["class_name"],

                "plant":
                    best["plant"],

                "disease":
                    best["disease"],

                "confidence":
                    best["confidence"],

                "confidence_level":
                    level,

                "needs_review":
                    confidence < HIGH_CONFIDENCE
            },

            "top_5":
                top_predictions
        }

        return result