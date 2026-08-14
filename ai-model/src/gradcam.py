"""
P1.4 - Grad-CAM Explainability
==============================

Grad-CAM for the EXISTING Plant Disease EfficientNet-B0 model.

IMPORTANT:
    - No retraining.
    - No modification of model weights.
    - Uses the exact same model architecture as predictor.py.
    - Uses the exact same preprocessing as predictor.py.
    - Uses class_names stored inside the checkpoint.

Grad-CAM shows regions that contributed to a prediction.
It is an explanation mechanism, not proof of disease location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from efficientnet_model import EfficientNetB0PlantDisease


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 38

IMAGE_SIZE = 224

TOP_K = 5


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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# PREPROCESSING
# ============================================================
#
# EXACTLY MATCHES predictor.py:
#
# Resize(256)
# CenterCrop(224)
# ToTensor()
# Normalize(ImageNet)
#
# ============================================================

from torchvision import transforms


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model() -> tuple[
    torch.nn.Module,
    list[str],
]:
    """
    Load the exact same architecture and checkpoint
    used by predictor.py.

    Returns:
        model
        class_names
    """

    print(
        "Loading Grad-CAM model..."
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"""
Final model not found:

{MODEL_FILE}

Expected:
models/efficientnet_b0_final.pth
"""
        )

    # --------------------------------------------------------
    # Create EXACT project model
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
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Model checkpoint must be a dictionary."
        )

    # --------------------------------------------------------
    # Verify state dictionary
    # --------------------------------------------------------

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    class_names = checkpoint.get(
        "class_names"
    )

    if not class_names:

        raise RuntimeError(
            "Class names are missing "
            "from the model checkpoint."
        )

    if len(class_names) != NUM_CLASSES:

        raise RuntimeError(
            f"""
Expected {NUM_CLASSES} classes.

Found:
{len(class_names)}
"""
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        f"Classes: {len(class_names)}"
    )

    print(
        "✓ Grad-CAM model ready."
    )

    return (
        model,
        class_names,
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """
    Grad-CAM implementation.

    IMPORTANT:
        The actual EfficientNet is wrapped inside:

            EfficientNetB0PlantDisease
                └── model

        Therefore the target layer is:

            model.model.features[-1]

        NOT:

            model.features[-1]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ):

        self.model = model

        self.target_layer = (
            target_layer
        )

        self.activations = None

        self.gradients = None

        # ----------------------------------------------------
        # Register hooks
        # ----------------------------------------------------

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = (
            output.detach()
        )

    # ========================================================
    # BACKWARD HOOK
    # ========================================================

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        if grad_output is None:

            return

        if len(grad_output) == 0:

            return

        self.gradients = (
            grad_output[0].detach()
        )

    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    def remove_hooks(self):

        if self.forward_handle:

            self.forward_handle.remove()

        if self.backward_handle:

            self.backward_handle.remove()

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[
        np.ndarray,
        int,
        float,
    ]:
        """
        Generate Grad-CAM heatmap.

        Returns:
            heatmap
            predicted class index
            confidence
        """

        # ----------------------------------------------------
        # Clear previous values
        # ----------------------------------------------------

        self.activations = None

        self.gradients = None

        # ----------------------------------------------------
        # Move tensor
        # ----------------------------------------------------

        input_tensor = (
            input_tensor.to(
                DEVICE
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT use torch.no_grad().
        #
        # Grad-CAM requires gradients.
        # ----------------------------------------------------

        self.model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = self.model(
            input_tensor
        )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

        predicted_class = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_class,
            ].item()
        )

        # ----------------------------------------------------
        # Target class
        # ----------------------------------------------------

        if target_class is None:

            target_class = (
                predicted_class
            )

        if not (
            0
            <= target_class
            < NUM_CLASSES
        ):

            raise ValueError(
                f"Invalid target class: "
                f"{target_class}"
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        target_score = outputs[
            0,
            target_class,
        ]

        self.model.zero_grad(
            set_to_none=True
        )

        target_score.backward()

        # ----------------------------------------------------
        # Verify hooks
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = (
            self.gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        cam = cam[0]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam_min = cam.min()

        cam_max = cam.max()

        difference = (
            cam_max
            - cam_min
        )

        if difference.item() < 1e-8:

            heatmap = torch.zeros_like(
                cam
            )

        else:

            heatmap = (
                cam - cam_min
            ) / difference

        # ----------------------------------------------------
        # NumPy
        # ----------------------------------------------------

        heatmap = (
            heatmap
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        return (
            heatmap,
            predicted_class,
            confidence,
        )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image_path: str | Path,
) -> tuple[
    Image.Image,
    torch.Tensor,
]:
    """
    Load original image and create the exact tensor
    used by predictor.py.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    if not image_path.is_file():

        raise ValueError(
            f"Path is not a file:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Exact predictor preprocessing
    # --------------------------------------------------------

    tensor = IMAGE_TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    return (
        image,
        tensor,
    )


# ============================================================
# CLASS NAME PARSING
# ============================================================

def parse_class_name(
    class_name: str,
) -> tuple[str, str]:
    """
    Convert:

        Apple___Apple_scab

    into:

        Apple
        Apple scab
    """

    parts = class_name.split(
        "___"
    )

    if len(parts) == 2:

        plant = (
            parts[0]
            .replace(
                "_",
                " ",
            )
            .strip()
        )

        disease = (
            parts[1]
            .replace(
                "_",
                " ",
            )
            .strip()
        )

        return (
            plant,
            disease,
        )

    return (
        class_name.replace(
            "_",
            " ",
        ),
        "Unknown",
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create heatmap overlay.

    Returns:
        OpenCV BGR uint8 image.
    """

    original = np.asarray(
        original_image
    )

    height, width = (
        original.shape[:2]
    )

    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap_resized = cv2.resize(
        heatmap,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # Convert to 0-255
    # --------------------------------------------------------

    heatmap_uint8 = (
        heatmap_resized
        * 255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Color map
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR,
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_bgr,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    return overlay


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_gradcam(
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Generate and save Grad-CAM.

    Returns a structured result that can later be incorporated
    into the AI -> backend contract.
    """

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if output_path is None:

        output_path = (
            REPORTS_DIR
            / f"gradcam_{image_path.stem}.jpg"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load exact project model
    # --------------------------------------------------------

    model, class_names = (
        load_model()
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image, input_tensor = (
        load_image(
            image_path
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # EfficientNetB0PlantDisease
    #       └── self.model
    #             └── features
    #
    # Therefore:
    #
    # model.model.features[-1]
    #
    # --------------------------------------------------------

    target_layer = (
        model.model.features[-1]
    )

    # --------------------------------------------------------
    # Create Grad-CAM
    # --------------------------------------------------------

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:

        (
            heatmap,
            predicted_class,
            confidence,
        ) = gradcam.generate(
            input_tensor
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    class_name = class_names[
        predicted_class
    ]

    plant, disease = (
        parse_class_name(
            class_name
        )
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_gradcam_overlay(
        original_image,
        heatmap,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_path),
        overlay,
    )

    if not success:

        raise RuntimeError(
            f"Failed to save Grad-CAM image:\n"
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Heatmap statistics
    # --------------------------------------------------------

    heatmap_min = float(
        heatmap.min()
    )

    heatmap_max = float(
        heatmap.max()
    )

    heatmap_mean = float(
        heatmap.mean()
    )

    # --------------------------------------------------------
    # Structured result
    # --------------------------------------------------------

    return {
        "success": True,

        "explanation": {

            "available": True,

            "method": "Grad-CAM",

            "target_layer": (
                "model.model.features[-1]"
            ),

            "target_class": (
                predicted_class
            ),

            "target_class_name": (
                class_name
            ),

            "plant": plant,

            "disease": disease,

            "confidence": round(
                confidence * 100.0,
                2,
            ),

            "heatmap": {

                "min": round(
                    heatmap_min,
                    6,
                ),

                "max": round(
                    heatmap_max,
                    6,
                ),

                "mean": round(
                    heatmap_mean,
                    6,
                ),
            },

            "heatmap_saved": True,

            "heatmap_path": str(
                output_path
            ),
        },
    }


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src\\gradcam.py "
            "<image_path>"
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    print(
        "=" * 60
    )

    print(
        "P1.4 GRAD-CAM"
    )

    print(
        "=" * 60
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Image: {image_path}"
    )

    print()

    result = save_gradcam(
        image_path
    )

    explanation = (
        result["explanation"]
    )

    print(
        "Prediction:"
    )

    print(
        f"Plant: "
        f"{explanation['plant']}"
    )

    print(
        f"Disease: "
        f"{explanation['disease']}"
    )

    print(
        f"Confidence: "
        f"{explanation['confidence']}%"
    )

    print()

    print(
        "Grad-CAM:"
    )

    print(
        f"Available: "
        f"{explanation['available']}"
    )

    print(
        f"Target class: "
        f"{explanation['target_class_name']}"
    )

    print(
        f"Target layer: "
        f"{explanation['target_layer']}"
    )

    print(
        f"Heatmap range: "
        f"{explanation['heatmap']['min']} "
        f"-> "
        f"{explanation['heatmap']['max']}"
    )

    print(
        f"Heatmap saved: "
        f"{explanation['heatmap_path']}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()