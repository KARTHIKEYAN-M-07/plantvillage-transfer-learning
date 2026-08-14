"""
P1.4 - Grad-CAM Test
====================

Tests Grad-CAM using:

1. Apple Scab PlantVillage image
2. Real leaf image

Requirements:

- Existing model must load successfully.
- Grad-CAM must produce a heatmap.
- Prediction must be valid.
- Output image must exist.
"""

from pathlib import Path
import sys

import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT
# ============================================================

from gradcam import (
    save_gradcam,
    MODEL_FILE,
)


# ============================================================
# TEST IMAGES
# ============================================================

APPLE_SCAB_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "PlantVillage"
    / "Apple___Apple_scab"
    / "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
)

REAL_LEAF_IMAGE = (
    PROJECT_ROOT
    / "test_images"
    / "leaf.jpg"
)


# ============================================================
# REPORT DIRECTORY
# ============================================================

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_gradcam_image(
    image_path: Path,
    output_name: str,
):

    print()

    print(
        "-" * 60
    )

    print(
        f"IMAGE: {image_path.name}"
    )

    print(
        "-" * 60
    )

    # --------------------------------------------------------
    # Image existence
    # --------------------------------------------------------

    assert image_path.exists(), (
        f"Image does not exist:\n"
        f"{image_path}"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_path = (
        REPORTS_DIR
        / output_name
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = save_gradcam(
        image_path,
        output_path,
    )

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    assert isinstance(
        result,
        dict,
    )

    assert (
        result.get("success")
        is True
    )

    assert (
        "explanation"
        in result
    )

    explanation = (
        result["explanation"]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

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
        f"Heatmap saved: "
        f"{explanation['heatmap_path']}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert (
        explanation["available"]
        is True
    )

    assert (
        explanation["method"]
        == "Grad-CAM"
    )

    assert (
        0
        <= explanation["target_class"]
        < 38
    )

    assert (
        0.0
        <= explanation["confidence"]
        <= 100.0
    )

    assert (
        explanation["heatmap_saved"]
        is True
    )

    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    saved_path = Path(
        explanation[
            "heatmap_path"
        ]
    )

    assert saved_path.exists(), (
        f"Grad-CAM output was not created:\n"
        f"{saved_path}"
    )

    # --------------------------------------------------------
    # Verify image can be read
    # --------------------------------------------------------

    output_image = cv2.imread(
        str(saved_path)
    )

    assert output_image is not None, (
        "OpenCV could not read "
        "the generated Grad-CAM image."
    )

    assert (
        output_image.size > 0
    )

    print(
        "Output image verified: True"
    )

    return explanation


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "P1.4 GRAD-CAM TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"Model exists: "
        f"{MODEL_FILE.exists()}"
    )

    assert MODEL_FILE.exists(), (
        f"Model checkpoint not found:\n"
        f"{MODEL_FILE}"
    )

    # ========================================================
    # APPLE SCAB
    # ========================================================

    apple_result = test_gradcam_image(
        APPLE_SCAB_IMAGE,
        "gradcam_apple_scab.jpg",
    )

    # --------------------------------------------------------
    # Important sanity check
    # --------------------------------------------------------

    print()

    print(
        "Apple Scab sanity check:"
    )

    print(
        f"Predicted class: "
        f"{apple_result['target_class_name']}"
    )

    # We expect this particular test image to predict
    # Apple___Apple_scab because your predictor already
    # successfully predicted it as Apple Scab.

    assert (
        apple_result["target_class_name"]
        == "Apple___Apple_scab"
    ), (
        "Grad-CAM prediction does not match "
        "the known predictor result."
    )

    # ========================================================
    # REAL LEAF
    # ========================================================

    leaf_result = test_gradcam_image(
        REAL_LEAF_IMAGE,
        "gradcam_real_leaf.jpg",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "P1.4 GRAD-CAM TESTS PASSED"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Generated files:"
    )

    print(
        "  reports/gradcam_apple_scab.jpg"
    )

    print(
        "  reports/gradcam_real_leaf.jpg"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()