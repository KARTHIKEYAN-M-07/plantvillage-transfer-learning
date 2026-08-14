"""
P1.3 - Leaf / Image Suitability Tests
======================================

These tests verify:

    1. Real leaf image is accepted.
    2. PlantVillage Apple Scab image is accepted.
    3. Uniform dark image is rejected.
    4. Uniform white image is rejected.
    5. Synthetic leaf-like image is accepted.
    6. Random image is NOT confidently accepted.
    7. Missing image is rejected.

IMPORTANT:
    The suitability system is heuristic.
    It is NOT a trained leaf classifier.
"""


from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


# ============================================================
# PROJECT PATH
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

from src.image_suitability import (
    validate_image_suitability
)


# ============================================================
# TEST IMAGE DIRECTORY
# ============================================================

TEST_IMAGES = (
    PROJECT_ROOT
    / "test_images"
)


# ============================================================
# PRINT RESULT
# ============================================================

def show_result(
    name: str,
    result: dict
) -> None:

    suitability = (
        result["suitability"]
    )

    print()
    print("-" * 60)
    print(name)
    print("-" * 60)

    print(
        "Status:",
        suitability["status"]
    )

    print(
        "Score:",
        suitability["score"]
    )

    print(
        "Resolution:",
        suitability["width"],
        "x",
        suitability["height"]
    )

    print(
        "Vegetation ratio:",
        suitability[
            "vegetation_ratio"
        ]
    )

    print(
        "Green ratio:",
        suitability[
            "green_ratio"
        ]
    )

    print(
        "Largest vegetation "
        "component ratio:",
        suitability[
            "largest_vegetation_component_ratio"
        ]
    )

    print(
        "Vegetation components:",
        suitability[
            "vegetation_component_count"
        ]
    )

    print(
        "Color variation:",
        suitability[
            "color_variation"
        ]
    )

    print(
        "Texture:",
        suitability[
            "texture_score"
        ]
    )

    print(
        "Issues:",
        suitability["issues"]
    )


# ============================================================
# REAL LEAF IMAGE
# ============================================================

def test_real_leaf():

    path = (
        TEST_IMAGES
        / "leaf.jpg"
    )

    if not path.exists():

        print(
            "WARNING: "
            "test_images/leaf.jpg "
            "does not exist."
        )

        return

    result = (
        validate_image_suitability(
            path
        )
    )

    show_result(
        "REAL LEAF IMAGE",
        result
    )

    status = (
        result["suitability"]["status"]
    )

    assert status in {
        "suitable",
        "review",
    }, (
        "Real leaf image should not "
        "be classified as unsuitable."
    )


# ============================================================
# REAL PLANTVILLAGE IMAGE
# ============================================================

def test_apple_scab():

    path = (
        PROJECT_ROOT
        / "data"
        / "PlantVillage"
        / "Apple___Apple_scab"
        / "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    )

    if not path.exists():

        print(
            "WARNING: "
            "Apple Scab test image "
            "does not exist."
        )

        return

    result = (
        validate_image_suitability(
            path
        )
    )

    show_result(
        "APPLE SCAB PLANTVILLAGE IMAGE",
        result
    )

    status = (
        result["suitability"]["status"]
    )

    assert status in {
        "suitable",
        "review",
    }, (
        "PlantVillage image should not "
        "be classified as unsuitable."
    )


# ============================================================
# UNIFORM DARK IMAGE
# ============================================================

def test_uniform_dark_image():

    image = Image.new(
        "RGB",
        (256, 256),
        (10, 10, 10),
    )

    result = (
        validate_image_suitability(
            image
        )
    )

    show_result(
        "UNIFORM DARK IMAGE",
        result
    )

    assert (
        result["suitability"]["status"]
        == "unsuitable"
    )


# ============================================================
# UNIFORM WHITE IMAGE
# ============================================================

def test_uniform_white_image():

    image = Image.new(
        "RGB",
        (256, 256),
        (255, 255, 255),
    )

    result = (
        validate_image_suitability(
            image
        )
    )

    show_result(
        "UNIFORM WHITE IMAGE",
        result
    )

    assert (
        result["suitability"]["status"]
        == "unsuitable"
    )


# ============================================================
# SYNTHETIC GREEN LEAF
# ============================================================

def test_green_leaf_like_image():

    # Start with a dark brown/green background.
    array = np.zeros(
        (256, 256, 3),
        dtype=np.uint8,
    )

    array[:] = (
        40,
        80,
        30,
    )

    # OpenCV uses BGR.
    #
    # Draw a green leaf-like ellipse.
    cv2.ellipse(
        array,
        (128, 128),
        (80, 110),
        0,
        0,
        360,
        (40, 170, 50),
        -1,
    )

    # Draw central leaf vein.
    cv2.line(
        array,
        (128, 30),
        (128, 225),
        (180, 230, 180),
        4,
    )

    # Add smaller veins.
    cv2.line(
        array,
        (128, 100),
        (80, 70),
        (160, 220, 160),
        2,
    )

    cv2.line(
        array,
        (128, 130),
        (75, 110),
        (160, 220, 160),
        2,
    )

    cv2.line(
        array,
        (128, 160),
        (80, 175),
        (160, 220, 160),
        2,
    )

    # Convert BGR -> RGB.
    image = Image.fromarray(
        cv2.cvtColor(
            array,
            cv2.COLOR_BGR2RGB,
        )
    )

    result = (
        validate_image_suitability(
            image
        )
    )

    show_result(
        "SYNTHETIC GREEN LEAF",
        result
    )

    assert (
        result["suitability"]["status"]
        == "suitable"
    ), (
        "Synthetic leaf-like image "
        "should be suitable."
    )


# ============================================================
# RANDOM IMAGE
# ============================================================

def test_random_image():

    rng = np.random.default_rng(
        seed=42
    )

    array = rng.integers(
        0,
        256,
        size=(
            256,
            256,
            3,
        ),
        dtype=np.uint8,
    )

    image = Image.fromarray(
        array
    )

    result = (
        validate_image_suitability(
            image
        )
    )

    show_result(
        "RANDOM IMAGE",
        result
    )

    # IMPORTANT:
    #
    # The previous implementation incorrectly returned
    # "suitable" for this image.
    #
    # We now require the heuristic to avoid confidently
    # accepting random noise.
    #
    # Acceptable:
    #     review
    #     unsuitable
    #
    # Not acceptable:
    #     suitable

    assert (
        result["suitability"]["status"]
        != "suitable"
    ), (
        "Random image must not be "
        "confidently classified as suitable."
    )


# ============================================================
# MISSING IMAGE
# ============================================================

def test_missing_image():

    path = (
        TEST_IMAGES
        / "does_not_exist.jpg"
    )

    result = (
        validate_image_suitability(
            path
        )
    )

    show_result(
        "MISSING IMAGE",
        result
    )

    assert (
        result["suitability"]["status"]
        == "unsuitable"
    )


# ============================================================
# INVALID IMAGE
# ============================================================

def test_invalid_image():

    result = (
        validate_image_suitability(
            b"this is not an image"
        )
    )

    show_result(
        "INVALID IMAGE",
        result
    )

    assert (
        result["suitability"]["status"]
        == "unsuitable"
    )


# ============================================================
# TINY IMAGE
# ============================================================

def test_tiny_image():

    image = Image.new(
        "RGB",
        (64, 64),
        (40, 150, 40),
    )

    result = (
        validate_image_suitability(
            image
        )
    )

    show_result(
        "TINY IMAGE",
        result
    )

    # Tiny images should not be confidently considered
    # suitable for the disease classifier.
    assert (
        result["suitability"]["status"]
        != "suitable"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "P1.3 LEAF / IMAGE "
        "SUITABILITY TEST"
    )

    print("=" * 60)

    # Real images.
    test_real_leaf()

    test_apple_scab()

    # Negative cases.
    test_uniform_dark_image()

    test_uniform_white_image()

    test_invalid_image()

    test_missing_image()

    # Synthetic cases.
    test_green_leaf_like_image()

    test_random_image()

    test_tiny_image()

    print()
    print("=" * 60)

    print(
        "P1.3 SUITABILITY TESTS PASSED"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()