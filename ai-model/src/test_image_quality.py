"""
P1.2 Image Quality Tests
"""


from pathlib import Path
import sys

from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.image_quality import (
    validate_image_quality
)


TEST_IMAGES = (
    PROJECT_ROOT
    / "test_images"
)


def show(name, result):

    quality = result["image_quality"]

    print()
    print("-" * 60)
    print(name)
    print("-" * 60)

    print(
        "Valid:",
        quality["valid"]
    )

    print(
        "Quality:",
        quality["quality_level"]
    )

    print(
        "Resolution:",
        quality["width"],
        "x",
        quality["height"]
    )

    print(
        "Blur:",
        quality["blur_score"]
    )

    print(
        "Brightness:",
        quality["brightness"]
    )

    print(
        "Issues:",
        quality["issues"]
    )


def test_missing():

    result = validate_image_quality(
        TEST_IMAGES / "missing.jpg"
    )

    show(
        "MISSING IMAGE",
        result
    )

    assert (
        result["image_quality"]["valid"]
        is False
    )


def test_invalid():

    result = validate_image_quality(
        b"not an image"
    )

    show(
        "INVALID IMAGE",
        result
    )

    assert (
        result["image_quality"]["valid"]
        is False
    )


def test_tiny():

    image = Image.new(
        "RGB",
        (64, 64),
        (120, 120, 120),
    )

    result = validate_image_quality(
        image
    )

    show(
        "TINY IMAGE",
        result
    )

    assert (
        result["image_quality"]["valid"]
        is False
    )


def test_dark():

    image = Image.new(
        "RGB",
        (256, 256),
        (10, 10, 10),
    )

    result = validate_image_quality(
        image
    )

    show(
        "DARK IMAGE",
        result
    )

    assert (
        result["image_quality"]["valid"]
        is False
    )


def test_bright():

    image = Image.new(
        "RGB",
        (256, 256),
        (250, 250, 250),
    )

    result = validate_image_quality(
        image
    )

    show(
        "BRIGHT IMAGE",
        result
    )

    assert (
        result["image_quality"]["valid"]
        is False
    )


def test_blurry():

    image = Image.new(
        "RGB",
        (256, 256),
        (120, 120, 120),
    )

    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=10
        )
    )

    result = validate_image_quality(
        image
    )

    show(
        "BLURRY IMAGE",
        result
    )

    assert (
        result["image_quality"]["valid"]
        is False
    )


def test_real_leaf():

    image_path = (
        TEST_IMAGES / "leaf.jpg"
    )

    if not image_path.exists():

        print(
            "\nWARNING:"
            " test_images/leaf.jpg "
            "does not exist."
        )

        return

    result = validate_image_quality(
        image_path
    )

    show(
        "REAL LEAF IMAGE",
        result
    )


def main():

    print("=" * 60)
    print("P1.2 IMAGE QUALITY TEST")
    print("=" * 60)

    test_missing()
    test_invalid()
    test_tiny()
    test_dark()
    test_bright()
    test_blurry()
    test_real_leaf()

    print()
    print("=" * 60)
    print("P1.2 IMAGE QUALITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()