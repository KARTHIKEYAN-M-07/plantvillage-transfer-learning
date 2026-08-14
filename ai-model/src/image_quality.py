"""
Image Quality Validation
========================

Checks whether an image is technically suitable for
plant-disease prediction.

Checks:
    - image existence
    - valid image
    - minimum resolution
    - blur
    - darkness
    - overexposure

This module does NOT predict disease.

All thresholds are heuristics.
"""


from pathlib import Path
from typing import Union, Any
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

MIN_WIDTH = 128
MIN_HEIGHT = 128

BLUR_POOR_THRESHOLD = 50.0
BLUR_WARNING_THRESHOLD = 100.0

DARK_THRESHOLD = 0.15
DARK_WARNING_THRESHOLD = 0.25

BRIGHT_THRESHOLD = 0.90
BRIGHT_WARNING_THRESHOLD = 0.80


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image: Union[
        str,
        Path,
        bytes,
        bytearray,
        Image.Image,
    ]
) -> Image.Image:

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):

        path = Path(image)

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        try:

            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                return img.convert("RGB")

        except UnidentifiedImageError:
            raise ValueError(
                "File is not a valid image."
            )

    if isinstance(image, (bytes, bytearray)):

        try:

            with Image.open(
                io.BytesIO(image)
            ) as img:

                img.verify()

            with Image.open(
                io.BytesIO(image)
            ) as img:

                return img.convert("RGB")

        except UnidentifiedImageError:

            raise ValueError(
                "Input bytes are not a valid image."
            )

    raise TypeError(
        "Unsupported image input."
    )


# ============================================================
# BLUR
# ============================================================

def calculate_blur_score(
    image: Image.Image
) -> float:

    array = np.asarray(image)

    gray = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2GRAY,
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    return float(laplacian.var())


# ============================================================
# BRIGHTNESS
# ============================================================

def calculate_brightness(
    image: Image.Image
) -> float:

    array = np.asarray(image)

    gray = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2GRAY,
    )

    return float(gray.mean() / 255.0)


# ============================================================
# VALIDATE
# ============================================================

def validate_image_quality(
    image: Union[
        str,
        Path,
        bytes,
        bytearray,
        Image.Image,
    ]
) -> dict[str, Any]:

    try:

        pil_image = load_image(image)

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as exc:

        return {
            "image_quality": {
                "valid": False,
                "quality_level": "poor",
                "width": None,
                "height": None,
                "blur_score": None,
                "brightness": None,
                "issues": [str(exc)],
            }
        }

    width, height = pil_image.size

    issues = []

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    if (
        width < MIN_WIDTH
        or height < MIN_HEIGHT
    ):

        issues.append(
            f"Image resolution is too low "
            f"({width}x{height}); "
            f"minimum is "
            f"{MIN_WIDTH}x{MIN_HEIGHT}"
        )

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blur_score = calculate_blur_score(
        pil_image
    )

    if blur_score < BLUR_POOR_THRESHOLD:

        issues.append(
            "Image is extremely blurry"
        )

    elif blur_score < BLUR_WARNING_THRESHOLD:

        issues.append(
            "Image may be blurry"
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = calculate_brightness(
        pil_image
    )

    if brightness < DARK_THRESHOLD:

        issues.append(
            "Image is too dark"
        )

    elif brightness < DARK_WARNING_THRESHOLD:

        issues.append(
            "Image may be too dark"
        )

    elif brightness > BRIGHT_THRESHOLD:

        issues.append(
            "Image is too bright/overexposed"
        )

    elif brightness > BRIGHT_WARNING_THRESHOLD:

        issues.append(
            "Image may be too bright"
        )

    # --------------------------------------------------------
    # Quality classification
    # --------------------------------------------------------

    critical = False

    if width < MIN_WIDTH:
        critical = True

    if height < MIN_HEIGHT:
        critical = True

    if blur_score < BLUR_POOR_THRESHOLD:
        critical = True

    if brightness < DARK_THRESHOLD:
        critical = True

    if brightness > BRIGHT_THRESHOLD:
        critical = True

    if critical:

        quality_level = "poor"
        valid = False

    elif issues:

        quality_level = "moderate"
        valid = True

    else:

        quality_level = "good"
        valid = True

    return {
        "image_quality": {
            "valid": valid,
            "quality_level": quality_level,
            "width": width,
            "height": height,
            "blur_score": round(
                blur_score,
                2,
            ),
            "brightness": round(
                brightness,
                4,
            ),
            "issues": issues,
        }
    }


# ============================================================
# SIMPLE HELPER
# ============================================================

def is_image_suitable(image) -> bool:

    result = validate_image_quality(image)

    return result["image_quality"]["valid"]


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src/image_quality.py "
            "<image_path>"
        )

        raise SystemExit(1)

    result = validate_image_quality(
        sys.argv[1]
    )

    quality = result["image_quality"]

    print("=" * 60)
    print("IMAGE QUALITY VALIDATION")
    print("=" * 60)

    print(
        f"Valid:          "
        f"{quality['valid']}"
    )

    print(
        f"Quality level:  "
        f"{quality['quality_level']}"
    )

    print(
        f"Resolution:     "
        f"{quality['width']}x"
        f"{quality['height']}"
    )

    print(
        f"Blur score:     "
        f"{quality['blur_score']}"
    )

    print(
        f"Brightness:     "
        f"{quality['brightness']}"
    )

    if quality["issues"]:

        print("Issues:")

        for issue in quality["issues"]:
            print(f"  - {issue}")

    else:

        print("Issues:         None")

    print("=" * 60)