"""
P1.3 - Leaf / Image Suitability Detection
==========================================

Conservative heuristic suitability gate for plant-disease
images.

IMPORTANT:
    This is NOT a trained leaf-vs-non-leaf classifier.

    The existing EfficientNet-B0 model is trained for disease
    classification across the 38 PlantVillage classes.

    This module estimates whether an image visually contains
    enough vegetation-like content to reasonably proceed toward
    plant disease prediction.

Possible results:

    suitable
        Image appears suitable for plant/leaf prediction.

    review
        Image is ambiguous and should be reviewed.

    unsuitable
        Image has strong indications that it is not suitable.

IMPORTANT:
    All thresholds are application-level heuristics.
    They are not calibrated probabilities.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any, Union
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Minimum resolution
# ------------------------------------------------------------

MIN_WIDTH = 128
MIN_HEIGHT = 128


# ------------------------------------------------------------
# Vegetation thresholds
# ------------------------------------------------------------

SUITABLE_VEGETATION_RATIO = 0.12
REVIEW_VEGETATION_RATIO = 0.05


# ------------------------------------------------------------
# Green color thresholds
# ------------------------------------------------------------

GREEN_HUE_LOW = 25
GREEN_HUE_HIGH = 100

GREEN_SATURATION_MIN = 35
GREEN_VALUE_MIN = 25


# ------------------------------------------------------------
# Yellow / brown plant regions
# ------------------------------------------------------------

YELLOW_HUE_LOW = 15
YELLOW_HUE_HIGH = 40

BROWN_HUE_LOW = 5
BROWN_HUE_HIGH = 25

PLANT_SATURATION_MIN = 30
PLANT_VALUE_MIN = 25


# ------------------------------------------------------------
# Image variation
# ------------------------------------------------------------

UNIFORMITY_STD_THRESHOLD = 12.0


# ------------------------------------------------------------
# Texture
# ------------------------------------------------------------

TEXTURE_LOW_THRESHOLD = 20.0


# ------------------------------------------------------------
# Spatial coherence
# ------------------------------------------------------------

# Minimum percentage of the image that must belong to the
# largest meaningful vegetation-like connected region.

MIN_LARGEST_VEGETATION_COMPONENT_RATIO = 0.03

MIN_VEGETATION_COMPONENT_COUNT = 1


# ============================================================
# IMAGE LOADING
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
    """
    Load an image from:

        - file path
        - bytes
        - bytearray
        - PIL Image

    Returns:
        RGB PIL Image.
    """

    # --------------------------------------------------------
    # PIL Image
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        return image.convert("RGB")

    # --------------------------------------------------------
    # File path
    # --------------------------------------------------------

    if isinstance(image, (str, Path)):

        path = Path(image)

        if not path.exists():

            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Image path is not a file: {path}"
            )

        try:

            # Verify image.
            with Image.open(path) as img:
                img.verify()

            # Reopen after verify().
            with Image.open(path) as img:
                return img.convert("RGB")

        except UnidentifiedImageError:

            raise ValueError(
                "File is not a valid image."
            )

    # --------------------------------------------------------
    # Bytes
    # --------------------------------------------------------

    if isinstance(image, (bytes, bytearray)):

        try:

            raw_bytes = bytes(image)

            buffer = io.BytesIO(
                raw_bytes
            )

            with Image.open(buffer) as img:
                img.verify()

            buffer = io.BytesIO(
                raw_bytes
            )

            with Image.open(buffer) as img:
                return img.convert("RGB")

        except UnidentifiedImageError:

            raise ValueError(
                "Input bytes are not a valid image."
            )

    # --------------------------------------------------------
    # Unsupported input
    # --------------------------------------------------------

    raise TypeError(
        "Unsupported image input. "
        "Use a file path, bytes, bytearray, "
        "or PIL Image."
    )


# ============================================================
# VEGETATION RATIO
# ============================================================

def calculate_vegetation_ratio(
    image: Image.Image
) -> float:
    """
    Estimate the percentage of pixels that look
    vegetation-like.

    Includes:
        green
        yellow
        brown

    Returns:
        Value between 0.0 and 1.0.
    """

    rgb = np.asarray(image)

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Green
    green_mask = (
        (hue >= GREEN_HUE_LOW)
        & (hue <= GREEN_HUE_HIGH)
        & (saturation >= GREEN_SATURATION_MIN)
        & (value >= GREEN_VALUE_MIN)
    )

    # Yellow
    yellow_mask = (
        (hue >= YELLOW_HUE_LOW)
        & (hue <= YELLOW_HUE_HIGH)
        & (saturation >= PLANT_SATURATION_MIN)
        & (value >= PLANT_VALUE_MIN)
    )

    # Brown
    brown_mask = (
        (hue >= BROWN_HUE_LOW)
        & (hue <= BROWN_HUE_HIGH)
        & (saturation >= PLANT_SATURATION_MIN)
        & (value >= PLANT_VALUE_MIN)
    )

    vegetation_mask = (
        green_mask
        | yellow_mask
        | brown_mask
    )

    return float(
        vegetation_mask.mean()
    )


# ============================================================
# GREEN RATIO
# ============================================================

def calculate_green_ratio(
    image: Image.Image
) -> float:
    """
    Calculate percentage of specifically green pixels.

    Returns:
        Value between 0.0 and 1.0.
    """

    rgb = np.asarray(image)

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    green_mask = (
        (hue >= GREEN_HUE_LOW)
        & (hue <= GREEN_HUE_HIGH)
        & (saturation >= GREEN_SATURATION_MIN)
        & (value >= GREEN_VALUE_MIN)
    )

    return float(
        green_mask.mean()
    )


# ============================================================
# COLOR VARIATION
# ============================================================

def calculate_color_variation(
    image: Image.Image
) -> float:
    """
    Calculate average RGB channel variation.

    Low value:
        visually uniform image.

    Higher value:
        greater visual/color variation.
    """

    rgb = np.asarray(
        image
    ).astype(
        np.float32
    )

    channel_std = np.std(
        rgb,
        axis=(0, 1),
    )

    return float(
        np.mean(channel_std)
    )


# ============================================================
# TEXTURE
# ============================================================

def calculate_texture_score(
    image: Image.Image
) -> float:
    """
    Estimate image texture using variance of Laplacian.

    This is an additional suitability signal.
    """

    rgb = np.asarray(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    return float(
        laplacian.var()
    )


# ============================================================
# SPATIAL COHERENCE
# ============================================================

def calculate_vegetation_spatial_coherence(
    image: Image.Image
) -> tuple[float, int]:
    """
    Determine whether vegetation-like pixels form meaningful
    connected regions.

    Returns:

        largest_component_ratio
            Area of the largest meaningful vegetation region
            divided by total image area.

        component_count
            Number of meaningful vegetation regions.

    This helps distinguish:

        real vegetation regions

    from:

        randomly scattered vegetation-colored pixels.
    """

    rgb = np.asarray(image)

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # --------------------------------------------------------
    # Green
    # --------------------------------------------------------

    green_mask = (
        (hue >= GREEN_HUE_LOW)
        & (hue <= GREEN_HUE_HIGH)
        & (saturation >= GREEN_SATURATION_MIN)
        & (value >= GREEN_VALUE_MIN)
    )

    # --------------------------------------------------------
    # Yellow
    # --------------------------------------------------------

    yellow_mask = (
        (hue >= YELLOW_HUE_LOW)
        & (hue <= YELLOW_HUE_HIGH)
        & (saturation >= PLANT_SATURATION_MIN)
        & (value >= PLANT_VALUE_MIN)
    )

    # --------------------------------------------------------
    # Brown
    # --------------------------------------------------------

    brown_mask = (
        (hue >= BROWN_HUE_LOW)
        & (hue <= BROWN_HUE_HIGH)
        & (saturation >= PLANT_SATURATION_MIN)
        & (value >= PLANT_VALUE_MIN)
    )

    # --------------------------------------------------------
    # Combined mask
    # --------------------------------------------------------

    vegetation_mask = (
        green_mask
        | yellow_mask
        | brown_mask
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    # Remove isolated pixels.
    vegetation_mask = cv2.morphologyEx(
        vegetation_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Join nearby vegetation pixels.
    vegetation_mask = cv2.morphologyEx(
        vegetation_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            vegetation_mask,
            connectivity=8,
        )
    )

    # Only background exists.
    if num_labels <= 1:

        return 0.0, 0

    image_area = (
        vegetation_mask.shape[0]
        * vegetation_mask.shape[1]
    )

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    # Ignore tiny noise components.
    minimum_component_area = max(
        int(image_area * 0.005),
        50,
    )

    meaningful_components = (
        component_areas[
            component_areas
            >= minimum_component_area
        ]
    )

    if len(meaningful_components) == 0:

        return 0.0, 0

    largest_component = float(
        np.max(
            meaningful_components
        )
    )

    largest_component_ratio = (
        largest_component
        / image_area
    )

    return (
        largest_component_ratio,
        int(
            len(
                meaningful_components
            )
        ),
    )


# ============================================================
# SUITABILITY ANALYSIS
# ============================================================

def analyze_suitability(
    image: Image.Image
) -> dict[str, Any]:
    """
    Analyze image suitability.

    Resolution is treated as a HARD requirement.
    """

    width, height = image.size

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    vegetation_ratio = (
        calculate_vegetation_ratio(
            image
        )
    )

    green_ratio = (
        calculate_green_ratio(
            image
        )
    )

    color_variation = (
        calculate_color_variation(
            image
        )
    )

    texture_score = (
        calculate_texture_score(
            image
        )
    )

    (
        largest_component_ratio,
        component_count,
    ) = (
        calculate_vegetation_spatial_coherence(
            image
        )
    )

    issues: list[str] = []

    # ========================================================
    # RESOLUTION CHECK
    # ========================================================

    resolution_valid = (
        width >= MIN_WIDTH
        and height >= MIN_HEIGHT
    )

    if not resolution_valid:

        issues.append(
            "Image resolution is too low"
        )

    # ========================================================
    # VISUAL VARIATION
    # ========================================================

    if (
        color_variation
        < UNIFORMITY_STD_THRESHOLD
    ):

        issues.append(
            "Image has very little visual variation"
        )

    # ========================================================
    # VEGETATION
    # ========================================================

    if (
        vegetation_ratio
        < REVIEW_VEGETATION_RATIO
    ):

        issues.append(
            "Very little vegetation-like "
            "content detected"
        )

    elif (
        vegetation_ratio
        < SUITABLE_VEGETATION_RATIO
    ):

        issues.append(
            "Vegetation-like content is limited"
        )

    # ========================================================
    # TEXTURE
    # ========================================================

    if (
        texture_score
        < TEXTURE_LOW_THRESHOLD
    ):

        issues.append(
            "Image contains very little texture"
        )

    # ========================================================
    # SPATIAL COHERENCE
    # ========================================================

    if (
        largest_component_ratio
        < MIN_LARGEST_VEGETATION_COMPONENT_RATIO
    ):

        issues.append(
            "Vegetation-like regions are not "
            "spatially coherent"
        )

    # ========================================================
    # DECISION
    # ========================================================
    #
    # Resolution is a HARD requirement.
    #
    # Therefore:
    #
    #     image < 128x128
    #         -> unsuitable
    #
    # regardless of vegetation/color score.
    #
    # ========================================================

    if not resolution_valid:

        suitability = "unsuitable"

    elif (
        vegetation_ratio
        < REVIEW_VEGETATION_RATIO
        and largest_component_ratio
        < MIN_LARGEST_VEGETATION_COMPONENT_RATIO
    ):

        suitability = "unsuitable"

    elif (
        largest_component_ratio
        < MIN_LARGEST_VEGETATION_COMPONENT_RATIO
    ):

        suitability = "review"

    elif (
        vegetation_ratio
        < REVIEW_VEGETATION_RATIO
    ):

        suitability = "review"

    elif (
        vegetation_ratio
        < SUITABLE_VEGETATION_RATIO
    ):

        suitability = "review"

    else:

        suitability = "suitable"

    # ========================================================
    # HEURISTIC SCORE
    # ========================================================
    #
    # IMPORTANT:
    #
    # This score is NOT:
    #
    #     probability
    #
    # It is only a diagnostic score from 0-100.
    #
    # ========================================================

    vegetation_component = min(
        vegetation_ratio
        / SUITABLE_VEGETATION_RATIO,
        1.0,
    )

    spatial_component = min(
        largest_component_ratio
        / MIN_LARGEST_VEGETATION_COMPONENT_RATIO,
        1.0,
    )

    texture_component = min(
        texture_score
        / 100.0,
        1.0,
    )

    variation_component = min(
        color_variation
        / 50.0,
        1.0,
    )

    suitability_score = (
        0.45 * vegetation_component
        + 0.30 * spatial_component
        + 0.10 * texture_component
        + 0.15 * variation_component
    ) * 100.0

    # --------------------------------------------------------
    # HARD RESOLUTION FAILURE
    # --------------------------------------------------------
    #
    # A tiny image cannot receive a positive suitability score.
    #

    if not resolution_valid:

        suitability_score = 0.0

    suitability_score = min(
        max(
            suitability_score,
            0.0,
        ),
        100.0,
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "suitability": {

            "status": suitability,

            "score": round(
                float(
                    suitability_score
                ),
                2,
            ),

            "width": width,

            "height": height,

            "vegetation_ratio": round(
                vegetation_ratio,
                4,
            ),

            "green_ratio": round(
                green_ratio,
                4,
            ),

            "largest_vegetation_component_ratio": round(
                largest_component_ratio,
                4,
            ),

            "vegetation_component_count": (
                component_count
            ),

            "color_variation": round(
                color_variation,
                2,
            ),

            "texture_score": round(
                texture_score,
                2,
            ),

            "issues": issues,
        }
    }


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_image_suitability(
    image: Union[
        str,
        Path,
        bytes,
        bytearray,
        Image.Image,
    ]
) -> dict[str, Any]:
    """
    Validate whether an image appears suitable for
    plant disease prediction.
    """

    try:

        pil_image = load_image(
            image
        )

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as exc:

        return {
            "suitability": {

                "status": "unsuitable",

                "score": 0.0,

                "width": None,

                "height": None,

                "vegetation_ratio": None,

                "green_ratio": None,

                "largest_vegetation_component_ratio": None,

                "vegetation_component_count": 0,

                "color_variation": None,

                "texture_score": None,

                "issues": [
                    str(exc)
                ],
            }
        }

    return analyze_suitability(
        pil_image
    )


# ============================================================
# SIMPLE HELPER
# ============================================================

def is_likely_leaf_image(
    image
) -> bool:
    """
    Return True only when the result is 'suitable'.

    'review' is intentionally treated as False so that
    ambiguous images are not automatically sent to the
    disease classifier.
    """

    result = (
        validate_image_suitability(
            image
        )
    )

    return (
        result["suitability"]["status"]
        == "suitable"
    )


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
            "python src/image_suitability.py "
            "<image_path>"
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    result = (
        validate_image_suitability(
            image_path
        )
    )

    suitability = (
        result["suitability"]
    )

    print("=" * 60)

    print(
        "P1.3 LEAF / IMAGE SUITABILITY"
    )

    print("=" * 60)

    print(
        f"Status:              "
        f"{suitability['status']}"
    )

    print(
        f"Suitability score:   "
        f"{suitability['score']}"
    )

    print(
        f"Resolution:          "
        f"{suitability['width']}x"
        f"{suitability['height']}"
    )

    print(
        f"Vegetation ratio:    "
        f"{suitability['vegetation_ratio']}"
    )

    print(
        f"Green ratio:         "
        f"{suitability['green_ratio']}"
    )

    print(
        f"Largest vegetation "
        f"component ratio:    "
        f"{suitability['largest_vegetation_component_ratio']}"
    )

    print(
        f"Vegetation components:"
        f" {suitability['vegetation_component_count']}"
    )

    print(
        f"Color variation:     "
        f"{suitability['color_variation']}"
    )

    print(
        f"Texture score:       "
        f"{suitability['texture_score']}"
    )

    if suitability["issues"]:

        print("Issues:")

        for issue in suitability[
            "issues"
        ]:

            print(
                f"  - {issue}"
            )

    else:

        print(
            "Issues:              None"
        )

    print("=" * 60)