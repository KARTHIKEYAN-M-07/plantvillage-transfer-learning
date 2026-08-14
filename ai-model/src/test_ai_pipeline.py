"""
AI PIPELINE PUBLIC CONTRACT TEST
================================

This test verifies the PUBLIC AI interface:

    ai_pipeline.predict()

The test is intentionally independent of the full
PlantVillage dataset.

Required test images:

    test_images/
    ├── apple_scab.jpg
    └── leaf.jpg

The PlantVillage training dataset is NOT required
to run this test.

Tests:

1. Apple Scab known prediction
2. Real leaf moderate-confidence prediction
3. Missing image handling
4. Complete AI contract structure
5. Image quality
6. Image suitability
7. Confidence handling
8. Grad-CAM
9. Disease information

IMPORTANT
---------
This test does NOT retrain the model.
It does NOT modify model weights.
"""

from __future__ import annotations

from pathlib import Path
import sys


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
# IMPORT PUBLIC AI INTERFACE
# ============================================================

from ai_pipeline import (
    predict,
)


# ============================================================
# REPOSITORY TEST IMAGES
# ============================================================

TEST_IMAGES_DIR = (
    PROJECT_ROOT
    / "test_images"
)


APPLE_SCAB_IMAGE = (
    TEST_IMAGES_DIR
    / "apple_scab.jpg"
)


REAL_LEAF_IMAGE = (
    TEST_IMAGES_DIR
    / "leaf.jpg"
)


# ============================================================
# REQUIRED CONTRACT SECTIONS
# ============================================================

REQUIRED_SECTIONS = [

    "success",

    "model",

    "image",

    "image_quality",

    "suitability",

    "prediction",

    "explanation",

    "disease_information",
]


# ============================================================
# VERIFY TEST FIXTURES
# ============================================================

def verify_test_images():

    print()
    print(
        "VERIFYING TEST IMAGES"
    )

    print(
        "-" * 60
    )

    assert TEST_IMAGES_DIR.exists(), (
        "test_images directory does not exist."
    )

    assert APPLE_SCAB_IMAGE.exists(), (
        "Missing test image: "
        f"{APPLE_SCAB_IMAGE}"
    )

    assert REAL_LEAF_IMAGE.exists(), (
        "Missing test image: "
        f"{REAL_LEAF_IMAGE}"
    )

    print(
        "Apple Scab image:",
        APPLE_SCAB_IMAGE.name
    )

    print(
        "Real leaf image:",
        REAL_LEAF_IMAGE.name
    )

    print(
        "Test images verified."
    )


# ============================================================
# CONTRACT VALIDATION
# ============================================================

def validate_contract(
    result: dict,
):

    # --------------------------------------------------------
    # Result must be dictionary
    # --------------------------------------------------------

    assert isinstance(
        result,
        dict,
    ), (
        "AI result must be a dictionary."
    )

    # --------------------------------------------------------
    # Successful result
    # --------------------------------------------------------

    assert (
        result.get("success")
        is True
    ), (
        "AI prediction was not successful."
    )

    # --------------------------------------------------------
    # Required top-level sections
    # --------------------------------------------------------

    for section in REQUIRED_SECTIONS:

        assert section in result, (
            f"Missing contract section: "
            f"{section}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = result[
        "model"
    ]

    assert (
        model["name"]
        == "EfficientNet-B0"
    ), (
        "Incorrect model name."
    )

    assert (
        model["classes"]
        == 38
    ), (
        "Incorrect number of classes."
    )

    # ========================================================
    # IMAGE
    # ========================================================

    image = result[
        "image"
    ]

    assert isinstance(
        image["width"],
        int,
    ), (
        "Image width must be an integer."
    )

    assert isinstance(
        image["height"],
        int,
    ), (
        "Image height must be an integer."
    )

    assert (
        image["width"]
        > 0
    )

    assert (
        image["height"]
        > 0
    )

    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    quality = result[
        "image_quality"
    ]

    assert isinstance(
        quality["valid"],
        bool,
    ), (
        "image_quality.valid must be boolean."
    )

    assert quality[
        "quality_level"
    ] in [
        "good",
        "acceptable",
        "poor",
    ], (
        "Invalid quality level."
    )

    assert (
        "blur_score"
        in quality
    )

    assert (
        "brightness"
        in quality
    )

    assert isinstance(
        quality["issues"],
        list,
    ), (
        "Image quality issues must be a list."
    )

    # ========================================================
    # SUITABILITY
    # ========================================================

    suitability = result[
        "suitability"
    ]

    assert suitability[
        "status"
    ] in [
        "suitable",
        "review",
        "unsuitable",
    ], (
        "Invalid suitability status."
    )

    assert isinstance(
        suitability["issues"],
        list,
    ), (
        "Suitability issues must be a list."
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = result[
        "prediction"
    ]

    assert isinstance(
        prediction["class_name"],
        str,
    ), (
        "class_name must be a string."
    )

    assert isinstance(
        prediction["plant"],
        str,
    ), (
        "plant must be a string."
    )

    assert isinstance(
        prediction["disease"],
        str,
    ), (
        "disease must be a string."
    )

    confidence = float(
        prediction[
            "confidence"
        ]
    )

    assert (
        0.0
        <= confidence
        <= 100.0
    ), (
        "Confidence must be between "
        "0 and 100."
    )

    assert prediction[
        "confidence_level"
    ] in [
        "high",
        "moderate",
        "low",
    ], (
        "Invalid confidence level."
    )

    assert isinstance(
        prediction[
            "needs_review"
        ],
        bool,
    ), (
        "needs_review must be boolean."
    )

    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = result[
        "explanation"
    ]

    assert isinstance(
        explanation["available"],
        bool,
    ), (
        "explanation.available must be boolean."
    )

    assert (
        explanation["method"]
        == "Grad-CAM"
    ), (
        "Explanation method must be Grad-CAM."
    )

    assert isinstance(
        explanation[
            "target_class"
        ],
        str,
    ), (
        "target_class must be a string."
    )

    if explanation[
        "available"
    ]:

        assert (
            explanation[
                "heatmap_path"
            ]
            is not None
        ), (
            "Grad-CAM heatmap path is missing."
        )

        heatmap_path = Path(
            explanation[
                "heatmap_path"
            ]
        )

        assert heatmap_path.exists(), (
            "Grad-CAM heatmap file does not exist: "
            f"{heatmap_path}"
        )

    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    disease_info = result[
        "disease_information"
    ]

    required_disease_fields = [

        "status",

        "severity",

        "description",

        "symptoms",

        "general_management",

        "prevention",
    ]

    for field in required_disease_fields:

        assert field in disease_info, (
            f"Missing disease information field: "
            f"{field}"
        )

    assert disease_info[
        "status"
    ] in [
        "healthy",
        "diseased",
    ], (
        "Invalid disease status."
    )

    assert isinstance(
        disease_info[
            "symptoms"
        ],
        list,
    )

    assert isinstance(
        disease_info[
            "general_management"
        ],
        list,
    )

    assert isinstance(
        disease_info[
            "prevention"
        ],
        list,
    )


# ============================================================
# TEST 1 — APPLE SCAB
# ============================================================

def test_apple_scab():

    print()
    print(
        "-" * 60
    )

    print(
        "TEST 1: APPLE SCAB"
    )

    print(
        "-" * 60
    )

    result = predict(
        APPLE_SCAB_IMAGE,
        generate_gradcam=True,
        gradcam_filename=(
            "contract_apple_scab_gradcam.jpg"
        ),
    )

    validate_contract(
        result
    )

    prediction = result[
        "prediction"
    ]

    print(
        "Plant:",
        prediction[
            "plant"
        ]
    )

    print(
        "Disease:",
        prediction[
            "disease"
        ]
    )

    print(
        "Confidence:",
        prediction[
            "confidence"
        ],
        "%",
    )

    print(
        "Confidence level:",
        prediction[
            "confidence_level"
        ]
    )

    print(
        "Needs review:",
        prediction[
            "needs_review"
        ]
    )

    # ========================================================
    # KNOWN APPLE SCAB RESULT
    # ========================================================

    assert (
        prediction[
            "class_name"
        ]
        == "Apple___Apple_scab"
    ), (
        "Apple Scab test predicted "
        "the wrong class."
    )

    assert (
        prediction[
            "plant"
        ]
        == "Apple"
    )

    assert (
        prediction[
            "disease"
        ]
        == "Apple scab"
    )

    assert (
        prediction[
            "confidence_level"
        ]
        == "high"
    )

    assert (
        prediction[
            "needs_review"
        ]
        is False
    )

    # --------------------------------------------------------
    # Image quality
    # --------------------------------------------------------

    assert (
        result[
            "image_quality"
        ][
            "valid"
        ]
        is True
    )

    # --------------------------------------------------------
    # Suitability
    # --------------------------------------------------------

    assert (
        result[
            "suitability"
        ][
            "status"
        ]
        == "suitable"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    assert (
        result[
            "explanation"
        ][
            "available"
        ]
        is True
    )

    # --------------------------------------------------------
    # Disease information
    # --------------------------------------------------------

    assert (
        result[
            "disease_information"
        ][
            "status"
        ]
        == "diseased"
    )

    print(
        "Apple Scab contract: PASSED"
    )


# ============================================================
# TEST 2 — REAL LEAF
# ============================================================

def test_real_leaf():

    print()
    print(
        "-" * 60
    )

    print(
        "TEST 2: REAL LEAF"
    )

    print(
        "-" * 60
    )

    result = predict(
        REAL_LEAF_IMAGE,
        generate_gradcam=True,
        gradcam_filename=(
            "contract_real_leaf_gradcam.jpg"
        ),
    )

    validate_contract(
        result
    )

    prediction = result[
        "prediction"
    ]

    print(
        "Plant:",
        prediction[
            "plant"
        ]
    )

    print(
        "Disease:",
        prediction[
            "disease"
        ]
    )

    print(
        "Confidence:",
        prediction[
            "confidence"
        ],
        "%",
    )

    print(
        "Confidence level:",
        prediction[
            "confidence_level"
        ]
    )

    print(
        "Needs review:",
        prediction[
            "needs_review"
        ]
    )

    # ========================================================
    # KNOWN REAL LEAF RESULT
    # ========================================================

    assert (
        prediction[
            "class_name"
        ]
        == "Tomato___Early_blight"
    ), (
        "Real leaf test predicted "
        "the wrong class."
    )

    assert (
        prediction[
            "plant"
        ]
        == "Tomato"
    )

    assert (
        prediction[
            "disease"
        ]
        == "Early blight"
    )

    assert (
        prediction[
            "confidence_level"
        ]
        == "moderate"
    )

    assert (
        prediction[
            "needs_review"
        ]
        is True
    )

    # --------------------------------------------------------
    # Image quality
    # --------------------------------------------------------

    assert (
        result[
            "image_quality"
        ][
            "valid"
        ]
        is True
    )

    # --------------------------------------------------------
    # Suitability
    # --------------------------------------------------------

    assert (
        result[
            "suitability"
        ][
            "status"
        ]
        == "suitable"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    assert (
        result[
            "explanation"
        ][
            "available"
        ]
        is True
    )

    print(
        "Real leaf contract: PASSED"
    )


# ============================================================
# TEST 3 — MISSING IMAGE
# ============================================================

def test_missing_image():

    print()
    print(
        "-" * 60
    )

    print(
        "TEST 3: MISSING IMAGE"
    )

    print(
        "-" * 60
    )

    missing_image = (
        TEST_IMAGES_DIR
        / "does_not_exist.jpg"
    )

    result = predict(
        missing_image,
        generate_gradcam=False,
    )

    assert (
        result["success"]
        is False
    ), (
        "Missing image should fail."
    )

    assert (
        result[
            "error"
        ][
            "stage"
        ]
        == "input"
    ), (
        "Missing image should fail "
        "at input validation."
    )

    print(
        "Missing image handled correctly."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "AI PUBLIC CONTRACT TEST"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Verify repository test fixtures first
    # --------------------------------------------------------

    verify_test_images()

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    test_apple_scab()

    test_real_leaf()

    test_missing_image()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "AI PUBLIC CONTRACT TEST PASSED"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "AI CONTRACT IS READY FOR FREEZE."
    )

    print()

    print(
        "The tests are independent of "
        "data/PlantVillage/."
    )

    print(
        "The repository test fixtures are:"
    )

    print(
        f"  {APPLE_SCAB_IMAGE}"
    )

    print(
        f"  {REAL_LEAF_IMAGE}"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()