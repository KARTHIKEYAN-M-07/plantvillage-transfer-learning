"""
FINAL AI PIPELINE INTEGRATION TEST
==================================

Complete AI-side validation:

1. Image quality
2. Leaf/image suitability
3. EfficientNet-B0 prediction
4. Confidence handling
5. Grad-CAM
6. Disease information

No retraining.
No model modification.
No backend modification.
"""

from __future__ import annotations

from pathlib import Path
import sys
import json


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from image_quality import (
    validate_image_quality,
)

from image_suitability import (
    validate_image_suitability,
)

from predictor import (
    PlantDiseasePredictor,
)

from gradcam import (
    save_gradcam,
)

from disease_information import (
    get_disease_information,
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIDENCE HANDLING
# ============================================================

def get_confidence_level(
    confidence: float,
) -> str:

    if confidence >= 80.0:
        return "high"

    if confidence >= 50.0:
        return "moderate"

    return "low"


def needs_review(
    confidence_level: str,
) -> bool:

    return confidence_level != "high"


# ============================================================
# PRINT SECTION
# ============================================================

def print_section(
    title: str,
):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# EXTRACT IMAGE QUALITY
# ============================================================

def extract_quality(
    result: dict,
) -> dict:

    if not isinstance(result, dict):

        raise RuntimeError(
            "Image quality returned "
            "an invalid result."
        )

    if "image_quality" in result:

        return result[
            "image_quality"
        ]

    return result


# ============================================================
# EXTRACT SUITABILITY
# ============================================================

def extract_suitability(
    result: dict,
) -> dict:

    if not isinstance(result, dict):

        raise RuntimeError(
            "Image suitability returned "
            "an invalid result."
        )

    if "suitability" in result:

        return result[
            "suitability"
        ]

    return result


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

def run_pipeline(
    image_path: Path,
    gradcam_filename: str,
) -> dict:

    print_section(
        f"PIPELINE: {image_path.name}"
    )

    # ========================================================
    # 1. IMAGE QUALITY
    # ========================================================

    print(
        "\n[1/6] IMAGE QUALITY"
    )

    quality_raw = (
        validate_image_quality(
            image_path
        )
    )

    quality = extract_quality(
        quality_raw
    )

    print(
        "Valid:",
        quality.get("valid")
    )

    print(
        "Quality level:",
        quality.get(
            "quality_level"
        )
    )

    print(
        "Resolution:",
        quality.get(
            "width"
        ),
        "x",
        quality.get(
            "height"
        )
    )

    print(
        "Blur score:",
        quality.get(
            "blur_score"
        )
    )

    print(
        "Brightness:",
        quality.get(
            "brightness"
        )
    )

    print(
        "Issues:",
        quality.get(
            "issues"
        )
    )

    # --------------------------------------------------------
    # Quality must be valid
    # --------------------------------------------------------

    if not quality.get(
        "valid",
        False,
    ):

        return {
            "success": False,
            "stage": "image_quality",
            "image_quality": quality,
        }

    # ========================================================
    # 2. LEAF SUITABILITY
    # ========================================================

    print(
        "\n[2/6] LEAF / IMAGE SUITABILITY"
    )

    suitability_raw = (
        validate_image_suitability(
            image_path
        )
    )

    suitability = (
        extract_suitability(
            suitability_raw
        )
    )

    print(
        "Status:",
        suitability.get(
            "status"
        )
    )

    print(
        "Score:",
        suitability.get(
            "suitability_score",
            suitability.get(
                "score"
            )
        )
    )

    print(
        "Vegetation ratio:",
        suitability.get(
            "vegetation_ratio"
        )
    )

    print(
        "Issues:",
        suitability.get(
            "issues"
        )
    )

    # --------------------------------------------------------
    # Reject only truly unsuitable images.
    #
    # "review" is allowed to continue because your P1.3
    # design intentionally distinguishes review from unsuitable.
    # --------------------------------------------------------

    if suitability.get(
        "status"
    ) == "unsuitable":

        return {
            "success": False,
            "stage": "image_suitability",
            "image_quality": quality,
            "suitability": suitability,
        }

    # ========================================================
    # 3. EFFICIENTNET PREDICTION
    # ========================================================

    print(
        "\n[3/6] EFFICIENTNET-B0 PREDICTION"
    )

    predictor = (
        PlantDiseasePredictor()
    )

    prediction_result = (
        predictor.predict(
            str(image_path)
        )
    )

    if not isinstance(
        prediction_result,
        dict,
    ):

        raise RuntimeError(
            "Predictor returned "
            "an invalid result."
        )

    if not prediction_result.get(
        "success",
        False,
    ):

        raise RuntimeError(
            "Predictor failed:\n"
            + json.dumps(
                prediction_result,
                indent=2,
            )
        )

    prediction = (
        prediction_result[
            "prediction"
        ]
    )

    class_name = (
        prediction[
            "class_name"
        ]
    )

    plant = (
        prediction[
            "plant"
        ]
    )

    disease = (
        prediction[
            "disease"
        ]
    )

    confidence = float(
        prediction[
            "confidence"
        ]
    )

    print(
        "Class:",
        class_name
    )

    print(
        "Plant:",
        plant
    )

    print(
        "Disease:",
        disease
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    # ========================================================
    # 4. CONFIDENCE
    # ========================================================

    print(
        "\n[4/6] CONFIDENCE HANDLING"
    )

    confidence_level = (
        get_confidence_level(
            confidence
        )
    )

    review_required = (
        needs_review(
            confidence_level
        )
    )

    print(
        "Confidence level:",
        confidence_level
    )

    print(
        "Needs review:",
        review_required
    )

    # ========================================================
    # 5. GRAD-CAM
    # ========================================================

    print(
        "\n[5/6] GRAD-CAM"
    )

    gradcam_path = (
        REPORTS_DIR
        / gradcam_filename
    )

    gradcam_result = (
        save_gradcam(
            image_path,
            gradcam_path,
        )
    )

    explanation = (
        gradcam_result[
            "explanation"
        ]
    )

    print(
        "Available:",
        explanation[
            "available"
        ]
    )

    print(
        "Method:",
        explanation[
            "method"
        ]
    )

    print(
        "Target class:",
        explanation[
            "target_class_name"
        ]
    )

    print(
        "Heatmap:",
        explanation[
            "heatmap_path"
        ]
    )

    # ========================================================
    # 6. DISEASE INFORMATION
    # ========================================================

    print(
        "\n[6/6] DISEASE INFORMATION"
    )

    disease_info_result = (
        get_disease_information(
            class_name
        )
    )

    disease_info = (
        disease_info_result[
            "disease_information"
        ]
    )

    print(
        "Disease:",
        disease_info[
            "disease"
        ]
    )

    print(
        "Status:",
        disease_info[
            "status"
        ]
    )

    print(
        "Severity:",
        disease_info[
            "severity"
        ]
    )

    # ========================================================
    # FINAL AI RESULT
    # ========================================================

    final_result = {

        "success": True,

        "model": {
            "name": "EfficientNet-B0",
            "classes": 38,
        },

        "image": {
            "path": str(
                image_path
            ),

            "width": quality.get(
                "width"
            ),

            "height": quality.get(
                "height"
            ),
        },

        "image_quality": quality,

        "suitability": suitability,

        "prediction": {

            "class_name": class_name,

            "plant": plant,

            "disease": disease,

            "confidence": round(
                confidence,
                2,
            ),

            "confidence_level": (
                confidence_level
            ),

            "needs_review": (
                review_required
            ),
        },

        "explanation": {

            "available": (
                explanation[
                    "available"
                ]
            ),

            "method": (
                explanation[
                    "method"
                ]
            ),

            "target_class": (
                explanation[
                    "target_class_name"
                ]
            ),

            "heatmap_path": (
                explanation[
                    "heatmap_path"
                ]
            ),
        },

        "disease_information": {

            "status": (
                disease_info[
                    "status"
                ]
            ),

            "severity": (
                disease_info[
                    "severity"
                ]
            ),

            "description": (
                disease_info[
                    "description"
                ]
            ),

            "symptoms": (
                disease_info[
                    "symptoms"
                ]
            ),

            "general_management": (
                disease_info[
                    "general_management"
                ]
            ),

            "prevention": (
                disease_info[
                    "prevention"
                ]
            ),
        },
    }

    return final_result


# ============================================================
# VALIDATE FINAL CONTRACT
# ============================================================

def validate_final_result(
    result: dict,
):

    assert result[
        "success"
    ] is True

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    assert (
        result["model"]["name"]
        == "EfficientNet-B0"
    )

    assert (
        result["model"]["classes"]
        == 38
    )

    # --------------------------------------------------------
    # Required sections
    # --------------------------------------------------------

    required_sections = [
        "image_quality",
        "suitability",
        "prediction",
        "explanation",
        "disease_information",
    ]

    for section in required_sections:

        assert section in result, (
            f"Missing final contract section: "
            f"{section}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = (
        result["prediction"]
    )

    assert (
        isinstance(
            prediction["class_name"],
            str,
        )
    )

    assert (
        isinstance(
            prediction["plant"],
            str,
        )
    )

    assert (
        isinstance(
            prediction["disease"],
            str,
        )
    )

    assert (
        0.0
        <= prediction[
            "confidence"
        ]
        <= 100.0
    )

    assert (
        prediction[
            "confidence_level"
        ]
        in [
            "high",
            "moderate",
            "low",
        ]
    )

    assert isinstance(
        prediction[
            "needs_review"
        ],
        bool,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = (
        result[
            "explanation"
        ]
    )

    assert (
        explanation[
            "available"
        ]
        is True
    )

    assert (
        explanation[
            "method"
        ]
        == "Grad-CAM"
    )

    heatmap_path = Path(
        explanation[
            "heatmap_path"
        ]
    )

    assert heatmap_path.exists(), (
        "Grad-CAM heatmap does not exist."
    )

    # --------------------------------------------------------
    # Disease information
    # --------------------------------------------------------

    disease_info = (
        result[
            "disease_information"
        ]
    )

    for field in [
        "status",
        "severity",
        "description",
        "symptoms",
        "general_management",
        "prevention",
    ]:

        assert field in disease_info, (
            f"Disease information missing: "
            f"{field}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "FINAL AI PIPELINE INTEGRATION TEST"
    )

    print("=" * 60)

    # ========================================================
    # APPLE SCAB
    # ========================================================

    apple_result = run_pipeline(
        APPLE_SCAB_IMAGE,
        "final_pipeline_apple_scab_gradcam.jpg",
    )

    validate_final_result(
        apple_result
    )

    # --------------------------------------------------------
    # Apple Scab sanity checks
    # --------------------------------------------------------

    assert (
        apple_result[
            "prediction"
        ][
            "class_name"
        ]
        == "Apple___Apple_scab"
    )

    assert (
        apple_result[
            "prediction"
        ][
            "plant"
        ]
        == "Apple"
    )

    assert (
        apple_result[
            "prediction"
        ][
            "disease"
        ]
        == "Apple scab"
    )

    assert (
        apple_result[
            "prediction"
        ][
            "confidence_level"
        ]
        == "high"
    )

    assert (
        apple_result[
            "prediction"
        ][
            "needs_review"
        ]
        is False
    )

    # ========================================================
    # REAL LEAF
    # ========================================================

    leaf_result = run_pipeline(
        REAL_LEAF_IMAGE,
        "final_pipeline_real_leaf_gradcam.jpg",
    )

    validate_final_result(
        leaf_result
    )

    # Your previously tested real leaf result was:
    # Tomato - Early blight - around 69%
    #
    # Therefore moderate confidence is expected.

    assert (
        leaf_result[
            "prediction"
        ][
            "confidence_level"
        ]
        == "moderate"
    )

    assert (
        leaf_result[
            "prediction"
        ][
            "needs_review"
        ]
        is True
    )

    # ========================================================
    # SAVE CONTRACT EXAMPLES
    # ========================================================

    apple_json = (
        REPORTS_DIR
        / "final_ai_contract_apple_scab.json"
    )

    leaf_json = (
        REPORTS_DIR
        / "final_ai_contract_real_leaf.json"
    )

    with open(
        apple_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            apple_result,
            file,
            indent=2,
        )

    with open(
        leaf_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            leaf_result,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print("=" * 60)

    print(
        "FINAL AI PIPELINE TEST PASSED"
    )

    print("=" * 60)

    print()

    print(
        "AI MODULE STATUS:"
    )

    print(
        "  Image Quality       ✓"
    )

    print(
        "  Leaf Suitability    ✓"
    )

    print(
        "  Prediction          ✓"
    )

    print(
        "  Confidence          ✓"
    )

    print(
        "  Grad-CAM            ✓"
    )

    print(
        "  Disease Information ✓"
    )

    print()

    print(
        "Contract examples:"
    )

    print(
        f"  {apple_json}"
    )

    print(
        f"  {leaf_json}"
    )

    print()

    print(
        "AI MODULE READY FOR CONTRACT FREEZE"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    