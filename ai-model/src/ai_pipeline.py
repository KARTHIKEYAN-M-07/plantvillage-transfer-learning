"""
AI PIPELINE
===========

Single public entry point for the complete Plant Disease
Detection AI system.

Pipeline:

    Image
      |
      v
    Image Quality
      |
      v
    Leaf / Image Suitability
      |
      v
    EfficientNet-B0 Prediction
      |
      v
    Confidence / Uncertainty
      |
      v
    Grad-CAM Explainability
      |
      v
    Disease Information
      |
      v
    Final AI Contract

IMPORTANT
---------
- Does NOT retrain the model.
- Does NOT modify model weights.
- Does NOT create a backend API.
- Does NOT depend on FastAPI.
- Backend integration will happen later.

The backend should eventually call this module rather than
calling all individual AI components separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


# ============================================================
# IMPORT AI MODULES
# ============================================================

from predictor import PlantDiseasePredictor
from image_quality import validate_image_quality
from image_suitability import validate_image_suitability
from gradcam import save_gradcam
from disease_information import get_disease_information


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

REPORTS_DIR = (
    PROJECT_ROOT / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# AI PIPELINE
# ============================================================

class PlantDiseaseAIPipeline:
    """
    Complete AI-side Plant Disease Detection pipeline.

    This class combines all P1 modules into one interface.
    """

    def __init__(
        self,
        predictor: PlantDiseasePredictor | None = None,
    ):
        """
        Initialize the AI pipeline.

        The predictor can optionally be injected for testing.
        """

        print(
            "=" * 60
        )

        print(
            "INITIALIZING PLANT DISEASE AI PIPELINE"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # EfficientNet predictor
        # ----------------------------------------------------

        if predictor is None:

            self.predictor = (
                PlantDiseasePredictor()
            )

        else:

            self.predictor = predictor

        print(
            "✓ AI pipeline ready."
        )

        print(
            "=" * 60
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def get_confidence_level(
        confidence: float,
    ) -> str:
        """
        Application-level confidence heuristic.

        >= 80.00 -> high
        50.00-79.99 -> moderate
        < 50.00 -> low

        IMPORTANT:
        These thresholds are application heuristics.
        They are NOT calibrated probabilities.
        """

        if confidence >= 80.0:

            return "high"

        if confidence >= 50.0:

            return "moderate"

        return "low"


    # ========================================================
    # REVIEW
    # ========================================================

    @staticmethod
    def needs_review(
        confidence_level: str,
    ) -> bool:
        """
        High-confidence predictions do not require review.

        Moderate and low confidence predictions require review.
        """

        return (
            confidence_level
            != "high"
        )


    # ========================================================
    # NORMALIZE QUALITY
    # ========================================================

    @staticmethod
    def _normalize_quality(
        result: dict,
    ) -> dict:

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Image quality module returned "
                "an invalid result."
            )

        if "image_quality" in result:

            return result[
                "image_quality"
            ]

        return result


    # ========================================================
    # NORMALIZE SUITABILITY
    # ========================================================

    @staticmethod
    def _normalize_suitability(
        result: dict,
    ) -> dict:

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Image suitability module returned "
                "an invalid result."
            )

        if "suitability" in result:

            return result[
                "suitability"
            ]

        return result


    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    def validate_quality(
        self,
        image_path: str | Path,
    ) -> dict:

        result = (
            validate_image_quality(
                image_path
            )
        )

        return (
            self._normalize_quality(
                result
            )
        )


    # ========================================================
    # IMAGE SUITABILITY
    # ========================================================

    def validate_suitability(
        self,
        image_path: str | Path,
    ) -> dict:

        result = (
            validate_image_suitability(
                image_path
            )
        )

        return (
            self._normalize_suitability(
                result
            )
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_disease(
        self,
        image_path: str | Path,
    ) -> dict:

        result = (
            self.predictor.predict(
                str(image_path)
            )
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Predictor returned "
                "an invalid result."
            )

        if not result.get(
            "success",
            False,
        ):

            raise RuntimeError(
                "Plant disease predictor failed:\n"
                + json.dumps(
                    result,
                    indent=2,
                )
            )

        return result


    # ========================================================
    # GRAD-CAM
    # ========================================================

    def generate_explanation(
        self,
        image_path: str | Path,
        class_name: str,
        output_filename: str | None = None,
    ) -> dict:
        """
        Generate Grad-CAM explanation.

        The Grad-CAM implementation determines the target
        class from the prediction.
        """

        image_path = Path(
            image_path
        )

        # ----------------------------------------------------
        # Default filename
        # ----------------------------------------------------

        if output_filename is None:

            safe_name = (
                class_name
                .replace(
                    "/",
                    "_",
                )
                .replace(
                    "\\",
                    "_",
                )
                .replace(
                    " ",
                    "_",
                )
            )

            output_filename = (
                f"gradcam_{safe_name}.jpg"
            )

        output_path = (
            REPORTS_DIR
            / output_filename
        )

        result = (
            save_gradcam(
                image_path,
                output_path,
            )
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Grad-CAM returned "
                "an invalid result."
            )

        if "explanation" not in result:

            raise RuntimeError(
                "Grad-CAM result does not contain "
                "'explanation'."
            )

        explanation = (
            result[
                "explanation"
            ]
        )

        return explanation


    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    def get_disease_information(
        self,
        class_name: str,
    ) -> dict:

        result = (
            get_disease_information(
                class_name
            )
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Disease information module returned "
                "an invalid result."
            )

        if "disease_information" not in result:

            raise RuntimeError(
                "Disease information result does not "
                "contain 'disease_information'."
            )

        return result[
            "disease_information"
        ]


    # ========================================================
    # COMPLETE PREDICTION
    # ========================================================

    def predict(
        self,
        image_path: str | Path,
        generate_gradcam: bool = True,
        gradcam_filename: str | None = None,
    ) -> dict:
        """
        Run the complete AI pipeline.

        Parameters
        ----------
        image_path:
            Path to the uploaded image.

        generate_gradcam:
            Whether Grad-CAM should be generated.

        gradcam_filename:
            Optional output filename for Grad-CAM.

        Returns
        -------
        dict
            Stable AI contract.
        """

        # ====================================================
        # IMAGE PATH
        # ====================================================

        image_path = Path(
            image_path
        )

        # ====================================================
        # BASIC FILE VALIDATION
        # ====================================================

        if not image_path.exists():

            return {
                "success": False,

                "error": {
                    "stage": "input",
                    "message": (
                        f"Image does not exist: "
                        f"{image_path}"
                    ),
                },
            }

        # ====================================================
        # 1. IMAGE QUALITY
        # ====================================================

        quality = (
            self.validate_quality(
                image_path
            )
        )

        # ----------------------------------------------------
        # If image quality is invalid, stop.
        # ----------------------------------------------------

        if not quality.get(
            "valid",
            False,
        ):

            return {

                "success": False,

                "stage": "image_quality",

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

                "message": (
                    "Image failed quality validation."
                ),
            }

        # ====================================================
        # 2. LEAF / IMAGE SUITABILITY
        # ====================================================

        suitability = (
            self.validate_suitability(
                image_path
            )
        )

        # ----------------------------------------------------
        # If image is unsuitable, stop.
        #
        # "review" is allowed to continue.
        # ----------------------------------------------------

        if suitability.get(
            "status"
        ) == "unsuitable":

            return {

                "success": False,

                "stage": "image_suitability",

                "image": {
                    "path": str(
                        image_path
                    ),
                    "width": suitability.get(
                        "width"
                    ),
                    "height": suitability.get(
                        "height"
                    ),
                },

                "image_quality": quality,

                "suitability": suitability,

                "message": (
                    "Image was determined to be "
                    "unsuitable for plant disease prediction."
                ),
            }

        # ====================================================
        # 3. MODEL PREDICTION
        # ====================================================

        prediction_result = (
            self.predict_disease(
                image_path
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

        # ====================================================
        # 4. CONFIDENCE
        # ====================================================

        confidence_level = (
            self.get_confidence_level(
                confidence
            )
        )

        review_required = (
            self.needs_review(
                confidence_level
            )
        )

        # ====================================================
        # 5. GRAD-CAM
        # ====================================================

        if generate_gradcam:

            explanation = (
                self.generate_explanation(
                    image_path=image_path,
                    class_name=class_name,
                    output_filename=gradcam_filename,
                )
            )

        else:

            explanation = {

                "available": False,

                "method": "Grad-CAM",

                "target_class": class_name,

                "heatmap_path": None,
            }

        # ====================================================
        # 6. DISEASE INFORMATION
        # ====================================================

        disease_information = (
            self.get_disease_information(
                class_name
            )
        )

        # ====================================================
        # FINAL AI CONTRACT
        # ====================================================

        result = {

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

                "available": explanation.get(
                    "available",
                    False,
                ),

                "method": explanation.get(
                    "method",
                    "Grad-CAM",
                ),

                "target_class": explanation.get(
                    "target_class_name",
                    explanation.get(
                        "target_class",
                        class_name,
                    ),
                ),

                "heatmap_path": explanation.get(
                    "heatmap_path"
                ),
            },

            "disease_information": {

                "status": disease_information.get(
                    "status"
                ),

                "severity": disease_information.get(
                    "severity"
                ),

                "description": disease_information.get(
                    "description"
                ),

                "symptoms": disease_information.get(
                    "symptoms",
                    [],
                ),

                "general_management": (
                    disease_information.get(
                        "general_management",
                        [],
                    )
                ),

                "prevention": disease_information.get(
                    "prevention",
                    [],
                ),
            },
        }

        return result


# ============================================================
# SIMPLE FUNCTION INTERFACE
# ============================================================

_pipeline: PlantDiseaseAIPipeline | None = None


def get_pipeline() -> PlantDiseaseAIPipeline:
    """
    Return the shared AI pipeline instance.

    This prevents the EfficientNet model from being loaded
    repeatedly when the same Python process handles multiple
    predictions.
    """

    global _pipeline

    if _pipeline is None:

        _pipeline = (
            PlantDiseaseAIPipeline()
        )

    return _pipeline


def predict(
    image_path: str | Path,
    generate_gradcam: bool = True,
    gradcam_filename: str | None = None,
) -> dict:
    """
    Simple public function for the backend.

    Example:

        from ai_pipeline import predict

        result = predict(
            "test_images/leaf.jpg"
        )
    """

    pipeline = (
        get_pipeline()
    )

    return pipeline.predict(
        image_path=image_path,
        generate_gradcam=generate_gradcam,
        gradcam_filename=gradcam_filename,
    )


# ============================================================
# SAVE CONTRACT
# ============================================================

def save_result(
    result: dict,
    output_path: str | Path,
) -> Path:
    """
    Save an AI result as JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plant Disease Detection "
            "AI Pipeline"
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to input image"
        ),
    )

    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help=(
            "Skip Grad-CAM generation"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output path"
        ),
    )

    args = parser.parse_args()

    result = predict(
        image_path=args.image,
        generate_gradcam=(
            not args.no_gradcam
        ),
    )

    print()

    print("=" * 60)

    print(
        "AI PIPELINE RESULT"
    )

    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Save JSON if requested
    # --------------------------------------------------------

    if args.output:

        saved_path = save_result(
            result,
            args.output,
        )

        print()

        print(
            f"Result saved to: {saved_path}"
        )