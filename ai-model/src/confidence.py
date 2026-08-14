"""
Confidence / Uncertainty Handling
==================================

Application-level confidence interpretation for the
Plant Disease Detection system.

IMPORTANT:
    These thresholds are heuristics.
    They are NOT calibrated probabilities.

Rules:
    confidence >= 80.0
        -> high
        -> needs_review = False

    50.0 <= confidence < 80.0
        -> moderate
        -> needs_review = True

    confidence < 50.0
        -> low
        -> needs_review = True
"""


from typing import Dict, Any


# ============================================================
# THRESHOLDS
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 80.0
MODERATE_CONFIDENCE_THRESHOLD = 50.0


# ============================================================
# VALIDATION
# ============================================================

def validate_confidence(confidence: float) -> float:
    """
    Validate and normalize confidence.

    Input:
        confidence in percentage form.

    Example:
        99.76

    Returns:
        float rounded to two decimal places.

    Raises:
        TypeError
        ValueError
    """

    if not isinstance(confidence, (int, float)):
        raise TypeError(
            "Confidence must be an int or float."
        )

    confidence = float(confidence)

    if confidence < 0.0 or confidence > 100.0:
        raise ValueError(
            "Confidence must be between 0 and 100."
        )

    return round(confidence, 2)


# ============================================================
# CLASSIFICATION
# ============================================================

def get_confidence_level(confidence: float) -> str:
    """
    Convert confidence percentage into a confidence level.

    Returns:
        "high"
        "moderate"
        "low"
    """

    confidence = validate_confidence(confidence)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"

    if confidence >= MODERATE_CONFIDENCE_THRESHOLD:
        return "moderate"

    return "low"


# ============================================================
# REVIEW DECISION
# ============================================================

def needs_review(confidence: float) -> bool:
    """
    Determine whether prediction should be reviewed.

    High confidence:
        False

    Moderate confidence:
        True

    Low confidence:
        True
    """

    level = get_confidence_level(confidence)

    return level != "high"


# ============================================================
# COMPLETE RESULT
# ============================================================

def analyze_confidence(confidence: float) -> Dict[str, Any]:
    """
    Return complete structured confidence information.

    Example:
        {
            "confidence": 69.19,
            "confidence_level": "moderate",
            "needs_review": True
        }
    """

    confidence = validate_confidence(confidence)

    level = get_confidence_level(confidence)

    review = needs_review(confidence)

    return {
        "confidence": confidence,
        "confidence_level": level,
        "needs_review": review,
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    test_values = [
        99.76,
        80.00,
        79.99,
        69.19,
        50.00,
        49.99,
        10.00,
    ]

    print("=" * 60)
    print("CONFIDENCE / UNCERTAINTY TEST")
    print("=" * 60)

    for value in test_values:

        result = analyze_confidence(value)

        print(
            f"{result['confidence']:6.2f}%"
            f" -> "
            f"{result['confidence_level']:8}"
            f" -> "
            f"needs_review={result['needs_review']}"
        )

    print("=" * 60)