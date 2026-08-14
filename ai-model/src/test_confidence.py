"""
Tests for confidence / uncertainty handling.
"""


from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.confidence import analyze_confidence


def test_confidence(value, expected_level, expected_review):

    result = analyze_confidence(value)

    assert result["confidence"] == value

    assert (
        result["confidence_level"]
        == expected_level
    )

    assert (
        result["needs_review"]
        == expected_review
    )

    print(
        f"PASS: {value}%"
        f" -> {expected_level}"
        f" -> needs_review={expected_review}"
    )


def main():

    print("=" * 60)
    print("P1.1 CONFIDENCE TEST")
    print("=" * 60)

    test_confidence(
        99.76,
        "high",
        False,
    )

    test_confidence(
        80.00,
        "high",
        False,
    )

    test_confidence(
        79.99,
        "moderate",
        True,
    )

    test_confidence(
        69.19,
        "moderate",
        True,
    )

    test_confidence(
        50.00,
        "moderate",
        True,
    )

    test_confidence(
        49.99,
        "low",
        True,
    )

    test_confidence(
        10.00,
        "low",
        True,
    )

    print()
    print("=" * 60)
    print("P1.1 CONFIDENCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()