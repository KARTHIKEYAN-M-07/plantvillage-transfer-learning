"""
P1.5 - Disease Information Tests
=================================

Tests:

1. All 38 PlantVillage classes exist.
2. Apple Scab information works.
3. Tomato Early Blight information works.
4. Healthy class works.
5. Unknown class is rejected.
6. Database structure is valid.
"""


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
# IMPORT
# ============================================================

from disease_information import (
    DISEASE_INFORMATION,
    EXPECTED_CLASSES,
    get_disease_information,
    is_healthy_class,
    validate_database,
)


# ============================================================
# TEST DATABASE
# ============================================================

def test_database():

    print()
    print("-" * 60)
    print("DATABASE VALIDATION")
    print("-" * 60)

    validate_database()

    assert len(
        DISEASE_INFORMATION
    ) == 38

    assert len(
        EXPECTED_CLASSES
    ) == 38

    for class_name in EXPECTED_CLASSES:

        assert class_name in (
            DISEASE_INFORMATION
        )

    print(
        "Classes expected: 38"
    )

    print(
        "Classes available:",
        len(DISEASE_INFORMATION)
    )

    print(
        "Database valid: True"
    )


# ============================================================
# TEST APPLE SCAB
# ============================================================

def test_apple_scab():

    print()
    print("-" * 60)
    print("APPLE SCAB")
    print("-" * 60)

    result = (
        get_disease_information(
            "Apple___Apple_scab"
        )
    )

    information = (
        result[
            "disease_information"
        ]
    )

    print(
        "Plant:",
        information["plant"]
    )

    print(
        "Disease:",
        information["disease"]
    )

    print(
        "Status:",
        information["status"]
    )

    print(
        "Severity:",
        information["severity"]
    )

    print(
        "Symptoms:",
        len(
            information["symptoms"]
        )
    )

    assert (
        information["plant"]
        == "Apple"
    )

    assert (
        information["disease"]
        == "Apple scab"
    )

    assert (
        information["status"]
        == "diseased"
    )

    assert len(
        information["symptoms"]
    ) > 0

    assert len(
        information["general_management"]
    ) > 0

    assert len(
        information["prevention"]
    ) > 0


# ============================================================
# TEST TOMATO EARLY BLIGHT
# ============================================================

def test_tomato_early_blight():

    print()
    print("-" * 60)
    print("TOMATO EARLY BLIGHT")
    print("-" * 60)

    result = (
        get_disease_information(
            "Tomato___Early_blight"
        )
    )

    information = (
        result[
            "disease_information"
        ]
    )

    print(
        "Plant:",
        information["plant"]
    )

    print(
        "Disease:",
        information["disease"]
    )

    print(
        "Status:",
        information["status"]
    )

    assert (
        information["plant"]
        == "Tomato"
    )

    assert (
        information["disease"]
        == "Early blight"
    )

    assert (
        information["status"]
        == "diseased"
    )


# ============================================================
# TEST HEALTHY
# ============================================================

def test_healthy():

    print()
    print("-" * 60)
    print("HEALTHY CLASS")
    print("-" * 60)

    result = (
        get_disease_information(
            "Apple___healthy"
        )
    )

    information = (
        result[
            "disease_information"
        ]
    )

    print(
        "Plant:",
        information["plant"]
    )

    print(
        "Disease:",
        information["disease"]
    )

    print(
        "Status:",
        information["status"]
    )

    assert (
        information["status"]
        == "healthy"
    )

    assert is_healthy_class(
        "Apple___healthy"
    ) is True

    assert is_healthy_class(
        "Apple___Apple_scab"
    ) is False


# ============================================================
# TEST REQUIRED FIELDS
# ============================================================

def test_required_fields():

    print()
    print("-" * 60)
    print("REQUIRED FIELD VALIDATION")
    print("-" * 60)

    required_fields = {
        "plant",
        "disease",
        "status",
        "description",
        "symptoms",
        "general_management",
        "prevention",
        "severity",
    }

    for class_name in EXPECTED_CLASSES:

        result = (
            get_disease_information(
                class_name
            )
        )

        information = (
            result[
                "disease_information"
            ]
        )

        for field in required_fields:

            assert field in information, (
                f"{class_name} missing "
                f"field: {field}"
            )

    print(
        "Required fields valid: True"
    )


# ============================================================
# TEST UNKNOWN CLASS
# ============================================================

def test_unknown_class():

    print()
    print("-" * 60)
    print("UNKNOWN CLASS")
    print("-" * 60)

    try:

        get_disease_information(
            "Unknown___Disease"
        )

        raise AssertionError(
            "Unknown class should raise ValueError."
        )

    except ValueError as exc:

        print(
            "Correctly rejected:"
        )

        print(
            exc
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "P1.5 DISEASE INFORMATION TEST"
    )

    print("=" * 60)

    test_database()

    test_apple_scab()

    test_tomato_early_blight()

    test_healthy()

    test_required_fields()

    test_unknown_class()

    print()

    print("=" * 60)

    print(
        "P1.5 DISEASE INFORMATION TESTS PASSED"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()