from pathlib import Path

from predictor import PlantDiseasePredictor


def main():

    print("=" * 70)
    print("AI → BACKEND CONTRACT TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Test image
    # --------------------------------------------------------

    image_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "PlantVillage"
        / "Apple___Apple_scab"
        / "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    )

    print(f"\nImage:")
    print(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Test image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load predictor
    # --------------------------------------------------------

    predictor = PlantDiseasePredictor()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result = predictor.predict(
        image_path
    )

    # --------------------------------------------------------
    # Verify response structure
    # --------------------------------------------------------

    assert result["success"] is True

    assert "model" in result

    assert "prediction" in result

    assert "top_5" in result

    prediction = result["prediction"]

    assert "class_name" in prediction

    assert "plant" in prediction

    assert "disease" in prediction

    assert "confidence" in prediction

    assert "confidence_level" in prediction

    assert "needs_review" in prediction

    # --------------------------------------------------------
    # Verify known prediction
    # --------------------------------------------------------

    assert (
        prediction["class_name"]
        == "Apple___Apple_scab"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTRACT TEST PASSED")
    print("=" * 70)

    print(
        f"\nPlant: "
        f"{prediction['plant']}"
    )

    print(
        f"Disease: "
        f"{prediction['disease']}"
    )

    print(
        f"Confidence: "
        f"{prediction['confidence']}%"
    )

    print(
        f"Confidence level: "
        f"{prediction['confidence_level']}"
    )

    print(
        f"Needs review: "
        f"{prediction['needs_review']}"
    )

    print(
        "\n✓ AI output structure is ready for backend integration."
    )


if __name__ == "__main__":
    main()