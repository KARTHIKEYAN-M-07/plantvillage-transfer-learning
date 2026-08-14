from predictor import PlantDiseasePredictor


def main():

    image_path = r".\test_images\leaf.jpg"

    predictor = (
        PlantDiseasePredictor()
    )

    result = predictor.predict(
        image_path
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    prediction = result["prediction"]

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
        f"{prediction['confidence']:.2f}%"
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
        "\nTop 5:"
    )

    for index, item in enumerate(
        result["top_5"],
        start=1
    ):

        print(
            f"{index}. "
            f"{item['plant']} - "
            f"{item['disease']} "
            f"({item['confidence']:.2f}%)"
        )


if __name__ == "__main__":

    main()