from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance

from predict import PlantDiseasePredictor


def main():

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src\\robustness.py image.jpg"
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    original = Image.open(
        image_path
    ).convert("RGB")

    predictor = (
        PlantDiseasePredictor()
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    print("\nORIGINAL")

    print(
        predictor.predict(
            original
        )
    )

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blurred = original.filter(
        ImageFilter.GaussianBlur(
            radius=3
        )
    )

    print("\nBLURRED")

    print(
        predictor.predict(
            blurred
        )
    )

    # --------------------------------------------------------
    # Darker
    # --------------------------------------------------------

    darker = ImageEnhance.Brightness(
        original
    ).enhance(
        0.5
    )

    print("\nDARKER")

    print(
        predictor.predict(
            darker
        )
    )

    # --------------------------------------------------------
    # Brighter
    # --------------------------------------------------------

    brighter = ImageEnhance.Brightness(
        original
    ).enhance(
        1.5
    )

    print("\nBRIGHTER")

    print(
        predictor.predict(
            brighter
        )
    )


if __name__ == "__main__":

    main()