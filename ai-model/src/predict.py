import json
from pathlib import Path

import torch
from PIL import Image

from config import (
    DEVICE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH
)

from preprocess import (
    get_eval_transform
)

from model import (
    create_model
)


class PlantDiseasePredictor:

    def __init__(self):

        # ----------------------------------------------------
        # Load classes
        # ----------------------------------------------------

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.class_names = json.load(
                file
            )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = create_model(
            num_classes=len(
                self.class_names
            ),
            pretrained=False
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            DEVICE
        )

        self.model.eval()

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        self.transform = (
            get_eval_transform()
        )

    def predict(
        self,
        image
    ):

        # ----------------------------------------------------
        # Accept path or PIL image
        # ----------------------------------------------------

        if isinstance(
            image,
            (str, Path)
        ):

            image = Image.open(
                image
            )

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
            )

            confidence, index = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        class_index = int(
            index.item()
        )

        confidence_value = float(
            confidence.item()
        )

        disease = self.class_names[
            class_index
        ]

        return {
            "success": True,
            "disease": disease,
            "confidence": confidence_value
        }


def main():

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src\\predict.py image.jpg"
        )

        return

    image_path = sys.argv[1]

    predictor = (
        PlantDiseasePredictor()
    )

    result = predictor.predict(
        image_path
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )


if __name__ == "__main__":

    main()