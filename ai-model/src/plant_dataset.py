from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_to_index,
        transform=None
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.class_to_index = (
            class_to_index
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        row = self.dataframe.iloc[
            index
        ]

        image_path = Path(
            row["path"]
        )

        class_name = row[
            "class_name"
        ]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = self.class_to_index[
            class_name
        ]

        return (
            image,
            label
        )