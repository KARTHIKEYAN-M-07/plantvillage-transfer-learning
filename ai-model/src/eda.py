import random
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

from config import (
    DATASET_DIR,
    REPORT_DIR,
    RANDOM_SEED
)

from dataset_utils import (
    get_all_records
)


random.seed(
    RANDOM_SEED
)


def class_distribution(records):

    counts = Counter(
        record["class_name"]
        for record in records
    )

    names = list(counts.keys())

    values = [
        counts[name]
        for name in names
    ]

    plt.figure(
        figsize=(18, 8)
    )

    plt.bar(
        names,
        values
    )

    plt.xticks(
        rotation=90,
        fontsize=8
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of images"
    )

    plt.title(
        "PlantVillage Class Distribution"
    )

    plt.tight_layout()

    output = (
        REPORT_DIR /
        "class_distribution.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def random_samples(records):

    count = min(
        12,
        len(records)
    )

    samples = random.sample(
        records,
        count
    )

    plt.figure(
        figsize=(12, 9)
    )

    for i, record in enumerate(samples):

        image = Image.open(
            record["path"]
        ).convert("RGB")

        plt.subplot(
            3,
            4,
            i + 1
        )

        plt.imshow(image)

        plt.title(
            record["class_name"],
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    output = (
        REPORT_DIR /
        "random_samples.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def main():

    records = get_all_records(
        DATASET_DIR
    )

    if not records:

        raise RuntimeError(
            "No images found."
        )

    print(
        f"Images: {len(records)}"
    )

    print(
        "Classes:",
        len(
            set(
                record["class_name"]
                for record in records
            )
        )
    )

    class_distribution(
        records
    )

    random_samples(
        records
    )


if __name__ == "__main__":
    main()