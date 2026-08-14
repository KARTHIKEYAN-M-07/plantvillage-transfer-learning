from collections import Counter

from config import (
    DATASET_DIR,
    REPORT_DIR,
    CLASS_NAMES_PATH
)

from dataset_utils import (
    get_class_directories,
    get_image_files,
    save_class_names
)


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def main():

    print("=" * 70)
    print("PLANTVILLAGE DATASET CHECK")
    print("=" * 70)

    # ========================================================
    # 1. Check dataset directory
    # ========================================================

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"""
Dataset not found.

Expected:

{DATASET_DIR}

Expected structure:

data/
└── PlantVillage/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    └── ...
"""
        )

    # ========================================================
    # 2. Find class folders
    # ========================================================

    class_dirs = get_class_directories(
        DATASET_DIR
    )

    class_names = [
        directory.name
        for directory in class_dirs
    ]

    print(
        f"\nNumber of classes: "
        f"{len(class_names)}"
    )

    # Save class names
    save_class_names(
        class_names,
        CLASS_NAMES_PATH
    )

    # ========================================================
    # 3. Count images
    # ========================================================

    total_images = 0

    class_counts = {}

    extension_counts = Counter()

    print(
        "\nCounting images..."
    )

    print(
        "-" * 70
    )

    for index, class_dir in enumerate(
        class_dirs,
        start=1
    ):

        image_files = get_image_files(
            class_dir
        )

        count = len(
            image_files
        )

        class_counts[
            class_dir.name
        ] = count

        total_images += count

        # Count extensions
        for image_path in image_files:

            extension_counts[
                image_path.suffix.lower()
            ] += 1

        print(
            f"[{index:02d}/{len(class_dirs)}] "
            f"{class_dir.name}: "
            f"{count} images"
        )

    # ========================================================
    # 4. Dataset summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        f"Total classes: "
        f"{len(class_names)}"
    )

    print(
        f"Total images: "
        f"{total_images}"
    )

    # ========================================================
    # 5. File extensions
    # ========================================================

    print(
        "\nImage file formats:"
    )

    for extension, count in (
        extension_counts.most_common()
    ):

        print(
            f"{extension}: {count}"
        )

    # ========================================================
    # 6. Class balance
    # ========================================================

    counts = list(
        class_counts.values()
    )

    if counts:

        minimum = min(
            counts
        )

        maximum = max(
            counts
        )

        print(
            "\nClass balance:"
        )

        print(
            f"Smallest class: "
            f"{minimum}"
        )

        print(
            f"Largest class: "
            f"{maximum}"
        )

        if minimum > 0:

            imbalance_ratio = (
                maximum / minimum
            )

            print(
                f"Imbalance ratio: "
                f"{imbalance_ratio:.2f}"
            )

    # ========================================================
    # 7. Save report
    # ========================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        REPORT_DIR /
        "dataset_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PLANTVILLAGE DATASET SUMMARY\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Number of classes: "
            f"{len(class_names)}\n"
        )

        file.write(
            f"Total images: "
            f"{total_images}\n\n"
        )

        file.write(
            "IMAGE FORMATS\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for extension, count in (
            extension_counts.most_common()
        ):

            file.write(
                f"{extension}: "
                f"{count}\n"
            )

        file.write(
            "\nIMAGES PER CLASS\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for name in class_names:

            file.write(
                f"{name}: "
                f"{class_counts[name]}\n"
            )

        file.write(
            "\nCLASS BALANCE\n"
        )

        file.write(
            f"Smallest class: "
            f"{minimum}\n"
        )

        file.write(
            f"Largest class: "
            f"{maximum}\n"
        )

        if minimum > 0:

            file.write(
                f"Imbalance ratio: "
                f"{maximum / minimum:.2f}\n"
            )

    # ========================================================
    # 8. Final output
    # ========================================================

    print(
        f"\nSaved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {CLASS_NAMES_PATH}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET CHECK COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()