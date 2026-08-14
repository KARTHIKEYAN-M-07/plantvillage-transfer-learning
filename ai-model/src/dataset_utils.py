import json
from pathlib import Path


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_class_directories(dataset_dir):

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    directories = sorted(
        [
            path
            for path in dataset_dir.iterdir()
            if path.is_dir()
        ]
    )

    if not directories:
        raise RuntimeError(
            f"No class folders found in {dataset_dir}"
        )

    return directories


def get_image_files(class_dir):

    class_dir = Path(class_dir)

    return sorted(
        [
            path
            for path in class_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in VALID_EXTENSIONS
            )
        ]
    )


def get_all_records(dataset_dir):

    records = []

    class_dirs = get_class_directories(
        dataset_dir
    )

    for class_dir in class_dirs:

        class_name = class_dir.name

        image_files = get_image_files(
            class_dir
        )

        for image_path in image_files:

            records.append(
                {
                    "path": str(image_path),
                    "class_name": class_name
                }
            )

    return records


def save_class_names(
    class_names,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_class_names(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)