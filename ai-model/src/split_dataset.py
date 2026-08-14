from pathlib import Path
import random

import pandas as pd

from config import (
    DATASET_DIR,
    REPORT_DIR,
    RANDOM_SEED,
    VALIDATION_RATIO
)

from dataset_utils import (
    get_class_directories,
    get_image_files
)


# ============================================================
# OFFICIAL PLANTVILLAGE SPLIT FILES
# ============================================================

TRAIN_FILE = (
    REPORT_DIR
    / "split_cache"
    / "color_train.txt"
)

TEST_FILE = (
    REPORT_DIR
    / "split_cache"
    / "color_test.txt"
)


# ============================================================
# READ OFFICIAL SPLIT
# ============================================================

def read_split_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"""
Split file not found:

{path}

Run the metadata download step first.
"""
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return lines


# ============================================================
# BUILD LOCAL IMAGE INDEX
# ============================================================

def build_local_index():

    print(
        "\nIndexing local PlantVillage images..."
    )

    print(
        "-" * 70
    )

    local_index = {}

    total_images = 0

    class_dirs = get_class_directories(
        DATASET_DIR
    )

    for class_number, class_dir in enumerate(
        class_dirs,
        start=1
    ):

        class_name = class_dir.name

        image_files = get_image_files(
            class_dir
        )

        for image_path in image_files:

            # Official split paths use:
            #
            # class_name/image_name.jpg
            #
            # Therefore class + filename is enough.

            key = (
                class_name.lower(),
                image_path.name.lower()
            )

            local_index[key] = image_path

            total_images += 1

        print(
            f"[{class_number:02d}/{len(class_dirs)}] "
            f"{class_name}: "
            f"{len(image_files)} images"
        )

    print(
        "-" * 70
    )

    print(
        f"Local images indexed: "
        f"{total_images:,}"
    )

    return local_index


# ============================================================
# MATCH OFFICIAL PATH
# ============================================================

def find_local_image(
    official_path,
    local_index
):

    normalized_path = (
        official_path
        .replace("\\", "/")
        .strip()
    )

    parts = normalized_path.split(
        "/"
    )

    if len(parts) < 2:

        return None

    class_name = parts[-2]

    filename = parts[-1]

    key = (
        class_name.lower(),
        filename.lower()
    )

    return local_index.get(
        key
    )


# ============================================================
# MATCH SPLIT
# ============================================================

def match_split(
    split_lines,
    split_name,
    local_index
):

    records = []

    missing = []

    total = len(
        split_lines
    )

    print(
        f"\nMatching {split_name} images..."
    )

    print(
        "-" * 70
    )

    for index, official_path in enumerate(
        split_lines,
        start=1
    ):

        local_path = find_local_image(
            official_path,
            local_index
        )

        if local_path is None:

            missing.append(
                official_path
            )

        else:

            parts = (
                official_path
                .replace("\\", "/")
                .split("/")
            )

            class_name = parts[-2]

            records.append(
                {
                    "path": str(
                        local_path.resolve()
                    ),
                    "class_name": class_name,
                    "official_split": split_name
                }
            )

        # Progress
        if (
            index % 1000 == 0
            or
            index == total
        ):

            percentage = (
                index / total
            ) * 100

            print(
                f"{split_name}: "
                f"{index:,}/{total:,} "
                f"({percentage:.1f}%)"
            )

    print(
        f"Matched: "
        f"{len(records):,}"
    )

    print(
        f"Missing: "
        f"{len(missing):,}"
    )

    return records, missing


# ============================================================
# CREATE STRATIFIED VALIDATION SPLIT
# ============================================================

def create_validation_split(
    train_df
):

    random.seed(
        RANDOM_SEED
    )

    train_records = []

    validation_records = []

    print(
        "\nCreating validation split..."
    )

    print(
        f"Validation ratio: "
        f"{VALIDATION_RATIO:.2f}"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Split separately inside every class.
    #
    # This keeps all 38 classes represented in both
    # training and validation.
    # --------------------------------------------------------

    for class_name, class_df in (
        train_df.groupby(
            "class_name"
        )
    ):

        indices = list(
            class_df.index
        )

        random.shuffle(
            indices
        )

        validation_count = max(
            1,
            int(
                len(indices)
                * VALIDATION_RATIO
            )
        )

        validation_indices = set(
            indices[
                :validation_count
            ]
        )

        for index in indices:

            record = class_df.loc[
                index
            ].to_dict()

            if index in validation_indices:

                record["split"] = (
                    "validation"
                )

                validation_records.append(
                    record
                )

            else:

                record["split"] = (
                    "train"
                )

                train_records.append(
                    record
                )

    train_result = pd.DataFrame(
        train_records
    )

    validation_result = pd.DataFrame(
        validation_records
    )

    return (
        train_result,
        validation_result
    )


# ============================================================
# CHECK DUPLICATES
# ============================================================

def check_duplicate_paths(df):

    duplicates = (
        df["path"]
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate image paths: "
        f"{duplicates}"
    )

    if duplicates > 0:

        duplicate_rows = df[
            df["path"].duplicated(
                keep=False
            )
        ]

        print(
            duplicate_rows.head(20)
        )

        raise RuntimeError(
            "Duplicate image paths detected."
        )


# ============================================================
# CHECK SPLIT SEPARATION
# ============================================================

def check_split_separation(df):

    train_paths = set(
        df.loc[
            df["split"] == "train",
            "path"
        ]
    )

    validation_paths = set(
        df.loc[
            df["split"] == "validation",
            "path"
        ]
    )

    test_paths = set(
        df.loc[
            df["split"] == "test",
            "path"
        ]
    )

    train_validation = (
        train_paths
        &
        validation_paths
    )

    train_test = (
        train_paths
        &
        test_paths
    )

    validation_test = (
        validation_paths
        &
        test_paths
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "IMAGE SPLIT LEAKAGE CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"Train ∩ Validation: "
        f"{len(train_validation)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(validation_test)}"
    )

    if (
        train_validation
        or
        train_test
        or
        validation_test
    ):

        raise RuntimeError(
            """
IMAGE LEAKAGE DETECTED!

The same image exists in multiple splits.
The split will NOT be saved.
"""
        )

    print(
        "\n✓ No image appears in multiple splits."
    )


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(df):

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASS DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    distribution = pd.crosstab(
        df["class_name"],
        df["split"]
    )

    print(
        distribution.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PLANTVILLAGE DATASET SPLIT"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Dataset exists?
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"""
Dataset not found:

{DATASET_DIR}
"""
        )

    # --------------------------------------------------------
    # Read official files
    # --------------------------------------------------------

    train_lines = read_split_file(
        TRAIN_FILE
    )

    test_lines = read_split_file(
        TEST_FILE
    )

    print(
        f"\nOfficial training images: "
        f"{len(train_lines):,}"
    )

    print(
        f"Official test images: "
        f"{len(test_lines):,}"
    )

    # --------------------------------------------------------
    # Build local index
    # --------------------------------------------------------

    local_index = build_local_index()

    # --------------------------------------------------------
    # Match official training images
    # --------------------------------------------------------

    train_records, train_missing = (
        match_split(
            train_lines,
            "training",
            local_index
        )
    )

    # --------------------------------------------------------
    # Match official test images
    # --------------------------------------------------------

    test_records, test_missing = (
        match_split(
            test_lines,
            "test",
            local_index
        )
    )

    # --------------------------------------------------------
    # Check missing images
    # --------------------------------------------------------

    if train_missing:

        print(
            "\nMissing training images:"
        )

        for path in train_missing[:20]:

            print(
                path
            )

    if test_missing:

        print(
            "\nMissing test images:"
        )

        for path in test_missing[:20]:

            print(
                path
            )

    # --------------------------------------------------------
    # We require complete matching.
    # --------------------------------------------------------

    if len(train_records) != len(
        train_lines
    ):

        raise RuntimeError(
            f"""
Training matching failed.

Expected:
{len(train_lines):,}

Matched:
{len(train_records):,}

Missing:
{len(train_missing):,}
"""
        )

    if len(test_records) != len(
        test_lines
    ):

        raise RuntimeError(
            f"""
Test matching failed.

Expected:
{len(test_lines):,}

Matched:
{len(test_records):,}

Missing:
{len(test_missing):,}
"""
        )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    official_train_df = pd.DataFrame(
        train_records
    )

    official_test_df = pd.DataFrame(
        test_records
    )

    # --------------------------------------------------------
    # Create validation ONLY from official training data
    # --------------------------------------------------------

    (
        final_train_df,
        validation_df
    ) = create_validation_split(
        official_train_df
    )

    # Official test remains untouched.
    official_test_df[
        "split"
    ] = "test"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            final_train_df,
            validation_df,
            official_test_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove temporary column
    # --------------------------------------------------------

    final_df = final_df[
        [
            "path",
            "class_name",
            "split",
            "official_split"
        ]
    ]

    # --------------------------------------------------------
    # Check total
    # --------------------------------------------------------

    expected_total = (
        len(train_lines)
        +
        len(test_lines)
    )

    actual_total = len(
        final_df
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOTAL IMAGE CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"Expected: "
        f"{expected_total:,}"
    )

    print(
        f"Actual:   "
        f"{actual_total:,}"
    )

    if actual_total != expected_total:

        raise RuntimeError(
            "Total image count mismatch."
        )

    print(
        "✓ Total image count correct."
    )

    # --------------------------------------------------------
    # Check duplicate paths
    # --------------------------------------------------------

    check_duplicate_paths(
        final_df
    )

    # --------------------------------------------------------
    # Check split leakage
    # --------------------------------------------------------

    check_split_separation(
        final_df
    )

    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    print_class_distribution(
        final_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        REPORT_DIR
        / "dataset_split.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        final_df[
            "split"
        ].value_counts()
    )

    print(
        f"\nTotal: "
        f"{len(final_df):,}"
    )

    print(
        f"\nSaved:"
    )

    print(
        output_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET SPLIT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()