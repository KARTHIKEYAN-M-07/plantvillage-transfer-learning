from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

DATASET_DIR = DATA_DIR / "PlantVillage"

REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# DATASET
# ============================================================

CLASS_NAMES_PATH = (
    PROJECT_ROOT / "class_names.json"
)


# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 224

NUM_CHANNELS = 3


# ============================================================
# SPLIT
# ============================================================

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# ============================================================
# GPU TRAINING
# ============================================================

# RTX 3050 6 GB
BATCH_SIZE = 32


# Windows
NUM_WORKERS = 4


# ============================================================
# CREATE DIRECTORIES
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)