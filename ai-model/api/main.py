"""
AI MODEL HTTP SERVICE
=====================

This file exposes the existing AI pipeline through FastAPI.

Architecture:

Frontend
    ↓
Backend
    ↓ HTTP
AI API (this file)
    ↓
ai_pipeline.predict()
    ↓
EfficientNet-B0
    ↓
Structured AI Contract
"""

from pathlib import Path
import sys
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT PATHS
# ============================================================

# ai-model/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# src/
SRC_DIR = PROJECT_ROOT / "src"

# reports/
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# MAKE src IMPORTABLE
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT EXISTING AI PIPELINE
# ============================================================

try:
    from ai_pipeline import predict
except Exception as e:
    raise RuntimeError(
        f"Failed to import AI pipeline: {e}"
    ) from e


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Plant Disease AI Service",
    description=(
        "AI inference service for the Plant Disease Detection system. "
        "Uses the existing EfficientNet-B0 AI pipeline."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# During development, allow requests from other machines.
#
# For hackathon LAN testing this is convenient.
# Production deployment should restrict this list.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 70)
    print("PLANT DISEASE AI SERVICE")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reports directory: {REPORTS_DIR}")

    if not REPORTS_DIR.exists():
        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    print("✓ AI service initialized.")
    print("=" * 70)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Plant Disease AI Service",
        "status": "running",
        "version": "1.0.0",
        "model": "EfficientNet-B0",
        "classes": 38,
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "plant-disease-ai",
        "model": "EfficientNet-B0",
        "classes": 38,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_image(
    image: UploadFile = File(...)
):
    """
    Receive an uploaded image and send it through
    the existing AI pipeline.

    Request:
        multipart/form-data

    Field:
        image

    Returns:
        AI Contract V1 JSON
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not image.filename:
        raise HTTPException(
            status_code=400,
            detail="No image filename provided."
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    extension = Path(image.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Supported formats: JPG, JPEG, PNG, BMP, WEBP."
            )
        )

    # --------------------------------------------------------
    # Create temporary directory
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="plant_ai_"
        )
    )

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    temp_image_path = (
        temp_dir / unique_filename
    )

    try:

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        with temp_image_path.open("wb") as buffer:

            shutil.copyfileobj(
                image.file,
                buffer
            )

        # ----------------------------------------------------
        # Verify file exists
        # ----------------------------------------------------

        if not temp_image_path.exists():

            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded image."
            )

        # ----------------------------------------------------
        # Run existing AI pipeline
        # ----------------------------------------------------

        result = predict(
            str(temp_image_path)
        )

        # ----------------------------------------------------
        # Return AI Contract V1
        # ----------------------------------------------------

        return result

    except HTTPException:
        raise

    except Exception as e:

        print("=" * 70)
        print("AI PREDICTION ERROR")
        print("=" * 70)
        print(str(e))
        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail="AI prediction failed."
        )

    finally:

        # ----------------------------------------------------
        # Remove uploaded temporary file
        # ----------------------------------------------------

        try:

            if temp_image_path.exists():
                temp_image_path.unlink()

            if temp_dir.exists():
                temp_dir.rmdir()

        except Exception:
            # Cleanup failure should not break the response.
            pass


# ============================================================
# GRAD-CAM FILE ENDPOINT
# ============================================================

@app.get("/heatmap/{filename}")
def get_heatmap(filename: str):
    """
    Serve a generated Grad-CAM heatmap.

    IMPORTANT:
    Only files inside the reports directory are allowed.
    """

    # --------------------------------------------------------
    # Prevent path traversal
    # --------------------------------------------------------

    safe_filename = Path(filename).name

    heatmap_path = (
        REPORTS_DIR / safe_filename
    ).resolve()

    reports_root = REPORTS_DIR.resolve()

    # Make sure requested file stays inside reports/
    try:

        heatmap_path.relative_to(
            reports_root
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid heatmap path."
        )

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not heatmap_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Heatmap not found."
        )

    # --------------------------------------------------------
    # Return image
    # --------------------------------------------------------

    return FileResponse(
        path=str(heatmap_path),
        media_type="image/jpeg",
        filename=heatmap_path.name,
    )


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/info")
def info():

    return {
        "service": "Plant Disease AI Service",
        "version": "1.0.0",
        "model": {
            "name": "EfficientNet-B0",
            "classes": 38,
        },
        "features": [
            "image_quality",
            "image_suitability",
            "plant_disease_prediction",
            "confidence_handling",
            "gradcam",
            "disease_information",
        ],
        "contract_version": "V1",
    }