const API_URL =
  import.meta.env.VITE_API_URL || "http://10.10.187.79:8000";

/**
 * Convert a backend-relative URL into a complete browser URL.
 *
 * Example:
 * /gradcam/example.jpg
 *
 * becomes:
 * http://10.10.187.79:8000/gradcam/example.jpg
 */
function makeBackendUrl(url) {
  if (!url) {
    return null;
  }

  if (
    url.startsWith("http://") ||
    url.startsWith("https://")
  ) {
    return url;
  }

  if (url.startsWith("/")) {
    return API_URL + url;
  }

  return API_URL + "/" + url;
}

/**
 * Convert the backend response into the structure
 * expected by the existing React UI.
 */
function normalizePredictionResponse(data) {
  return {
    ...data,

    // ==============================
    // PREDICTION
    // ==============================

    prediction: {
      plant: data.plant,
      disease: data.disease,
      confidence: data.confidence,
      confidence_level: data.confidence_level,
      needs_review: data.needs_review,
    },

    // ==============================
    // DISEASE INFORMATION
    // ==============================

    disease_information:
      data.disease_info ||
      data.disease_information ||
      null,

    // ==============================
    // GRAD-CAM
    // ==============================

    explanation: {
      ...(data.explanation || {}),

      available:
        data.explanation?.available ??
        Boolean(data.gradcam_url),

      method:
        data.explanation?.method ||
        "Grad-CAM",

      heatmap_url: makeBackendUrl(
        data.gradcam_url ||
        data.explanation?.heatmap_url
      ),
    },
  };
}

/**
 * Send plant leaf image to backend.
 *
 * Frontend PC
 *      |
 *      | POST /predict
 *      v
 * Backend PC :8000
 *      |
 *      v
 * AI PC :8001
 */
export async function predictPlant(image) {
  const formData = new FormData();

  formData.append("image", image);

  let response;

  try {
    response = await fetch(API_URL + "/predict", {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    console.error(
      "Backend connection error:",
      error
    );

    throw new Error("BACKEND_ERROR");
  }

  // ==============================
  // HTTP ERROR HANDLING
  // ==============================

  if (!response.ok) {
    if (response.status === 400) {
      throw new Error("INVALID_IMAGE");
    }

    if (response.status === 413) {
      throw new Error("FILE_TOO_LARGE");
    }

    if (response.status === 422) {
      throw new Error("UNSUITABLE");
    }

    if (response.status >= 500) {
      throw new Error("AI_ERROR");
    }

    throw new Error("BACKEND_ERROR");
  }

  // ==============================
  // READ JSON RESPONSE
  // ==============================

  let data;

  try {
    data = await response.json();
  } catch (error) {
    console.error(
      "Invalid JSON response:",
      error
    );

    throw new Error("BACKEND_ERROR");
  }

  // ==============================
  // BACKEND SUCCESS CHECK
  // ==============================

  if (!data.success) {
    throw new Error("AI_ERROR");
  }

  // ==============================
  // DEBUG
  // ==============================

  console.log(
    "Raw backend response:",
    data
  );

  // ==============================
  // NORMALIZE RESPONSE
  // ==============================

  const normalizedData =
    normalizePredictionResponse(data);

  console.log(
    "Normalized frontend response:",
    normalizedData
  );

  return normalizedData;
}
