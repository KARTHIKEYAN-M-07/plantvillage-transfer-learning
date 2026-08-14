import { useState } from "react";
import "./App.css";
import { predictPlant } from "./services/api";

function App() {
  // ==============================
  // THEME
  // ==============================
  const [darkMode, setDarkMode] = useState(false);

  // ==============================
  // LANGUAGE
  // ==============================
  const [language, setLanguage] = useState("en");

  // ==============================
  // IMAGE
  // ==============================
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  // ==============================
  // RESULT
  // ==============================
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ==============================
  // TRANSLATIONS
  // ==============================
  const text = {
    en: {
      appTitle: "PlantVillage AI",
      appSubtitle: "Plant Disease Diagnoser",

      badge: "🤖 AI POWERED",

      title1: "Detect Plant Diseases",
      title2: "Instantly with AI",

      description:
        "Upload a plant leaf image and our AI model will analyze it to identify possible diseases.",

      uploadTitle: "Upload Plant Leaf",

      uploadDescription:
        "Choose a clear image of a plant leaf for disease detection.",

      chooseImage: "📷 Choose Leaf Image",

      supported: "Supported: JPG, JPEG, PNG",

      selected: "Selected Image",

      analyze: "🔍 Analyze Plant",

      analyzing: "🔍 Analyzing your leaf...",

      uploading: "Uploading image...",

      resultTitle: "AI Analysis Result",

      waiting:
        "Upload a plant leaf image and click Analyze Plant to see the result.",

      plant: "Plant",

      disease: "Disease",

      confidence: "Confidence",

      high: "✓ High Confidence",

      moderate: "⚠ Moderate Confidence",

      low: "⚠ Low Confidence",

      reviewTitle: "⚠️ Prediction Needs Review",

      reviewText:
        "The AI has lower confidence in this prediction. Consider uploading a clearer leaf image or verifying the result.",

      imageQuality: "Image Quality",

      good: "✓ Good",

      poor: "⚠ Poor",

      qualityIssues: "Image Quality Issues",

      suitability: "Leaf Suitability",

      suitable: "✓ Suitable for analysis",

      reviewSuitability: "⚠ Image suitability uncertain",

      unsuitable: "❌ Image is unsuitable for plant analysis",

      score: "Score",

      suitabilityIssues: "Suitability Issues",

      explanation: "AI Explanation",

      gradcam: "Grad-CAM Heatmap",

      diseaseInfo: "Disease Information",

      status: "Status",

      severity: "Severity",

      descriptionLabel: "Description",

      symptoms: "Symptoms",

      management: "General Management",

      prevention: "Prevention",

      healthy: "✓ Plant appears healthy",

      analyzeAnother: "🔄 Analyze Another Image",

      backendError:
        "Unable to connect to the server. Please try again.",

      invalidImage:
        "Please upload a valid JPG, JPEG, or PNG image.",

      poorImage:
        "The image quality is too poor for reliable analysis. Please upload a clearer image.",

      unsuitableImage:
        "This does not appear to be a suitable plant/leaf image. Please upload a clear leaf image.",

      aiError:
        "We couldn't analyze this image. Please try again.",

      light: "☀️ Light",

      dark: "🌙 Dark",

      footer:
        "© 2026 PlantVillage AI — AI Based Plant Disease Detection",
    },

    ta: {
      appTitle: "PlantVillage AI",

      appSubtitle: "தாவர நோய் கண்டறியும் அமைப்பு",

      badge: "🤖 செயற்கை நுண்ணறிவு",

      title1: "தாவர நோய்களை கண்டறியுங்கள்",

      title2: "AI மூலம் உடனடியாக",

      description:
        "தாவர இலை படத்தை பதிவேற்றவும். எங்களின் AI மாதிரி படத்தை பகுப்பாய்வு செய்து சாத்தியமான நோயை கண்டறியும்.",

      uploadTitle: "தாவர இலை பதிவேற்றம்",

      uploadDescription:
        "நோயை கண்டறிய தெளிவான தாவர இலை படத்தை தேர்வு செய்யவும்.",

      chooseImage: "📷 இலை படத்தை தேர்வு செய்யவும்",

      supported: "ஆதரிக்கப்படும் வகைகள்: JPG, JPEG, PNG",

      selected: "தேர்ந்தெடுக்கப்பட்ட படம்",

      analyze: "🔍 தாவரத்தை பகுப்பாய்வு செய்யவும்",

      analyzing: "🔍 உங்கள் இலை படத்தை பகுப்பாய்வு செய்கிறது...",

      uploading: "படம் பதிவேற்றப்படுகிறது...",

      resultTitle: "AI பகுப்பாய்வு முடிவு",

      waiting:
        "தாவர இலை படத்தை பதிவேற்றி பகுப்பாய்வு பொத்தானை கிளிக் செய்யவும்.",

      plant: "தாவரம்",

      disease: "நோய்",

      confidence: "நம்பகத்தன்மை",

      high: "✓ அதிக நம்பகத்தன்மை",

      moderate: "⚠ மிதமான நம்பகத்தன்மை",

      low: "⚠ குறைந்த நம்பகத்தன்மை",

      reviewTitle: "⚠️ கணிப்பு சரிபார்க்கப்பட வேண்டும்",

      reviewText:
        "இந்த கணிப்பில் AI-க்கு குறைந்த நம்பகத்தன்மை உள்ளது. தெளிவான இலை படத்தை பதிவேற்றவும் அல்லது முடிவை சரிபார்க்கவும்.",

      imageQuality: "படத்தின் தரம்",

      good: "✓ நல்லது",

      poor: "⚠ மோசமானது",

      qualityIssues: "படத்தின் தர சிக்கல்கள்",

      suitability: "இலை பொருத்தம்",

      suitable: "✓ பகுப்பாய்விற்கு பொருத்தமானது",

      reviewSuitability: "⚠ இலை பொருத்தம் உறுதியற்றது",

      unsuitable: "❌ தாவர பகுப்பாய்விற்கு பொருத்தமற்ற படம்",

      score: "மதிப்பெண்",

      suitabilityIssues: "பொருத்த சிக்கல்கள்",

      explanation: "AI விளக்கம்",

      gradcam: "Grad-CAM வெப்ப வரைபடம்",

      diseaseInfo: "நோய் தகவல்",

      status: "நிலை",

      severity: "தீவிரம்",

      descriptionLabel: "விளக்கம்",

      symptoms: "அறிகுறிகள்",

      management: "பொதுவான மேலாண்மை",

      prevention: "தடுப்பு",

      healthy: "✓ தாவரம் ஆரோக்கியமாக உள்ளது",

      analyzeAnother: "🔄 மற்றொரு படத்தை பகுப்பாய்வு செய்யவும்",

      backendError:
        "சேவையகத்துடன் இணைக்க முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",

      invalidImage:
        "சரியான JPG, JPEG அல்லது PNG படத்தை பதிவேற்றவும்.",

      poorImage:
        "படத்தின் தரம் நம்பகமான பகுப்பாய்விற்கு மிகவும் குறைவாக உள்ளது. தெளிவான படத்தை பதிவேற்றவும்.",

      unsuitableImage:
        "இது பொருத்தமான தாவர/இலை படம் போல் தெரியவில்லை. தெளிவான இலை படத்தை பதிவேற்றவும்.",

      aiError:
        "இந்த படத்தை பகுப்பாய்வு செய்ய முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",

      light: "☀️ வெளிச்சம்",

      dark: "🌙 இருள்",

      footer:
        "© 2026 PlantVillage AI — AI அடிப்படையிலான தாவர நோய் கண்டறிதல்",
    },
  };

  const t = text[language];

  // ==============================
  // FILE SELECT
  // ==============================
  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    const allowedTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
    ];

    if (!allowedTypes.includes(file.type)) {
      setError(t.invalidImage);
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setError("");

    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
  };

  // ==============================
  // ANALYZE IMAGE
  // ==============================
  const handlePrediction = async () => {
    if (!selectedFile) {
      setError(t.invalidImage);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await predictPlant(selectedFile);

      setResult(data);
    } catch (err) {
      console.error(err);

      if (err?.message === "UNSUITABLE") {
        setError(t.unsuitableImage);
      } else if (err?.message === "POOR_IMAGE") {
        setError(t.poorImage);
      } else if (err?.message === "BACKEND_ERROR") {
        setError(t.backendError);
      } else {
        setError(t.aiError);
      }
    } finally {
      setLoading(false);
    }
  };

  // ==============================
  // RESET
  // ==============================
  const handleReset = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setResult(null);
    setError("");
    setLoading(false);
  };

  // ==============================
  // CONFIDENCE
  // ==============================
  const getConfidenceText = (level) => {
    if (level === "high") return t.high;
    if (level === "moderate") return t.moderate;
    if (level === "low") return t.low;

    return "";
  };

  // ==============================
  // MAIN UI
  // ==============================
  return (
    <div
      className={`app ${
        darkMode ? "dark-mode" : "light-mode"
      }`}
    >
      {/* HEADER */}
      <header className="header">
        <div className="brand">
          <div className="logo">🌱</div>

          <div>
            <h1>{t.appTitle}</h1>
            <p>{t.appSubtitle}</p>
          </div>
        </div>

        <div className="settings">
          {/* THEME */}
          <button
            type="button"
            className="theme-button"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? t.light : t.dark}
          </button>

          {/* LANGUAGE */}
          <div className="language-switcher">
            <span>🌐</span>

            <button
              type="button"
              className={
                language === "en"
                  ? "active-language"
                  : ""
              }
              onClick={() => setLanguage("en")}
            >
              🇬🇧 English
            </button>

            <button
              type="button"
              className={
                language === "ta"
                  ? "active-language"
                  : ""
              }
              onClick={() => setLanguage("ta")}
            >
              🇮🇳 தமிழ்
            </button>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="main-container">
        {/* HERO */}
        <section className="hero">
          <div className="badge">{t.badge}</div>

          <h2>
            {t.title1}
            <br />
            <span>{t.title2}</span>
          </h2>

          <p>{t.description}</p>
        </section>

        {/* UPLOAD */}
        {!result && (
          <div className="content">
            <div className="upload-card">
              <div className="upload-icon">🌿</div>

              <h2>{t.uploadTitle}</h2>

              <p>{t.uploadDescription}</p>

              <label
                htmlFor="leaf-image"
                className="upload-button"
              >
                {t.chooseImage}
              </label>

              <input
                id="leaf-image"
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />

              <small>{t.supported}</small>

              {selectedFile && imagePreview && (
                <div className="selected-file">
                  <img
                    src={imagePreview}
                    alt="Selected plant leaf"
                    className="image-preview"
                  />

                  <div>
                    {t.selected}

                    <strong>
                      {selectedFile.name}
                    </strong>
                  </div>
                </div>
              )}

              <button
                type="button"
                className="predict-button"
                onClick={handlePrediction}
                disabled={!selectedFile || loading}
              >
                {loading ? t.analyzing : t.analyze}
              </button>

              {loading && (
                <div className="loading">
                  <div className="spinner"></div>

                  <p>{t.analyzing}</p>

                  <small>{t.uploading}</small>
                </div>
              )}

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}
            </div>

            {!loading && (
              <div className="result-card">
                <div className="result-icon">🧪</div>

                <h2>{t.resultTitle}</h2>

                <div className="result-placeholder">
                  <p>{t.waiting}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* RESULT PAGE */}
        {result && (
          <section className="analysis-result">
            <div className="result-header">
              <div className="result-icon">🌱</div>

              <h2>{t.resultTitle}</h2>
            </div>

            {/* ORIGINAL IMAGE */}
            {imagePreview && (
              <div className="original-image-card">
                <h3>{t.selected}</h3>

                <img
                  src={imagePreview}
                  alt="Original plant leaf"
                  className="result-image"
                />
              </div>
            )}

            {/* SUITABILITY PRIORITY */}
            {result.suitability?.status ===
              "unsuitable" && (
              <div className="warning-card danger">
                <h3>{t.unsuitable}</h3>

                {result.suitability.issues?.map(
                  (issue, index) => (
                    <p key={index}>• {issue}</p>
                  )
                )}
              </div>
            )}

            {/* BASIC PREDICTION */}
            {result.suitability?.status !==
              "unsuitable" &&
              result.prediction && (
                <div className="prediction-grid">
                  <div className="info-card">
                    <span>🌿</span>
                    <small>{t.plant}</small>
                    <strong>
                      {result.prediction.plant || "-"}
                    </strong>
                  </div>

                  <div className="info-card">
                    <span>🦠</span>
                    <small>{t.disease}</small>
                    <strong>
                      {result.prediction.disease || "-"}
                    </strong>
                  </div>

                  <div className="info-card">
                    <span>🎯</span>
                    <small>{t.confidence}</small>
                    <strong>
                      {result.prediction.confidence ??
                        "-"}
                      %
                    </strong>
                  </div>

                  <div className="info-card">
                    <span>📊</span>
                    <small>
                      {getConfidenceText(
                        result.prediction
                          .confidence_level
                      )}
                    </small>

                    <strong>
                      {result.prediction
                        .confidence_level || "-"}
                    </strong>
                  </div>
                </div>
              )}

            {/* REVIEW WARNING */}
            {result.prediction?.needs_review && (
              <div className="warning-card">
                <h3>{t.reviewTitle}</h3>

                <p>{t.reviewText}</p>
              </div>
            )}

            {/* HEALTHY */}
            {result.disease_information?.status ===
              "healthy" && (
              <div className="healthy-card">
                <h2>{t.healthy}</h2>
              </div>
            )}

            {/* IMAGE QUALITY */}
            {result.image_quality && (
              <div className="detail-card">
                <h3>{t.imageQuality}</h3>

                <div
                  className={
                    result.image_quality.valid
                      ? "status-good"
                      : "status-warning"
                  }
                >
                  {result.image_quality.valid
                    ? t.good
                    : t.poor}
                </div>

                {result.image_quality.issues
                  ?.length > 0 && (
                  <>
                    <h4>{t.qualityIssues}</h4>

                    <ul>
                      {result.image_quality.issues.map(
                        (issue, index) => (
                          <li key={index}>
                            {issue}
                          </li>
                        )
                      )}
                    </ul>
                  </>
                )}
              </div>
            )}

            {/* SUITABILITY */}
            {result.suitability && (
              <div className="detail-card">
                <h3>{t.suitability}</h3>

                <div
                  className={`suitability-status ${result.suitability.status}`}
                >
                  {result.suitability.status ===
                    "suitable" && t.suitable}

                  {result.suitability.status ===
                    "review" &&
                    t.reviewSuitability}

                  {result.suitability.status ===
                    "unsuitable" &&
                    t.unsuitable}
                </div>

                {result.suitability.score !==
                  undefined && (
                  <p>
                    <strong>{t.score}:</strong>{" "}
                    {result.suitability.score}
                  </p>
                )}

                {result.suitability.issues
                  ?.length > 0 && (
                  <>
                    <h4>{t.suitabilityIssues}</h4>

                    <ul>
                      {result.suitability.issues.map(
                        (issue, index) => (
                          <li key={index}>
                            {issue}
                          </li>
                        )
                      )}
                    </ul>
                  </>
                )}
              </div>
            )}

            {/* GRAD-CAM */}
            {result.explanation?.available &&
              result.explanation?.heatmap_url && (
                <div className="detail-card">
                  <h3>{t.explanation}</h3>

                  <p>
                    {result.explanation.method ||
                      "Grad-CAM"}
                  </p>

                  <img
                    src={result.explanation.heatmap_url}
                    alt="Grad-CAM heatmap"
                    className="heatmap-image"
                  />

                  <h4>{t.gradcam}</h4>
                </div>
              )}

            {/* DISEASE INFORMATION */}
            {result.disease_information &&
              result.disease_information.status !==
                "healthy" && (
                <div className="disease-information">
                  <h2>{t.diseaseInfo}</h2>

                  <div className="detail-card">
                    <p>
                      <strong>{t.status}:</strong>{" "}
                      {result.disease_information.status ||
                        "-"}
                    </p>

                    <p>
                      <strong>{t.severity}:</strong>{" "}
                      {result.disease_information
                        .severity || "-"}
                    </p>

                    <h3>{t.descriptionLabel}</h3>

                    <p>
                      {result.disease_information
                        .description || "-"}
                    </p>
                  </div>

                  {/* SYMPTOMS */}
                  {result.disease_information
                    .symptoms?.length > 0 && (
                    <div className="detail-card">
                      <h3>{t.symptoms}</h3>

                      <ul>
                        {result.disease_information.symptoms.map(
                          (item, index) => (
                            <li key={index}>
                              {item}
                            </li>
                          )
                        )}
                      </ul>
                    </div>
                  )}

                  {/* MANAGEMENT */}
                  {result.disease_information
                    .general_management?.length > 0 && (
                    <div className="detail-card">
                      <h3>{t.management}</h3>

                      <ul>
                        {result.disease_information.general_management.map(
                          (item, index) => (
                            <li key={index}>
                              {item}
                            </li>
                          )
                        )}
                      </ul>
                    </div>
                  )}

                  {/* PREVENTION */}
                  {result.disease_information
                    .prevention?.length > 0 && (
                    <div className="detail-card">
                      <h3>{t.prevention}</h3>

                      <ul>
                        {result.disease_information.prevention.map(
                          (item, index) => (
                            <li key={index}>
                              {item}
                            </li>
                          )
                        )}
                      </ul>
                    </div>
                  )}
                </div>
              )}

            {/* RESET */}
            <button
              type="button"
              className="reset-button"
              onClick={handleReset}
            >
              {t.analyzeAnother}
            </button>
          </section>
        )}
      </main>

      {/* FOOTER */}
      <footer>{t.footer}</footer>
    </div>
  );
}

export default App;