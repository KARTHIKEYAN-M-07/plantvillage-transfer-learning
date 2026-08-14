function PredictionResult({ result, t }) {

  if (!result) {

    return (

      <div className="result-card">

        <div className="result-icon">
          🧪
        </div>

        <h2>
          {t.resultTitle}
        </h2>

        <p>
          {t.uploadFirst}
        </p>

        <div className="result-placeholder">

          {t.disease}: {t.noPrediction}

          <br />

          {t.confidence}: {t.noPrediction}

        </div>

      </div>

    );
  }

  return (

    <div className="result-card">

      <div className="result-icon">
        🌱
      </div>

      <h2>
        {t.resultTitle}
      </h2>

      <div className="prediction-result">

        <p>

          <strong>
            {t.disease}:
          </strong>

          <br />

          {result.disease}

        </p>

        <p>

          <strong>
            {t.confidence}:
          </strong>

          <br />

          {(result.confidence * 100).toFixed(2)}%

        </p>

      </div>

    </div>

  );
}

export default PredictionResult;