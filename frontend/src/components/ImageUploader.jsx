function ImageUploader({ onFileChange, t }) {

  const handleChange = (event) => {

    const file = event.target.files[0];

    if (file) {
      onFileChange(file);
    }

  };

  return (

    <div className="upload-card">

      <div className="upload-icon">
        🌿
      </div>

      <h2>
        {t.uploadTitle}
      </h2>

      <p>
        {t.uploadDescription}
      </p>

      <label className="upload-button">

        {t.chooseImage}

        <input
          type="file"
          accept=".jpg,.jpeg,.png"
          onChange={handleChange}
          hidden
        />

      </label>

      <small>
        {t.supportedFormats}
      </small>

    </div>

  );
}

export default ImageUploader;