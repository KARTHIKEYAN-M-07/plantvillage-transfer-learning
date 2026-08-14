function Loading({ t }) {

  return (

    <div className="loading">

      <div className="spinner"></div>

      <p>
        {t.analyzingImage}
      </p>

      <small>
        {t.pleaseWait}
      </small>

    </div>

  );
}

export default Loading;