export default function VideoPreview({ src, type, downloadUrl, onDownload }) {
  return (
    <div className="video-preview">
      {src ? (
        <video controls>
          {type ? <source src={src} type={type} /> : null}
          Your browser does not support video.
        </video>
      ) : (
        <p style={{ opacity: 0.8 }}>Upload a clip to preview.</p>
      )}
      {downloadUrl ? (
        <button type="button" className="download-btn" onClick={onDownload}>
          Start Download
        </button>
      ) : null}
    </div>
  );
}
