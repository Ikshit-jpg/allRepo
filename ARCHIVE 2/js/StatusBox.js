export default function StatusBox({ text }) {
  return (
    <pre
      id="archiveStatus"
      style={{
        whiteSpace: "pre-wrap",
        fontSize: "0.86rem",
        opacity: 0.95,
        padding: "10px 12px",
        border: "1px solid rgba(255,255,255,0.2)",
        borderRadius: "10px",
        background: "rgba(255,255,255,0.03)",
      }}
    >
      {text}
    </pre>
  );
}
