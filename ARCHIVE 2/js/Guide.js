export default function Guide() {
  return (
    <>
      <header className="brand-header">
        <div className="logo-shell">
          <h1 className="logo" data-text="ARCHIVE">
            ARCHIVE
          </h1>
          <p className="logo-subtitle">AI Anime Edit Engine</p>
        </div>
      </header>
      <p className="tagline">Ai Video Editor, Only Anime Edits</p>
      <section className="guide-bubbles" aria-label="How to use ARCHIVE">
        <article className="guide-bubble">
          <h2>1. Set Your Edit</h2>
          <p>Write your prompt with style + constraints.</p>
          <p className="guide-example">Example: "Fire boundary on impact beats, no color change."</p>
        </article>
        <article className="guide-bubble">
          <h2>2. Upload & Render</h2>
          <p>Upload your clip, click Start Editing, then preview and download.</p>
          <p className="guide-example">Tip: Keep clips under 45s for fastest results.</p>
        </article>
      </section>
    </>
  );
}
