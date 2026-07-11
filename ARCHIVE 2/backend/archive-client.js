(() => {
  const { useState, useEffect, useMemo, useRef } = React;
  const h = React.createElement;
  const API_BASE = window.ARCHIVE_API_BASE || window.location.origin;
  /*
  const SUPABASE_URL = "https://pymvlhtyeohnbanlweuf.supabase.co";
  const SUPABASE_ANON_KEY = "sb_publishable_e-VL1bDjYkxGzm3SIvJPLw_QPsEYIRJ";
  const supabase = window.supabase
    ? window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
    : null;
  */
  const supabase = null;

  const formatDuration = (secondsRaw) => {
    const total = Math.max(0, Math.floor(Number(secondsRaw) || 0));
    if (total < 60) return `${total}s`;
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${minutes}m ${seconds}s`;
  };

  const StarField = React.memo(({ count = 70 }) => {
    const stars = useMemo(
      () =>
        Array.from({ length: count }).map((_, i) => {
          const size = Math.random() * 4 + 3;
          const twinkle = Math.random() * 1 + 2.5;
          const drift = Math.random() * 2 + 6;
          const alpha = Math.random() * 0.25 + 0.55;
          const blur = Math.random() * 6 + 8;
          return {
            id: i,
            size,
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            delay: `${Math.random() * 5}s`,
            twinkle,
            drift,
            alpha,
            blur,
          };
        }),
      [count]
    );

    return h(
      "div",
      { className: "stars-container" },
      stars.map((s) =>
        h("div", {
          key: s.id,
          className: "star",
          style: {
            width: `${s.size}px`,
            height: `${s.size}px`,
            top: s.top,
            left: s.left,
            animationDelay: s.delay,
            "--twinkle": `${s.twinkle}s`,
            "--drift": `${s.drift}s`,
            "--star-alpha": s.alpha,
            "--star-blur": `${s.blur}px`,
          },
        })
      )
    );
  });

  const StatusBox = ({ text }) =>
    h(
      "pre",
      {
        id: "archiveStatus",
        style: {
          whiteSpace: "pre-wrap",
          fontSize: "0.86rem",
          opacity: 0.95,
          padding: "10px 12px",
          border: "1px solid rgba(255,255,255,0.2)",
          borderRadius: "10px",
          background: "rgba(255,255,255,0.03)",
        },
      },
      text
    );

  const LoadingOverlay = ({ showing, elapsed, onCancel }) => {
    if (!showing) return null;
    return h(
      "div",
      { className: "archive-loading-overlay" },
      h("div", { className: "archive-loading-box" }, [
        h("div", { className: "archive-loading-spinner", key: "sp" }),
        h("div", { className: "archive-loading-title", key: "ttl" }, "Editing your video..."),
        h("div", { className: "archive-loading-time", key: "tm" }, `Elapsed: ${formatDuration(elapsed)}`),
        h(
          "button",
          { type: "button", className: "archive-cancel-btn", onClick: onCancel, key: "btn" },
          "Cancel"
        ),
      ])
    );
  };

  const Guide = () =>
    h(
      React.Fragment,
      null,
      h("header", { className: "brand-header" }, [
        h("div", { className: "logo-shell", key: "ls" }, [
          h("h1", { className: "logo", "data-text": "ARCHIVE", key: "lg" }, "ARCHIVE"),
          h("p", { className: "logo-subtitle", key: "sub" }, "AI Anime Edit Engine"),
        ]),
      ]),
      h("p", { className: "tagline" }, "Ai Video Editor, Only Anime Edits"),
      h("section", { className: "guide-bubbles", "aria-label": "How to use ARCHIVE" }, [
        h("article", { className: "guide-bubble", key: "g1" }, [
          h("h2", null, "1. Set Your Edit"),
          h("p", null, "Write your prompt with style + constraints."),
          h("p", { className: "guide-example" }, 'Example: "Fire boundary on impact beats, no color change."'),
        ]),
        h("article", { className: "guide-bubble", key: "g2" }, [
          h("h2", null, "2. Upload & Render"),
          h("p", null, "Upload your clip, click Start Editing, then preview and download."),
          h("p", { className: "guide-example" }, "Tip: Keep clips under 45s for fastest results."),
        ]),
      ])
    );

  const Footer = () =>
    h(
      "footer",
      { className: "site-footer", "aria-label": "ARCHIVE credentials" },
      h("div", { className: "footer-badge" }, "AI Anime Editing Platform"),
      h("p", { className: "footer-title" }, "ARCHIVE Studio"),
      h("p", { className: "footer-credit" }, [
        "Crafted by ",
        h("span", { key: "nm" }, "IKSHIT GOYAL"),
      ]),
      h(
        "div",
        { className: "footer-pills", "aria-label": "platform highlights" },
        ["Adaptive Beat Sync", "Anime-Only Pipeline", "Prompt Aware Effects", "Privacy-First Runtime"].map((t) =>
          h("span", { key: t }, t)
        )
      ),
      h(
        "p",
        { className: "footer-meta" },
        "Intelligent beat-sync edits with smart effect timing, masked transitions, and controlled audio matching."
      )
    );

  const VideoPreview = ({ src, type, downloadUrl, onDownload }) =>
    h(
      "div",
      { className: "video-preview" },
      src
        ? h(
            "video",
            {
              key: src,
              controls: true,
              src,
              type: type || "video/mp4",
            },
            "Your browser does not support video."
          )
        : h("p", { style: { opacity: 0.8 } }, "Upload a clip to preview."),
      downloadUrl
        ? h(
            "button",
            { type: "button", className: "download-btn", onClick: onDownload },
            "Start Download"
          )
        : null
    );

  function App() {
    const [prompt, setPrompt] = useState("");
    const [file, setFile] = useState(null);
    const [fileUrl, setFileUrl] = useState("");
    const [renderUrl, setRenderUrl] = useState("");
    const [downloadUrl, setDownloadUrl] = useState("");
    const [status, setStatus] = useState("ARCHIVE backend: idle");
    const [isRendering, setIsRendering] = useState(false);
    const [elapsed, setElapsed] = useState(0);
    const [authReady, setAuthReady] = useState(false);
    const [session, setSession] = useState(null);
    const [authBusy, setAuthBusy] = useState(false);
    const [authError, setAuthError] = useState("");
    const [authMode, setAuthMode] = useState("login");
    const [authEmail, setAuthEmail] = useState("");
    const [authPassword, setAuthPassword] = useState("");
    const [showStars, setShowStars] = useState(false);

    const fileInputRef = useRef(null);
    const timerRef = useRef(null);
    const abortRef = useRef(null);
    const downloadBusyRef = useRef(false);
    const renderIdRef = useRef(null);

    useEffect(() => {
      try {
        localStorage.removeItem("archive_last_render_url");
        localStorage.removeItem("archive_last_render_summary");
      } catch (err) {
        console.warn("Could not clear old render cache", err);
      }
    }, []);

    useEffect(() => {
      setAuthReady(true);
      return () => {};
    }, []);

    useEffect(() => {
      const schedule =
        window.requestIdleCallback ||
        ((cb) => window.setTimeout(cb, 180));
      const cancel =
        window.cancelIdleCallback ||
        ((id) => window.clearTimeout(id));
      const id = schedule(() => setShowStars(true));
      return () => cancel(id);
    }, []);

    useEffect(() => {
      if (!isRendering) {
        setElapsed(0);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        return;
      }
      const started = Date.now();
      timerRef.current = setInterval(() => {
        setElapsed(Math.max(0, Math.floor((Date.now() - started) / 1000)));
      }, 1000);
      return () => {
        if (timerRef.current) clearInterval(timerRef.current);
      };
    }, [isRendering]);

    const buttonLabel = file ? (isRendering ? "Rendering..." : "Start Editing") : "Upload Video";

    const handleUploadClick = () => {
      if (!file) {
        fileInputRef.current?.click();
        return;
      }
      if (!isRendering) {
        void handleRender();
      }
    };

    const handleFileChange = (event) => {
      const chosen = event.target.files && event.target.files[0];
      if (!chosen) return;
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      const url = URL.createObjectURL(chosen);
      setFile(chosen);
      setFileUrl(url);
      setRenderUrl("");
      setDownloadUrl("");
    };

    const handleDownload = async () => {
      if (!downloadUrl || downloadBusyRef.current) return;
      downloadBusyRef.current = true;
      try {
        const res = await fetch(downloadUrl);
        if (!res.ok) throw new Error("Download failed");
        const blob = await res.blob();
        const objUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = objUrl;
        link.download = `archive_edit_${Date.now()}.mp4`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(objUrl);
      } catch (err) {
        alert(`Download error: ${err.message}`);
      } finally {
        downloadBusyRef.current = false;
      }
    };

    const cancelRender = () => {
      // Abort the HTTP request
      if (abortRef.current) {
        abortRef.current.abort();
      }
      
      // Also tell the backend to cancel the render process
      if (renderIdRef.current) {
        const renderId = renderIdRef.current;
        fetch(`${API_BASE}/api/cancel/${renderId}`, {
          method: "POST",
        }).catch(err => console.warn("Could not cancel render on backend", err));
      }
    };

    const buildSummary = (data) => {
      const activeEffects = data.style_profile
        ? Object.entries(data.style_profile)
            .filter(
              ([key, value]) =>
                key !== "intensity" &&
                !["selective_beats", "all_beats", "fast_render", "quality_priority", "smart_music"].includes(key) &&
                Boolean(value)
            )
            .map(([key]) => key)
            .join(", ")
        : "n/a";
      const visualMood = data.track_selection?.visual_vibe?.visual_mood ?? "n/a";
      const targetTempo = data.track_selection?.target_tempo ?? "n/a";
      const tracksConsidered = data.track_selection?.tracks_considered
        ? `${data.track_selection.tracks_analyzed}/${data.track_selection.tracks_considered}`
        : "n/a";
      const proxyMethod = data.track_selection?.proxy_evaluation?.method ?? "n/a";
      const timingBreakdown = data.timings_sec
        ? `select ${data.timings_sec.track_select}s | proxy ${data.timings_sec.proxy_rank ?? 0}s | beats ${data.timings_sec.beat_detect}s | render ${data.timings_sec.render_encode}s`
        : "n/a";

      return [
        `Mood: ${data.mood}`,
        `Selected audio: ${data.selected_audio || data.selected_phonk || "none"}`,
        `BPM (librosa): ${data.detected_bpm}`,
        `Vibe recheck: ${visualMood} | target tempo ${targetTempo}`,
        `Track search: ${tracksConsidered} (analyzed/available) | method ${proxyMethod}`,
        `Scene intelligence: ${data.scene_intelligence ? "on" : "off"}`,
        `Edit brain model: ${data.edit_brain_loaded ? "on" : "off"}`,
        `Encoder: ${data.encoder_codec || "libx264"}`,
        `Style intensity: ${data.style_profile?.intensity ?? "n/a"}`,
        `Active effects: ${activeEffects || "none"}`,
        `Beat markers: ${data.beat_markers.length}`,
        `Planned actions: ${data.edit_actions.length}`,
        `Render length: ${formatDuration(data.render_duration_sec)}`,
        `Processing time: ${data.processing_time_sec == null ? "n/a" : formatDuration(data.processing_time_sec)}`,
        `Timing breakdown: ${timingBreakdown}`,
        `Labels dataset: ${data.labels_csv || "n/a"}`,
      ].join("\n");
    };

    const handleRender = async () => {
      if (!session && supabase) {
        alert("Please log in first.");
        return;
      }
      if (!prompt.trim()) {
        alert("Describe your edit mood first.");
        return;
      }
      if (!file) {
        alert("Upload a video first.");
        return;
      }
      if (isRendering) return;

      const form = new FormData();
      form.append("prompt", prompt.trim());
      form.append("video", file);
      
      // Generate a render_id so we can cancel it if needed
      const renderId = `render_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      form.append("render_id", renderId);
      renderIdRef.current = renderId;

      setStatus("ARCHIVE backend: analyzing audio + rendering with PyAV/FFmpeg...");
      setIsRendering(true);
      setRenderUrl("");
      setDownloadUrl("");
      abortRef.current = new AbortController();

      try {
        const res = await fetch(`${API_BASE}/api/render`, {
          method: "POST",
          body: form,
          signal: abortRef.current.signal,
        });
        let data = null;
        let rawText = "";
        try {
          data = await res.json();
        } catch (_jsonErr) {
          try {
            rawText = await res.text();
          } catch (_textErr) {
            rawText = "";
          }
        }
        if (!res.ok) {
          const serverError = (data && data.error) || rawText || `Request failed with status ${res.status}`;
          throw new Error(serverError);
        }
        if (!data) {
          throw new Error("Server returned empty response.");
        }

        // Store render_id for potential cancellation
        if (data.render_id) {
          renderIdRef.current = data.render_id;
        }

        setStatus(buildSummary(data));
        const outputUrl = `${API_BASE}${data.render_url}`;
        const dlUrl = data.download_url ? `${API_BASE}${data.download_url}` : outputUrl;
        setRenderUrl(outputUrl);
        setDownloadUrl(dlUrl);
      } catch (err) {
        if (err && err.name === "AbortError") {
          setStatus("ARCHIVE render cancelled.");
        } else {
          setStatus(`ARCHIVE backend error: ${err.message}`);
        }
      } finally {
        setIsRendering(false);
        abortRef.current = null;
      }
    };

    return h(
      React.Fragment,
      null,
      null,
      showStars ? h(StarField) : null,
      h(Guide),
      h(
        "main",
        null,
        h("div", { className: "editor-box" }, [
          h("textarea", {
            placeholder: "Describe how you want your video to be edited...",
            value: prompt,
            onChange: (e) => setPrompt(e.target.value),
            key: "ta",
          }),
          h("div", { className: "upload-section", key: "up" }, [
            h("input", {
              type: "file",
              id: "videoInput",
              accept: "video/*",
              hidden: true,
              ref: fileInputRef,
              onChange: handleFileChange,
            }),
            h(
              "button",
              { id: "uploadBtn", type: "button", onClick: handleUploadClick, disabled: isRendering },
              buttonLabel
            ),
          ]),
          h(VideoPreview, {
            key: "vp",
            src: renderUrl || fileUrl,
            type: renderUrl ? "video/mp4" : file ? file.type : undefined,
            downloadUrl: renderUrl ? downloadUrl : "",
            onDownload: handleDownload,
          }),
          h(StatusBox, { key: "sb", text: status }),
        ])
      ),
      h(Footer),
      h(LoadingOverlay, { showing: isRendering, elapsed, onCancel: cancelRender })
    );
  }

  const root = document.getElementById("root");
  ReactDOM.createRoot(root).render(h(App));
})();
