import React, { useEffect, useRef, useState } from "./react-shim.js";
import StarField from "./StarField.js";
import Guide from "./Guide.js";
import Footer from "./Footer.js";
import StatusBox from "./StatusBox.js";
import LoadingOverlay from "./LoadingOverlay.js";
import VideoPreview from "./VideoPreview.js";

const API_BASE = window.ARCHIVE_API_BASE || "http://127.0.0.1:5000";

const formatDuration = (secondsRaw) => {
  const total = Math.max(0, Math.floor(Number(secondsRaw) || 0));
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}m ${s}s`;
};

const estimateRenderWindow = (promptRaw, meta) => {
  const prompt = String(promptRaw || "").toLowerCase();
  const duration = Math.max(1, Math.min(45, Number(meta?.duration) || 20));
  const width = Number(meta?.width) || 1280;
  const height = Number(meta?.height) || 720;
  const pixels = Math.max(1, width * height);

  // For a potato PC, 1 second of source video = ~20-60 seconds of rendering time
  let multiplier = 20;
  let overhead = 60;

  if (/\bfast render|quick render|minimum time|low time|faster\b/.test(prompt)) multiplier *= 0.5;
  if (/\bpro\b|\bpro pack\b|\ball effects\b|\beverything\b|\bmax effects\b/.test(prompt)) multiplier *= 1.8;
  if (/\bbrain\b|\bcinematic\b|\bdramatic\b|\bintensity 3\b|\bintense\b/.test(prompt)) multiplier *= 2.5;
  if (/\bclean\b|\bminimal\b|\bsimple\b/.test(prompt)) multiplier *= 0.7;

  if (pixels >= 3840 * 2160) multiplier *= 2.5;
  else if (pixels >= 2560 * 1440) multiplier *= 1.6;
  else if (pixels >= 1920 * 1080) multiplier *= 1.2;

  const expected = Math.max(60, Math.round(duration * multiplier + overhead));
  const min = Math.max(45, Math.round(expected * 0.75));
  const max = Math.max(min + 60, Math.round(expected * 1.5));
  return { expected, min, max };
};

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState("");
  const [renderUrl, setRenderUrl] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [status, setStatus] = useState("ARCHIVE backend: idle");
  const [isRendering, setIsRendering] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [videoMeta, setVideoMeta] = useState({ duration: 0, width: 0, height: 0 });
  const [etaWindow, setEtaWindow] = useState(null);
  const [backendProgress, setBackendProgress] = useState(null);

  const fileInputRef = useRef(null);
  const timerRef = useRef(null);
  const abortRef = useRef(null);
  const downloadBusyRef = useRef(false);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  const startStatusPolling = (renderId) => {
    if (!renderId) return;
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/render-status/${encodeURIComponent(renderId)}`);
        if (!res.ok) return;
        const data = await res.json();
        setBackendProgress({
          renderId: data.render_id || renderId,
          stage: data.stage || "rendering",
          message: data.message || "Rendering...",
          progressPct: Number(data.progress_pct) || 0,
          done: Boolean(data.done),
          ok: data.ok,
        });
      } catch (_err) {
        // Silence polling errors; the final /api/render response handles hard failures.
      }
    };
    void poll();
    pollRef.current = setInterval(() => {
      void poll();
    }, 1500);
  };

  const stopStatusPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    try {
      localStorage.removeItem("archive_last_render_url");
      localStorage.removeItem("archive_last_render_summary");
    } catch (err) {
      console.warn("Could not clear old render cache", err);
    }
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

    const probe = document.createElement("video");
    probe.preload = "metadata";
    probe.onloadedmetadata = () => {
      const duration = Number(probe.duration) || 0;
      const width = Number(probe.videoWidth) || 0;
      const height = Number(probe.videoHeight) || 0;
      setVideoMeta({ duration, width, height });
      probe.src = "";
    };
    probe.onerror = () => {
      setVideoMeta({ duration: 0, width: 0, height: 0 });
      probe.src = "";
    };
    probe.src = url;
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
    if (abortRef.current) {
      abortRef.current.abort();
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
    const renderId =
      (window.crypto && typeof window.crypto.randomUUID === "function" && window.crypto.randomUUID()) ||
      `${Date.now()}_${Math.random().toString(36).slice(2)}`;
    form.append("prompt", prompt.trim());
    form.append("video", file);
    form.append("render_id", renderId);

    setStatus("ARCHIVE backend: analyzing audio + rendering with librosa/moviepy...");
    setIsRendering(true);
    setEtaWindow(estimateRenderWindow(prompt.trim(), videoMeta));
    setBackendProgress({
      renderId,
      stage: "accepted",
      message: "Upload received. Waiting for backend pipeline...",
      progressPct: 1,
      done: false,
      ok: null,
    });
    setRenderUrl("");
    setDownloadUrl("");
    abortRef.current = new AbortController();
    startStatusPolling(renderId);

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
      stopStatusPolling();
      setIsRendering(false);
      setEtaWindow(null);
      abortRef.current = null;
    }
  };

  return (
    <>
      <StarField />
      <Guide />
      <main>
        <div className="editor-box">
          <textarea
            placeholder="Describe how you want your video to be edited..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          <div className="upload-section">
            <input
              type="file"
              id="videoInput"
              accept="video/*"
              hidden
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            <button id="uploadBtn" type="button" onClick={handleUploadClick} disabled={isRendering}>
              {buttonLabel}
            </button>
          </div>

          <VideoPreview
            src={renderUrl || fileUrl}
            type={renderUrl ? "video/mp4" : file ? file.type : undefined}
            downloadUrl={renderUrl ? downloadUrl : ""}
            onDownload={handleDownload}
          />

          <StatusBox text={status} />
        </div>
      </main>
      <Footer />
      <LoadingOverlay
        showing={isRendering}
        elapsed={elapsed}
        etaWindow={etaWindow}
        backendProgress={backendProgress}
        onCancel={cancelRender}
        formatDuration={formatDuration}
      />
    </>
  );
}
