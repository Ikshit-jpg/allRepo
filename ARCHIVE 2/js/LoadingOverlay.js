export default function LoadingOverlay({ showing, elapsed, etaWindow, backendProgress, onCancel, formatDuration }) {
  if (!showing) return null;
  const expected = Number(etaWindow?.expected) || 0;
  const min = Number(etaWindow?.min) || 0;
  const max = Number(etaWindow?.max) || 0;
  const remaining = expected > 0 ? Math.max(0, expected - elapsed) : 0;
  const overEstimate = expected > 0 && elapsed > expected;
  const progressPct = Math.max(0, Math.min(100, Number(backendProgress?.progressPct) || 0));
  const stageLabel = backendProgress?.stage ? String(backendProgress.stage).replace(/_/g, " ") : "";
  const stageMessage = backendProgress?.message || "";

  return (
    <div className="archive-loading-overlay">
      <div className="archive-loading-box">
        <div className="archive-loading-spinner" />
        <div className="archive-loading-title">Editing your video...</div>
        {stageLabel ? <div className="archive-loading-time">Stage: {stageLabel}</div> : null}
        {stageMessage ? <div className="archive-loading-time">{stageMessage}</div> : null}
        {progressPct > 0 ? <div className="archive-loading-time">Backend progress: {progressPct.toFixed(0)}%</div> : null}
        <div className="archive-loading-time">Elapsed: {formatDuration(elapsed)}</div>
        {expected > 0 ? (
          <>
            <div className="archive-loading-time">
              Estimated Total: ~{formatDuration(expected)} (range {formatDuration(min)} - {formatDuration(max)})
            </div>
            <div className="archive-loading-time">
              {overEstimate ? "Still processing: past estimate, likely in heavy render path." : `Estimated Remaining: ~${formatDuration(remaining)}`}
            </div>
          </>
        ) : null}
        <button type="button" className="archive-cancel-btn" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
