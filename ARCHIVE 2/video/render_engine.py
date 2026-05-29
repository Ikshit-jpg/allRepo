from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .cache import prune_files


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""


def run_subprocess(
    cmd: list[str],
    *,
    timeout: int,
    capture_output: bool = False,
    text: bool = True,
) -> CommandResult:
    """
    Run a subprocess safely with thread-based pipe draining.

    PIPE DEADLOCK FIX: When capture_output=True, subprocess.run blocks if the
    child writes >64KB to stderr/stdout. Instead we drain pipes in background
    threads so neither side blocks.
    """
    if not capture_output:
        # No output capture needed — simple run
        try:
            run = subprocess.run(
                cmd, capture_output=False, text=text,
                timeout=timeout, check=False,
            )
            return CommandResult(
                ok=run.returncode == 0,
                returncode=int(run.returncode),
            )
        except Exception as exc:
            return CommandResult(ok=False, returncode=-1, stderr=str(exc))

    # Thread-based pipe draining — prevents deadlock
    import queue
    import threading

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _drain(pipe, lines):
        try:
            for line in pipe:
                if line:
                    lines.append(line.rstrip())
                    if len(lines) > 200:
                        del lines[:100]
        except (ValueError, OSError):
            pass

    proc = None
    try:
        popen_kwargs = {}
        import platform
        if platform.system() == 'Windows':
            popen_kwargs['creationflags'] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=text, encoding='utf-8' if text else None,
            errors='replace' if text else None,
            **popen_kwargs,
        )

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        # Wait for process with timeout
        import time
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if time.monotonic() > deadline:
                # Kill process tree on timeout
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(
                            ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                            capture_output=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                    else:
                        proc.kill()
                except Exception:
                    proc.kill()
                return CommandResult(
                    ok=False, returncode=-1,
                    stdout='\n'.join(stdout_lines),
                    stderr='\n'.join(stderr_lines) + f'\n[TIMEOUT] after {timeout}s',
                )
            time.sleep(0.05)

        t_out.join(timeout=3)
        t_err.join(timeout=3)

        return CommandResult(
            ok=proc.returncode == 0,
            returncode=int(proc.returncode),
            stdout='\n'.join(stdout_lines),
            stderr='\n'.join(stderr_lines),
        )
    except Exception as exc:
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        return CommandResult(ok=False, returncode=-1, stderr=str(exc))



def ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or ""


def ffmpeg_hwaccel_input_args() -> list[str]:
    mode = os.getenv("ARCHIVE_FFMPEG_HWACCEL", "auto").strip().lower()
    if mode in {"0", "false", "off", "none"}:
        return []
    if mode in {"cuda", "nvdec"}:
        return ["-hwaccel", "cuda"]
    if mode in {"qsv", "d3d11va", "dxva2"}:
        return ["-hwaccel", mode]
    return ["-hwaccel", "auto"]


def prune_runtime_temp(runtime_dir: Path, *, max_age_seconds: int = 3600) -> int:
    removed = 0
    removed += prune_files(runtime_dir / "uploads", "temp_audio*.m4a", max_files=24, max_age_seconds=max_age_seconds)
    removed += prune_files(runtime_dir / "uploads", "audio_*.wav", max_files=24, max_age_seconds=max_age_seconds)
    removed += prune_files(runtime_dir / "outputs", "*_mpy.mp4", max_files=12, max_age_seconds=max_age_seconds)
    return removed
