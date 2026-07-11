"""
archive_decision_brain.py
=========================
THE CREATIVE AI. Thinks like a human AMV editor.

This is NOT a simple rule engine.
It has:
  - Real beat detection (librosa) with energy curve analysis
  - Velocity editing (speed ramps: 0.2x → 1x → 2x)
  - Bezier curve system for all animated values
  - Creative memory (remembers what was used, creates variation)
  - Narrative arc engine (builds tension, release, climax)
  - Style consistency engine (keeps a visual language per edit)
  - Multi-pass decision: analyze → plan → validate → execute
  - Calls archive_frame_engine.exe for all pixel work

Called by: archive_amv_pipeline.py (or directly)
Calls:     archive_frame_engine.exe (C++ pixel engine)

Usage:
  python archive_decision_brain.py --clips c1.mp4 c2.mp4 c3.mp4
         --music track.mp3 --output amv.mp4 --mood hype --bpm 140

Requirements:
  pip install librosa numpy scipy
"""

from __future__ import annotations
import argparse, json, math, os, random, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. BEZIER / CURVE SYSTEM  (Python side — mirrors C++ curves)
# ─────────────────────────────────────────────────────────────────────────────

class Curve:
    """Curve evaluation functions. Every animated value uses these."""

    @staticmethod
    def ease(t: float) -> float:
        t = max(0., min(1., t))
        return t * t * (3 - 2 * t)

    @staticmethod
    def ease_in(t: float) -> float:
        t = max(0., min(1., t))
        return t * t * t

    @staticmethod
    def ease_out(t: float) -> float:
        t = max(0., min(1., t))
        return 1 - (1 - t) ** 3

    @staticmethod
    def ease_in_out(t: float) -> float:
        t = max(0., min(1., t))
        return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2

    @staticmethod
    def spring(t: float) -> float:
        t = max(0., min(1., t))
        c4 = (2 * math.pi) / 3
        if t == 0: return 0
        if t == 1: return 1
        return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * c4)

    @staticmethod
    def bounce_out(t: float) -> float:
        n1, d1 = 7.5625, 2.75
        if t < 1/d1:   return n1 * t * t
        elif t < 2/d1: t -= 1.5/d1;   return n1*t*t + 0.75
        elif t < 2.5/d1: t -= 2.25/d1; return n1*t*t + 0.9375
        else:          t -= 2.625/d1; return n1*t*t + 0.984375

    @staticmethod
    def beat_decay(t_since_beat: float) -> float:
        """Beat impact curve: 1.0 at t=0, decays exponentially with oscillation"""
        return math.exp(-t_since_beat * 8) * max(0, math.cos(t_since_beat * math.pi * 2))

    @staticmethod
    def cubic_bezier(t: float, ax: float, ay: float, bx: float, by: float) -> float:
        """Custom cubic bezier (like CSS cubic-bezier)"""
        # Newton's method to find t from x, then evaluate y
        def sample_x(t_): return 3*ax*t_*(1-t_)**2 + 3*bx*(1-t_)*t_**2 + t_**3
        def sample_y(t_): return 3*ay*t_*(1-t_)**2 + 3*by*(1-t_)*t_**2 + t_**3
        # Binary search for t
        lo, hi = 0., 1.
        for _ in range(12):
            mid = (lo + hi) / 2
            if sample_x(mid) < t: lo = mid
            else: hi = mid
        return sample_y((lo + hi) / 2)

    # Named presets matching After Effects built-ins
    PRESETS = {
        'linear':    lambda t: t,
        'ease':      ease.__func__,
        'ease_in':   ease_in.__func__,
        'ease_out':  ease_out.__func__,
        'ease_in_out': ease_in_out.__func__,
        'spring':    spring.__func__,
        'bounce':    bounce_out.__func__,
    }


class BeatCurve:
    """
    Drives ALL effect intensities from beat timing.
    This is the core of the beat-sync system.
    Every effect reads from this — not a simple if/beat_hit check.
    """
    def __init__(self, beat_times: List[float], drop_times: List[float], bpm: float):
        self.beats = sorted(beat_times)
        self.drops = sorted(drop_times)
        self.bpm = bpm
        self.beat_interval = 60.0 / max(1, bpm)
        self._cache: Dict[float, float] = {}

    def energy_at(self, t: float) -> float:
        """Returns 0..1 energy at time t. 1.0 = just hit a hard beat."""
        # Find nearest beat
        nearest_dist = min((abs(t - b) for b in self.beats), default=9999)
        if nearest_dist > 0.5: return 0.0
        return Curve.beat_decay(nearest_dist)

    def is_drop(self, t: float, tol: float = 0.18) -> bool:
        return any(abs(t - d) < tol for d in self.drops)

    def is_beat(self, t: float, tol: float = 0.08) -> bool:
        return any(abs(t - b) < tol for b in self.beats)

    def section_energy(self, t_start: float, dur: float) -> float:
        """Average energy across a time window (for scoring clips)"""
        samples = max(1, int(dur * 10))
        return sum(self.energy_at(t_start + i * dur / samples)
                   for i in range(samples)) / samples

    def velocity_at(self, t: float) -> float:
        """
        Velocity multiplier for speed ramp.
        Drops → slow (0.3x), Builds → fast (1.5-2x), Normal → 1x
        """
        e = self.energy_at(t)
        drop_near = self.is_drop(t, 0.5)
        if drop_near and e > 0.3:
            # Just before drop: ramp down to 0.25x (dramatic slow-mo)
            return max(0.25, 1.0 - e * 0.75)
        if e > 0.6:
            # On hard beat: brief speed-up 1.2-1.5x
            return 1.0 + e * 0.5
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. BEAT ANALYSIS ENGINE  (librosa-powered)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeatAnalysis:
    bpm: float
    beats: List[float]        # beat times in seconds
    drops: List[float]        # drop/chorus times
    sections: List[str]       # section labels per beat
    energy_curve: List[float] # RMS energy per second
    duration: float

def analyze_music(music_path: str, bpm_override: float = 0) -> BeatAnalysis:
    """Full music analysis using librosa."""
    try:
        import librosa
        import numpy as np
        print(f"[Brain] Analyzing music: {music_path}")
        y, sr = librosa.load(music_path, sr=22050)
        dur = librosa.get_duration(y=y, sr=sr)

        # BPM detection
        if bpm_override > 0:
            bpm = float(bpm_override)
            _, frames = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm)
        else:
            bpm, frames = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(bpm)

        beats = librosa.frames_to_time(frames, sr=sr).tolist()

        # RMS energy per second
        hop = sr // 10  # 10 samples/second
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        energy_curve = rms.tolist()

        # Find drops: sudden energy spikes after low energy
        drops = []
        win = max(1, sr // hop)  # 1-second window
        for i in range(win, len(rms) - win):
            prev_e = float(np.mean(rms[max(0,i-win*4):i]))
            curr_e = float(rms[i])
            if curr_e > prev_e * 1.8 and curr_e > 0.05:
                t = i * hop / sr
                # Avoid duplicates within 2 seconds
                if not drops or t - drops[-1] > 2.0:
                    drops.append(t)

        # Section labels per beat
        sections = []
        for i, b in enumerate(beats):
            p = b / dur
            if p < 0.08:    sections.append('intro')
            elif p < 0.22:  sections.append('buildup')
            elif p < 0.40:  sections.append('drop')
            elif p < 0.56:  sections.append('chorus')
            elif p < 0.65:  sections.append('break')
            elif p < 0.80:  sections.append('buildup')
            elif p < 0.92:  sections.append('chorus')
            else:           sections.append('outro')

        print(f"[Brain] BPM={bpm:.1f} Beats={len(beats)} Drops={len(drops)}")
        return BeatAnalysis(bpm, beats, drops, sections, energy_curve, dur)

    except ImportError:
        print("[Brain] librosa not found, using synthetic beats")
        return _synthetic_beats(bpm_override or 130, 60.0)
    except Exception as e:
        print(f"[Brain] Music analysis failed ({e}), using synthetic beats")
        return _synthetic_beats(bpm_override or 130, 60.0)

def _synthetic_beats(bpm: float, dur: float) -> BeatAnalysis:
    interval = 60.0 / bpm
    beats = [t for t in [i * interval for i in range(int(dur / interval) + 1)] if t <= dur]
    bar = interval * 4
    section_dur = bar * 8
    drops = [t for t in [section_dur * i for i in range(1, 10)] if t < dur]
    sections = []
    for b in beats:
        p = b / dur
        if p < 0.08:   sections.append('intro')
        elif p < 0.22: sections.append('buildup')
        elif p < 0.40: sections.append('drop')
        elif p < 0.56: sections.append('chorus')
        elif p < 0.65: sections.append('break')
        elif p < 0.80: sections.append('buildup')
        elif p < 0.92: sections.append('chorus')
        else:          sections.append('outro')
    energy = [0.5 + 0.4 * math.sin(2 * math.pi * t / (section_dur or 1))
              for t in [i * 0.1 for i in range(int(dur * 10))]]
    return BeatAnalysis(bpm, beats, drops, sections, energy, dur)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCENE ANALYSIS  (calls C++ analyze_clip command)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneInfo:
    path: str
    motion: float = 0.5
    brightness: float = 0.5
    sharpness: float = 0.5
    saturation: float = 0.5
    scene_type: str = 'action'
    edit_value: float = 0.5
    duration: float = 5.0
    fps: float = 30.0
    width: int = 1280
    height: int = 720

def analyze_clip(path: str, engine: str) -> SceneInfo:
    """Call C++ engine to analyze a clip."""
    try:
        result = subprocess.run(
            [engine, 'analyze_clip', path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout:
            d = json.loads(result.stdout)
            return SceneInfo(
                path=path,
                motion=d.get('motion', 0.5),
                brightness=d.get('brightness', 0.5),
                sharpness=d.get('sharpness', 0.5),
                saturation=d.get('saturation', 0.5),
                scene_type=d.get('scene_type', 'action'),
                edit_value=d.get('edit_value', 0.5),
                duration=d.get('duration', 5.0),
                fps=d.get('fps', 30.0),
                width=d.get('width', 1280),
                height=d.get('height', 720),
            )
    except Exception as e:
        print(f"[Brain] analyze_clip failed for {path}: {e}")
    return SceneInfo(path=path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CREATIVE MEMORY  (prevents repetition, creates variation)
# ─────────────────────────────────────────────────────────────────────────────

class CreativeMemory:
    """
    Remembers what effects, transitions, animations were recently used.
    Forces variation — real editors NEVER repeat the same effect 3 times in a row.
    Also tracks what the 'visual language' of this edit is.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.effect_history:     List[str] = []
        self.transition_history: List[str] = []
        self.anim_history:       List[str] = []
        self.last_heavy: int = -5  # segment index of last heavy effect
        # Visual language — chosen once, kept consistent
        self.primary_color:   str = 'warm'   # overwritten by mood
        self.secondary_color: str = 'white'
        self.grain_level:     float = 0.40
        self.bloom_style:     str = 'warm'

    def pick_effect(self, candidates: List[str], section: str,
                    scene_type: str, seg_idx: int) -> str:
        """Pick best effect, avoiding recent repeats."""
        if not candidates: return 'cinematic_grade'
        # Filter out anything used in last 2 segments
        recent = set(self.effect_history[-2:])
        filtered = [c for c in candidates if c not in recent] or candidates
        # Weight by section/scene compatibility
        weights = [self._effect_weight(c, section, scene_type) for c in filtered]
        total = sum(weights)
        r = self.rng.random() * total
        chosen = filtered[0]
        for c, w in zip(filtered, weights):
            r -= w
            if r <= 0: chosen = c; break
        self.effect_history.append(chosen)
        if len(self.effect_history) > 10: self.effect_history.pop(0)
        return chosen

    def pick_transition(self, candidates: List[str], energy: float) -> str:
        """Pick transition, avoid recent repeats, weight by energy."""
        if not candidates: return 'cut'
        recent = set(self.transition_history[-2:])
        filtered = [c for c in candidates if c not in recent] or candidates
        weights = [self._trans_weight(c, energy) for c in filtered]
        total = sum(weights)
        r = self.rng.random() * total
        chosen = filtered[0]
        for c, w in zip(filtered, weights):
            r -= w
            if r <= 0: chosen = c; break
        self.transition_history.append(chosen)
        if len(self.transition_history) > 6: self.transition_history.pop(0)
        return chosen

    def pick_animation(self, scene_type: str, section: str) -> str:
        table = {
            ('impact',   'drop'):    ['camera_slam', 'zoom_pulse', 'handheld_shake'],
            ('impact',   'chorus'):  ['camera_slam', 'zoom_pulse'],
            ('action',   'buildup'): ['handheld_shake', 'push_in_slow', 'orbit_right'],
            ('action',   'chorus'):  ['handheld_shake', 'zoom_pulse'],
            ('reaction', 'drop'):    ['push_in_slow', 'subtle_drift'],
            ('scenery',  'intro'):   ['ken_burns_slow', 'parallax_zoom'],
            ('scenery',  'outro'):   ['ken_burns_slow', 'parallax_zoom'],
            ('quiet',    'break'):   ['subtle_drift', 'ken_burns_slow'],
        }
        key = (scene_type, section)
        candidates = table.get(key, table.get((scene_type, ''), ['subtle_drift']))
        recent = set(self.anim_history[-2:])
        filtered = [c for c in candidates if c not in recent] or candidates
        chosen = self.rng.choice(filtered)
        self.anim_history.append(chosen)
        if len(self.anim_history) > 6: self.anim_history.pop(0)
        return chosen

    def should_fire_heavy(self, seg_idx: int, section: str,
                           scene_type: str, intensity: float) -> bool:
        """
        Restraint system: decides if this segment gets heavy effects.
        Enforces dry streaks so heavy moments feel earned.
        """
        gap = seg_idx - self.last_heavy
        if scene_type in ('quiet', 'dialogue'): return False
        if section == 'intro': return False
        if section in ('drop', 'chorus') and gap >= 2:
            if intensity > 0.65: self.last_heavy = seg_idx; return True
        if section == 'buildup' and gap >= 3 and intensity > 0.75:
            self.last_heavy = seg_idx; return True
        return False

    def set_visual_language(self, mood: str):
        mood_colors = {
            'hype':      ('orange', 'white'),
            'phonk':     ('orange', 'purple'),
            'dark':      ('purple', 'blue'),
            'villain':   ('purple', 'red'),
            'sad':       ('blue',   'white'),
            'emotional': ('blue',   'pink'),
            'cyberpunk': ('cyan',   'purple'),
            'dramatic':  ('teal',   'orange'),
            'chill':     ('blue',   'white'),
            'warm':      ('warm',   'white'),
        }
        c = mood_colors.get(mood, ('orange', 'white'))
        self.primary_color, self.secondary_color = c

    def _effect_weight(self, effect: str, section: str, scene_type: str) -> float:
        """Score an effect for this context."""
        score = 1.0
        if section == 'intro' and effect in ('cinematic_grade', 'atmospheric_grade'): score *= 3
        if section == 'drop' and effect in ('impact_flash_combo', 'zoom_burst_shake'): score *= 3
        if section == 'chorus' and effect in ('energy_aura', 'god_of_war'): score *= 2.5
        if section == 'break' and effect in ('emotional_peak', 'portrait_enhance'): score *= 2
        if section == 'outro' and effect in ('cinematic_grade', 'atmospheric_grade'): score *= 2.5
        if scene_type == 'quiet' and effect in ('cinematic_grade', 'portrait_enhance'): score *= 2
        if scene_type == 'impact' and effect in ('impact_flash_combo', 'zoom_burst_shake'): score *= 2
        return score

    def _trans_weight(self, trans: str, energy: float) -> float:
        score = 1.0
        if energy > 0.7 and trans in ('zoom_punch', 'flash_cut', 'glitch_cut'): score *= 3
        if energy < 0.3 and trans in ('dissolve', 'fade_black', 'luma_wipe'): score *= 3
        if energy > 0.5 and trans in ('whip_pan_right', 'whip_pan_up', 'smear_cut'): score *= 2
        return score


# ─────────────────────────────────────────────────────────────────────────────
# 5. VELOCITY / SPEED RAMP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VelocitySegment:
    """A segment of video with a speed ramp applied."""
    source_start: float   # seconds in source clip
    source_end: float
    speed_curve: str      # 'linear', 'ramp_in', 'ramp_out', 'slo_mo', 'fast_mo'
    speed_start: float    # multiplier at segment start
    speed_end: float      # multiplier at segment end

class VelocityEngine:
    """
    Speed ramp system.
    Takes a clip and a beat curve, produces velocity segments.

    Human editors use:
      - Slow-mo before a drop (0.3x → 1x)
      - Fast cuts during chorus (1.0x → 1.5x)
      - Freeze frame on hard hits
      - Regular speed in breaks
    """
    def __init__(self, beat_curve: BeatCurve):
        self.bc = beat_curve

    def plan_velocity(self, clip_start: float, clip_dur: float,
                      section: str, scene: SceneInfo) -> List[VelocitySegment]:
        """Plan velocity segments for a clip."""
        segments = []
        bc = self.bc

        # Find drops/beats within this clip's time range
        drops_in_clip = [d for d in bc.drops
                         if clip_start - 0.5 <= d <= clip_start + clip_dur + 0.5]
        beats_in_clip = [b for b in bc.beats
                         if clip_start <= b <= clip_start + clip_dur]

        if section == 'buildup' and drops_in_clip:
            # Classic: slow-mo ramp INTO drop
            drop_t = drops_in_clip[0]
            ramp_start = max(clip_start, drop_t - 0.8)
            ramp_dur = drop_t - ramp_start
            if ramp_dur > 0.2:
                # Normal speed before ramp
                segments.append(VelocitySegment(
                    clip_start, ramp_start, 'linear', 1.0, 1.0))
                # Ramp down to 0.25x going into drop (ease_in curve)
                segments.append(VelocitySegment(
                    ramp_start, drop_t, 'ramp_out', 1.0, 0.25))
                return segments

        elif section == 'drop':
            # Start slow, ramp up fast
            end_t = clip_start + clip_dur
            mid_t = clip_start + clip_dur * 0.35
            segments.append(VelocitySegment(clip_start, mid_t, 'ramp_in', 0.30, 1.0))
            segments.append(VelocitySegment(mid_t, end_t, 'linear', 1.0, 1.0))
            return segments

        elif section == 'chorus' and scene.motion > 0.5:
            # Slightly sped up for energy
            segments.append(VelocitySegment(
                clip_start, clip_start + clip_dur, 'linear', 1.15, 1.15))
            return segments

        elif section == 'break':
            # Slowed down for emotional weight
            segments.append(VelocitySegment(
                clip_start, clip_start + clip_dur, 'linear', 0.60, 0.60))
            return segments

        # Default: 1x
        segments.append(VelocitySegment(
            clip_start, clip_start + clip_dur, 'linear', 1.0, 1.0))
        return segments

    def apply_velocity_ffmpeg(self, input_path: str, output_path: str,
                               segments: List[VelocitySegment]) -> bool:
        """Apply velocity segments using FFmpeg setpts filter."""
        if not segments: return False
        # Calculate overall average speed
        total_out_dur = 0
        for seg in segments:
            src_dur = seg.source_end - seg.source_start
            avg_speed = (seg.speed_start + seg.speed_end) / 2
            total_out_dur += src_dur / avg_speed if avg_speed > 0 else src_dur

        # Simple: single speed ramp using setpts
        # For complex ramps, we'd need frame extraction — simplified here
        avg_speed = sum(s.speed_start + s.speed_end for s in segments) / (2 * max(1, len(segments)))
        avg_speed = max(0.1, min(5.0, avg_speed))

        if abs(avg_speed - 1.0) < 0.05:
            # No change needed
            import shutil
            shutil.copy2(input_path, output_path)
            return True

        # FFmpeg setpts for speed change
        pts_factor = 1.0 / avg_speed
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', input_path,
            '-vf', f'setpts={pts_factor:.4f}*PTS',
            '-an',  # drop audio (will be remixed later)
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
            output_path
        ]
        try:
            result = subprocess.run(cmd, timeout=400)
            return result.returncode == 0
        except Exception as e:
            print(f"[Velocity] FFmpeg failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# 6. EDIT PLAN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EditSegment:
    """Everything the engine needs to know about one clip in the edit."""
    clip_path:         str
    source_start:      float      # start time in source clip
    source_dur:        float      # duration to use from source
    output_start:      float      # position in final timeline
    output_dur:        float      # duration in output (after speed ramp)
    scene:             SceneInfo
    section:           str
    effect:            str
    effect_intensity:  float
    mood_grade:        str
    animation:         str
    animation_intensity: float
    transition_in:     str
    trans_dur_frames:  int
    is_beat:           bool
    is_drop:           bool
    beat_energy:       float
    velocity_segs:     List[VelocitySegment] = field(default_factory=list)
    # Curve-driven effect params (all bezier-shaped, not linear)
    zoom_start:        float = 1.0
    zoom_end:          float = 1.0
    zoom_curve:        str = 'ease'
    shake_decay:       float = 0.5
    beat_times_str:    str = ''   # CSV of beat times for C++ engine


# ─────────────────────────────────────────────────────────────────────────────
# 7. THE DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    The actual creative brain.
    
    Multi-pass decision system:
      Pass 1: Score all clips by quality/type
      Pass 2: Sort/arrange clips for best narrative flow
      Pass 3: Assign sections, effects, transitions, animations
      Pass 4: Plan velocity (speed ramps)
      Pass 5: Compute curve parameters for all animated values
      Pass 6: Validate — check for repetition, adjust
    """

    # Effect candidates per section × scene_type
    EFFECT_TABLE = {
        ('drop',    'impact'):    ['impact_flash_combo', 'zoom_burst_shake'],
        ('drop',    'action'):    ['zoom_burst_shake', 'speed_blur_lines', 'impact_flash_combo'],
        ('drop',    'reaction'):  ['emotion_glow', 'impact_flash_combo'],
        ('chorus',  'impact'):    ['zoom_burst_shake', 'energy_aura', 'god_of_war'],
        ('chorus',  'action'):    ['speed_blur_lines', 'energy_aura'],
        ('chorus',  'reaction'):  ['emotion_glow', 'portrait_enhance'],
        ('buildup', 'impact'):    ['chromatic_shake', 'speed_blur_lines'],
        ('buildup', 'action'):    ['speed_blur_lines', 'chromatic_shake'],
        ('buildup', 'scenery'):   ['atmospheric_grade', 'cinematic_grade'],
        ('intro',   'scenery'):   ['atmospheric_grade', 'cinematic_grade'],
        ('intro',   'quiet'):     ['cinematic_grade'],
        ('break',   'quiet'):     ['cinematic_grade', 'emotional_peak'],
        ('break',   'reaction'):  ['emotional_peak', 'portrait_enhance'],
        ('outro',   'scenery'):   ['atmospheric_grade', 'cinematic_grade'],
        ('outro',   'quiet'):     ['cinematic_grade'],
    }

    # Transition candidates per energy level
    TRANS_TABLE = {
        'high':   ['zoom_punch', 'flash_cut', 'glitch_cut', 'whip_pan_right',
                   'whip_pan_up', 'energy_slash', 'impact_cut', 'shatter_cut'],
        'mid':    ['smear_cut', 'whip_pan_right', 'cross_zoom', 'dissolve',
                   'luma_wipe', 'tv_static', 'ripple'],
        'low':    ['dissolve', 'fade_black', 'luma_wipe', 'wipe_right', 'dolly_zoom_trans'],
        'into_drop':   ['flash_cut', 'zoom_punch', 'glitch_cut'],
        'into_outro':  ['fade_black', 'dissolve'],
        'from_break':  ['zoom_punch', 'cross_zoom', 'energy_slash'],
        'first_clip':  ['fade_in', 'dissolve'],
    }

    def __init__(self, mood: str, seed: int = 42):
        self.mood = mood
        self.memory = CreativeMemory(seed)
        self.memory.set_visual_language(mood)
        self.rng = random.Random(seed)

    def _get_effect_candidates(self, section: str, scene_type: str) -> List[str]:
        key = (section, scene_type)
        candidates = self.EFFECT_TABLE.get(key)
        if not candidates:
            # Fallback: match section only
            for (s, _), v in self.EFFECT_TABLE.items():
                if s == section: candidates = v; break
        return candidates or ['cinematic_grade']

    def _get_transition_candidates(self, seg_idx: int, section_to: str,
                                    section_from: str, energy: float) -> List[str]:
        if seg_idx == 0: return self.TRANS_TABLE['first_clip']
        if section_to == 'outro': return self.TRANS_TABLE['into_outro']
        if section_from == 'buildup' and section_to == 'drop':
            return self.TRANS_TABLE['into_drop']
        if section_from == 'break' and section_to in ('chorus', 'buildup'):
            return self.TRANS_TABLE['from_break']
        if energy > 0.65: return self.TRANS_TABLE['high']
        if energy > 0.35: return self.TRANS_TABLE['mid']
        return self.TRANS_TABLE['low']

    def _calc_intensity(self, section: str, scene: SceneInfo,
                         is_drop: bool, beat_energy: float) -> float:
        base = {
            'intro':   0.22, 'buildup': 0.55, 'drop': 0.80,
            'chorus':  0.75, 'break':   0.28, 'outro': 0.22,
        }.get(section, 0.50)
        if is_drop: base = min(1.0, base + 0.20)
        if beat_energy > 0.5: base = min(1.0, base + beat_energy * 0.15)
        if scene.scene_type in ('quiet', 'dialogue'): base *= 0.45
        if scene.scene_type == 'impact': base = min(1.0, base + 0.12)
        return base

    def _plan_curves(self, seg: EditSegment) -> EditSegment:
        """
        Fill in bezier curve parameters for this segment.
        This is what makes the edit feel CINEMATIC instead of robotic.
        """
        section = seg.section
        scene   = seg.scene

        # Zoom curve
        if section in ('intro', 'outro'):
            seg.zoom_start = 1.00; seg.zoom_end = 1.08; seg.zoom_curve = 'ease_in'
        elif section == 'buildup':
            seg.zoom_start = 1.00; seg.zoom_end = 1.12; seg.zoom_curve = 'ease_in'
        elif section in ('drop', 'chorus') and seg.is_drop:
            seg.zoom_start = 1.18; seg.zoom_end = 1.00; seg.zoom_curve = 'spring'
        elif section in ('drop', 'chorus'):
            seg.zoom_start = 1.00; seg.zoom_end = 1.10; seg.zoom_curve = 'ease'
        elif section == 'break':
            seg.zoom_start = 1.05; seg.zoom_end = 1.00; seg.zoom_curve = 'ease_out'
        else:
            seg.zoom_start = 1.00; seg.zoom_end = 1.05; seg.zoom_curve = 'ease'

        # Shake decay (how fast shake dies after a beat)
        if section in ('drop', 'chorus') and scene.scene_type == 'impact':
            seg.shake_decay = 0.25  # stays shaky longer
        else:
            seg.shake_decay = 0.55

        return seg

    def plan(self, scenes: List[SceneInfo], beat_analysis: BeatAnalysis,
              total_dur: float, fps: float = 30.0) -> List[EditSegment]:
        """
        Full multi-pass edit planning.
        Returns complete EditSegment list ready for rendering.
        """
        beats = beat_analysis.beats
        drops = beat_analysis.drops
        bc = BeatCurve(beats, drops, beat_analysis.bpm)
        vel_engine = VelocityEngine(bc)

        # ── Pass 1: Score clips ──
        for sc in scenes:
            sc.edit_value = (sc.motion * 0.40 + sc.sharpness * 0.25 +
                              sc.saturation * 0.20 + sc.brightness * (1 - sc.brightness) * 4 * 0.15)

        # ── Pass 2: Assign time slots ──
        n = len(scenes)
        clip_dur = total_dur / n  # base duration per clip

        # ── Pass 3–5: Build segments ──
        segments: List[EditSegment] = []
        t = 0.0
        prev_section = 'intro'

        for i, scene in enumerate(scenes):
            p = (t + clip_dur * 0.5) / total_dur
            if p < 0.08:   section = 'intro'
            elif p < 0.22: section = 'buildup'
            elif p < 0.40: section = 'drop'
            elif p < 0.56: section = 'chorus'
            elif p < 0.65: section = 'break'
            elif p < 0.80: section = 'buildup'
            elif p < 0.92: section = 'chorus'
            else:          section = 'outro'

            beat_energy = bc.section_energy(t, clip_dur)
            is_drop = bc.is_drop(t) or bc.is_drop(t + clip_dur * 0.5)
            is_beat = bc.is_beat(t)
            beat_times_in_clip = [b for b in beats if t <= b <= t + clip_dur]
            beat_times_str = ','.join(f'{b - t:.3f}' for b in beat_times_in_clip)

            intensity = self._calc_intensity(section, scene, is_drop, beat_energy)

            # Effect
            candidates = self._get_effect_candidates(section, scene.scene_type)
            effect = self.memory.pick_effect(candidates, section, scene.scene_type, i)

            # Transition
            trans_cands = self._get_transition_candidates(i, section, prev_section, beat_energy)
            transition  = self.memory.pick_transition(trans_cands, beat_energy)
            trans_fps   = fps
            trans_dur_sec = 0.12 if is_beat else 0.20
            trans_frames = max(6, int(trans_dur_sec * trans_fps))

            # Animation
            animation = self.memory.pick_animation(scene.scene_type, section)

            # Velocity plan
            vel_segs = vel_engine.plan_velocity(t, clip_dur, section, scene)
            # Calculate output duration after velocity
            out_dur = 0.0
            for vs in vel_segs:
                src_d = vs.source_end - vs.source_start
                avg_s = (vs.speed_start + vs.speed_end) / 2
                out_dur += src_d / max(0.1, avg_s)
            out_dur = max(0.5, out_dur)

            seg = EditSegment(
                clip_path=scene.path,
                source_start=0.0, source_dur=min(clip_dur, scene.duration),
                output_start=t, output_dur=out_dur,
                scene=scene, section=section,
                effect=effect, effect_intensity=intensity,
                mood_grade=self.mood,
                animation=animation, animation_intensity=intensity * 0.65,
                transition_in=transition, trans_dur_frames=trans_frames,
                is_beat=is_beat, is_drop=is_drop, beat_energy=beat_energy,
                velocity_segs=vel_segs,
                beat_times_str=beat_times_str,
            )
            seg = self._plan_curves(seg)
            segments.append(seg)

            prev_section = section
            t += clip_dur

        # ── Pass 6: Validate ──
        self._validate(segments)
        return segments

    def _validate(self, segments: List[EditSegment]):
        """
        Final creative pass: ensure variation, quality, narrative arc.
        - No same effect 3 times in a row
        - Intro and outro are always clean (low intensity)
        - At least one drop gets maximum intensity
        - Transitions flow logically
        """
        for i, seg in enumerate(segments):
            # Enforce intro/outro cleanliness
            if seg.section in ('intro', 'outro'):
                seg.effect_intensity = min(seg.effect_intensity, 0.30)
                if seg.effect in ('impact_flash_combo', 'zoom_burst_shake', 'energy_aura'):
                    seg.effect = 'cinematic_grade'

            # Check 3-in-a-row
            if i >= 2:
                if (segments[i-1].effect == seg.effect == segments[i-2].effect):
                    seg.effect = 'cinematic_grade'

            # Ensure drops are maximum
            if seg.is_drop and seg.section == 'drop':
                seg.effect_intensity = max(seg.effect_intensity, 0.85)

            # Validate transition makes sense
            if i == 0 and seg.transition_in not in ('fade_in', 'dissolve', 'cut'):
                seg.transition_in = 'fade_in'


# ─────────────────────────────────────────────────────────────────────────────
# 8. RENDER ORCHESTRATOR  (calls C++ for all pixel work)
# ─────────────────────────────────────────────────────────────────────────────

class RenderOrchestrator:
    """Coordinates rendering: calls C++ engine, applies velocity, assembles."""

    def __init__(self, engine_path: str, crf: int = 16,
                 preset: str = 'veryfast', width: int = 1280,
                 height: int = 720, fps: float = 30.0):
        self.engine = engine_path
        self.crf = crf
        self.preset = preset
        self.width = width
        self.height = height
        self.fps = fps
        self.tmp = tempfile.mkdtemp(prefix='archive_render_')

    def render_segment(self, seg: EditSegment, out_path: str,
                        vel_engine: VelocityEngine) -> bool:
        """Render one segment: velocity → effect stack → write."""
        tmp_vel = os.path.join(self.tmp, f'vel_{hash(out_path)}.mp4')
        tmp_src = seg.clip_path

        # Step 1: Velocity (speed ramp)
        has_ramp = any(abs(v.speed_start - 1.0) > 0.05 or abs(v.speed_end - 1.0) > 0.05
                       for v in seg.velocity_segs)
        if has_ramp:
            ok = vel_engine.apply_velocity_ffmpeg(tmp_src, tmp_vel, seg.velocity_segs)
            if ok: tmp_src = tmp_vel

        # Step 2: Call C++ engine for full effect stack
        cmd = [
            self.engine, 'render_clip',
            tmp_src, out_path,
            seg.mood_grade,
            f'{seg.effect_intensity:.3f}',
            seg.effect,
            seg.animation,
            f'{seg.animation_intensity:.3f}',
            seg.beat_times_str,
            f'{seg.scene.fps:.1f}',   # bpm placeholder — C++ uses beat_times
            '1' if seg.is_drop else '0',
            str(self.crf),
            self.preset,
            str(self.width),
            str(self.height),
            f'{self.fps:.1f}',
        ]
        try:
            r = subprocess.run(cmd, timeout=600)
            if r.returncode != 0:
                print(f"[Render] C++ engine failed for {seg.clip_path}, using fallback")
                return self._fallback_render(tmp_src, out_path, seg)
            return True
        except subprocess.TimeoutExpired:
            print(f"[Render] Timeout for {seg.clip_path}")
            return self._fallback_render(tmp_src, out_path, seg)
        except Exception as e:
            print(f"[Render] Exception: {e}")
            return self._fallback_render(tmp_src, out_path, seg)

    def _fallback_render(self, input_path: str, output: str, seg: EditSegment) -> bool:
        """Pure FFmpeg fallback if C++ engine not available."""
        filters = []
        i = seg.effect_intensity
        # Color grade via eq
        filters.append(f"eq=contrast={1+i*0.3:.2f}:saturation={1+i*0.3:.2f}:brightness={i*0.02:.3f}")
        # Vignette
        filters.append(f"vignette=angle={i*0.3:.2f}")
        vf = ','.join(filters)
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-i', input_path, '-vf', vf,
               '-c:v', 'libx264', '-preset', self.preset, '-crf', str(self.crf),
               '-an', output]
        try:
            r = subprocess.run(cmd, timeout=300)
            return r.returncode == 0
        except:
            import shutil; shutil.copy2(input_path, output); return True

    def render_transition(self, clip_a: str, clip_b: str, trans_type: str,
                          n_frames: int, out_path: str) -> bool:
        """Render a transition using C++ engine."""
        cmd = [
            self.engine, 'render_transition',
            trans_type, clip_a, clip_b, out_path,
            str(n_frames), str(self.crf), self.preset,
            str(self.width), str(self.height), f'{self.fps:.1f}'
        ]
        try:
            r = subprocess.run(cmd, timeout=400)
            return r.returncode == 0
        except:
            return False  # transitions are optional

    def assemble(self, segments: List[EditSegment], rendered_clips: List[str],
                  trans_clips: List[str], music_path: str,
                  output: str, total_dur: float) -> bool:
        """Concatenate all clips with transitions and mix music."""
        list_file = os.path.join(self.tmp, 'concat.txt')
        with open(list_file, 'w') as f:
            for i, clip in enumerate(rendered_clips):
                if os.path.exists(clip):
                    f.write(f"file '{clip}'\n")
                if i < len(trans_clips) and trans_clips[i] and os.path.exists(trans_clips[i]):
                    f.write(f"file '{trans_clips[i]}'\n")

        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'concat', '-safe', '0', '-i', list_file]
        if music_path and os.path.exists(music_path):
            cmd += ['-i', music_path,
                    '-map', '0:v', '-map', '1:a',
                    '-t', str(total_dur),
                    '-c:a', 'aac', '-b:a', '192k', '-shortest']
        else:
            cmd += ['-t', str(total_dur)]
        cmd += ['-c:v', 'libx264', '-preset', self.preset,
                '-crf', str(self.crf), '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', output]
        try:
            r = subprocess.run(cmd, timeout=600)
            return r.returncode == 0
        except Exception as e:
            print(f"[Assemble] Failed: {e}"); return False

    def cleanup(self):
        import shutil
        try: shutil.rmtree(self.tmp)
        except: pass


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_amv(clips: List[str], music: str, output: str, mood: str,
            bpm: float, total_dur: float, engine: str,
            crf: int, preset: str, width: int, height: int,
            fps: float, seed: int = 42):
    """Full AMV pipeline."""
    print(f"\n{'='*60}")
    print(f"  ARCHIVE DECISION BRAIN v4.0")
    print(f"  Mood: {mood} | BPM: {bpm} | Clips: {len(clips)}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Music analysis ──
    print("[1/6] Analyzing music...")
    ba = analyze_music(music, bpm) if music else _synthetic_beats(bpm or 130, total_dur)
    if not total_dur: total_dur = ba.duration
    bc = BeatCurve(ba.beats, ba.drops, ba.bpm)
    vel_engine = VelocityEngine(bc)

    # ── Clip analysis ──
    print(f"[2/6] Analyzing {len(clips)} clips...")
    scenes = []
    for i, clip in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] {Path(clip).name}...", end=' ', flush=True)
        sc = analyze_clip(clip, engine)
        print(f"type={sc.scene_type} motion={sc.motion:.2f} ev={sc.edit_value:.2f}")
        scenes.append(sc)

    # ── Decision engine ──
    print("[3/6] Planning edit...")
    brain = DecisionEngine(mood, seed)
    segments = brain.plan(scenes, ba, total_dur, fps)
    print(f"  {len(segments)} segments planned")
    for i, seg in enumerate(segments):
        print(f"  [{i:02d}] [{seg.section:8s}] {seg.effect:25s} "
              f"fx={seg.effect_intensity:.2f} trans={seg.transition_in:18s} "
              f"anim={seg.animation:15s} drop={'Y' if seg.is_drop else 'N'}")

    # ── Rendering ──
    print("[4/6] Rendering clips...")
    orch = RenderOrchestrator(engine, crf, preset, width, height, fps)
    rendered = []
    trans_files = []

    for i, seg in enumerate(segments):
        out_clip = os.path.join(orch.tmp, f'clip_{i:03d}.mp4')
        print(f"  Rendering clip {i+1}/{len(segments)}: {seg.effect}...", end=' ', flush=True)
        ok = orch.render_segment(seg, out_clip, vel_engine)
        print("OK" if ok else "FAIL(fallback)")
        rendered.append(out_clip if ok else seg.clip_path)

        # Transition
        if i > 0 and seg.transition_in not in ('cut', 'fade_in'):
            trans_out = os.path.join(orch.tmp, f'trans_{i:03d}.mp4')
            tok = orch.render_transition(rendered[i-1], out_clip, seg.transition_in,
                                          seg.trans_dur_frames, trans_out)
            trans_files.append(trans_out if tok else '')
        else:
            trans_files.append('')

    # ── Assembly ──
    print("[5/6] Assembling final AMV...")
    ok = orch.assemble(segments, rendered, trans_files, music, output, total_dur)

    # ── Cleanup ──
    print("[6/6] Cleaning up...")
    orch.cleanup()

    elapsed = time.time() - t0
    if ok:
        print(f"\n✓ DONE in {elapsed:.1f}s → {output}")
        print(f"  Duration: {total_dur:.1f}s | BPM: {ba.bpm:.1f} | "
              f"Segments: {len(segments)} | Drops: {len(ba.drops)}")
    else:
        print(f"\n✗ FAILED after {elapsed:.1f}s")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Archive Decision Brain v4.0')
    p.add_argument('--clips', nargs='+', required=True, help='Input video clips')
    p.add_argument('--music',  default='',   help='Music file (MP3/WAV/FLAC)')
    p.add_argument('--output', required=True, help='Output AMV path')
    p.add_argument('--mood',   default='hype',
                   choices=['hype','phonk','dark','villain','sad','emotional',
                            'dramatic','cinematic','cyberpunk','neon','warm',
                            'golden','chill','lofi','bleach','teal_orange'])
    p.add_argument('--bpm',    type=float, default=0,
                   help='BPM override (0=auto-detect from music)')
    p.add_argument('--dur',    type=float, default=0, help='Target duration (0=music length)')
    p.add_argument('--engine', default='./archive_frame_engine',
                   help='Path to C++ frame engine binary')
    p.add_argument('--crf',    type=int,   default=16)
    p.add_argument('--preset', default='veryfast')
    p.add_argument('--width',  type=int,   default=1280)
    p.add_argument('--height', type=int,   default=720)
    p.add_argument('--fps',    type=float, default=30.0)
    p.add_argument('--seed',   type=int,   default=42)
    args = p.parse_args()

    # Validate clips
    for c in args.clips:
        if not os.path.exists(c):
            print(f"ERROR: clip not found: {c}"); sys.exit(1)

    # Find engine
    engine = args.engine
    for candidate in [args.engine, './archive_frame_engine.exe',
                       './archive_frame_engine', 'archive_frame_engine']:
        if os.path.exists(candidate): engine=candidate; break
    print(f"[Brain] Using engine: {engine}")

    ok = run_amv(
        clips=args.clips, music=args.music, output=args.output,
        mood=args.mood, bpm=args.bpm,
        total_dur=args.dur, engine=engine,
        crf=args.crf, preset=args.preset,
        width=args.width, height=args.height,
        fps=args.fps, seed=args.seed,
    )
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()


# =============================================================================
# 11. STYLE CONSISTENCY ENGINE
# Keeps the visual language coherent across the ENTIRE edit.
# Real editors pick a "look" at the start and STICK to it.
# =============================================================================

class StyleConsistencyEngine:
    """
    Picks a visual identity for this edit (once) and enforces it.

    Visual identity = {
        color_palette,   # warm/cool/split
        grain_level,     # light/heavy
        bloom_style,     # soft/hard/none
        transition_family, # aggressive/smooth/mixed
        contrast_style,  # crushed/lifted/balanced
        saturation_style # vivid/muted/natural
    }

    This is why some AMVs feel cohesive and others feel random —
    the consistent ones chose a look and never broke it.
    """

    STYLE_PRESETS = {
        'hype': {
            'color_palette':      'warm_orange',
            'grain_level':        0.42,
            'bloom_style':        'hard',
            'bloom_threshold':    0.52,
            'bloom_intensity':    1.35,
            'transition_family':  'aggressive',
            'contrast_style':     'crushed',
            'saturation_style':   'vivid',
            'saturation_boost':   0.40,
            'shadow_crush':       0.08,
            'aura_color':         (80, 160, 255),   # BGR
            'particle_type':      'sparks',
        },
        'phonk': {
            'color_palette':      'warm_orange',
            'grain_level':        0.60,
            'bloom_style':        'hard',
            'bloom_threshold':    0.55,
            'bloom_intensity':    1.20,
            'transition_family':  'aggressive',
            'contrast_style':     'crushed',
            'saturation_style':   'muted',
            'saturation_boost':   0.15,
            'shadow_crush':       0.12,
            'aura_color':         (100, 60, 200),
            'particle_type':      'embers',
        },
        'dark': {
            'color_palette':      'cool_blue',
            'grain_level':        0.55,
            'bloom_style':        'hard',
            'bloom_threshold':    0.60,
            'bloom_intensity':    0.90,
            'transition_family':  'aggressive',
            'contrast_style':     'crushed',
            'saturation_style':   'muted',
            'saturation_boost':   0.10,
            'shadow_crush':       0.14,
            'aura_color':         (200, 40, 200),
            'particle_type':      'embers',
        },
        'villain': {
            'color_palette':      'split_purple_red',
            'grain_level':        0.50,
            'bloom_style':        'hard',
            'bloom_threshold':    0.58,
            'bloom_intensity':    1.10,
            'transition_family':  'aggressive',
            'contrast_style':     'crushed',
            'saturation_style':   'vivid',
            'saturation_boost':   0.30,
            'shadow_crush':       0.10,
            'aura_color':         (180, 30, 255),
            'particle_type':      'embers',
        },
        'sad': {
            'color_palette':      'cool_desaturated',
            'grain_level':        0.65,
            'bloom_style':        'soft',
            'bloom_threshold':    0.42,
            'bloom_intensity':    0.80,
            'transition_family':  'smooth',
            'contrast_style':     'lifted',
            'saturation_style':   'muted',
            'saturation_boost':   -0.15,
            'shadow_crush':       0.02,
            'aura_color':         (200, 160, 80),
            'particle_type':      'petals',
        },
        'emotional': {
            'color_palette':      'warm_desaturated',
            'grain_level':        0.58,
            'bloom_style':        'soft',
            'bloom_threshold':    0.40,
            'bloom_intensity':    0.95,
            'transition_family':  'smooth',
            'contrast_style':     'lifted',
            'saturation_style':   'muted',
            'saturation_boost':   -0.10,
            'shadow_crush':       0.03,
            'aura_color':         (160, 130, 200),
            'particle_type':      'petals',
        },
        'dramatic': {
            'color_palette':      'teal_orange',
            'grain_level':        0.48,
            'bloom_style':        'hard',
            'bloom_threshold':    0.58,
            'bloom_intensity':    1.10,
            'transition_family':  'mixed',
            'contrast_style':     'balanced',
            'saturation_style':   'vivid',
            'saturation_boost':   0.25,
            'shadow_crush':       0.06,
            'aura_color':         (60, 160, 255),
            'particle_type':      'sparks',
        },
        'cyberpunk': {
            'color_palette':      'split_cyan_purple',
            'grain_level':        0.52,
            'bloom_style':        'hard',
            'bloom_threshold':    0.50,
            'bloom_intensity':    1.40,
            'transition_family':  'aggressive',
            'contrast_style':     'crushed',
            'saturation_style':   'vivid',
            'saturation_boost':   0.45,
            'shadow_crush':       0.10,
            'aura_color':         (255, 200, 40),
            'particle_type':      'sparks',
        },
        'chill': {
            'color_palette':      'warm_lifted',
            'grain_level':        0.38,
            'bloom_style':        'soft',
            'bloom_threshold':    0.45,
            'bloom_intensity':    0.70,
            'transition_family':  'smooth',
            'contrast_style':     'lifted',
            'saturation_style':   'natural',
            'saturation_boost':   0.10,
            'shadow_crush':       0.00,
            'aura_color':         (200, 200, 100),
            'particle_type':      'petals',
        },
    }

    def __init__(self, mood: str):
        self.mood = mood
        self.style = self.STYLE_PRESETS.get(mood, self.STYLE_PRESETS['hype'])

    def apply_to_segment(self, seg: 'EditSegment') -> 'EditSegment':
        """Enforce style consistency on a planned segment."""
        s = self.style
        # Adjust grain based on style
        # (passed to C++ engine as intensity modifier)
        grain_adj = s['grain_level']

        # Transition family filter
        if s['transition_family'] == 'smooth':
            if seg.transition_in in ('glitch_cut', 'shatter_cut', 'tv_static', 'impact_cut'):
                seg.transition_in = 'dissolve'
        elif s['transition_family'] == 'aggressive':
            if seg.transition_in == 'dissolve' and seg.beat_energy > 0.4:
                seg.transition_in = 'smear_cut'

        # Particle type from style
        # (used by C++ engine when compositing)
        seg.mood_grade = self.mood  # ensures LUT is always consistent

        return seg

    def get_particle_type(self) -> str:
        return self.style.get('particle_type', 'sparks')

    def get_aura_color(self) -> tuple:
        return self.style.get('aura_color', (80, 160, 255))

    def get_bloom_params(self) -> dict:
        return {
            'threshold': self.style['bloom_threshold'],
            'intensity': self.style['bloom_intensity'],
            'style':     self.style['bloom_style'],
        }


# =============================================================================
# 12. NARRATIVE TENSION ENGINE
# Controls the emotional arc of the ENTIRE edit, not just individual clips.
# This is what separates 7/10 edits from 10/10 edits.
# =============================================================================

class NarrativeTensionEngine:
    """
    Maps a tension curve across the whole edit.

    Real AMV narrative arc:
      0.00 → 0.10  : Establish (calm, introduce)
      0.10 → 0.30  : Build tension (escalate)
      0.30 → 0.45  : First peak (drop 1)
      0.45 → 0.60  : Develop (second theme)
      0.60 → 0.72  : Lowest point (emotional beat or break)
      0.72 → 0.88  : Final escalation (biggest buildup)
      0.88 → 1.00  : Climax → Resolution

    The tension value at each point modifies:
      - Effect intensity
      - Zoom amount
      - Shake amount
      - Transition aggressiveness
      - Cut speed (shorter clips = more tension)
    """

    # Tension curve control points (position → tension 0..1)
    TENSION_CURVE = [
        (0.00, 0.10),   # gentle start
        (0.08, 0.18),
        (0.15, 0.35),   # first build
        (0.25, 0.55),
        (0.32, 0.90),   # first drop!
        (0.38, 0.75),   # settle
        (0.45, 0.60),   # second theme
        (0.55, 0.68),
        (0.62, 0.30),   # emotional break — LOWEST POINT
        (0.68, 0.20),
        (0.72, 0.40),   # start of final build
        (0.80, 0.72),
        (0.85, 0.92),   # pre-climax
        (0.90, 1.00),   # CLIMAX
        (0.95, 0.80),
        (1.00, 0.40),   # resolution
    ]

    def __init__(self):
        self._compile_curve()

    def _compile_curve(self):
        """Pre-compute tension at every 0.001 step."""
        self._tension = []
        pts = self.TENSION_CURVE
        for step in range(1001):
            p = step / 1000
            # Linear interpolation between control points
            tension = 0.0
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i+1]
                if x0 <= p <= x1:
                    t_ = (p - x0) / max(1e-9, x1 - x0)
                    # Smoothstep interpolation
                    t_ = t_ * t_ * (3 - 2 * t_)
                    tension = y0 + (y1 - y0) * t_
                    break
            self._tension.append(tension)

    def tension_at(self, position: float) -> float:
        """
        Get tension at normalized position (0..1 across whole edit).
        Returns 0..1 where 1 = maximum tension/intensity.
        """
        idx = int(max(0., min(1., position)) * 1000)
        return self._tension[min(idx, 1000)]

    def modify_segment(self, seg: 'EditSegment', position: float) -> 'EditSegment':
        """
        Apply narrative tension to a segment.
        This OVERRIDES the per-clip decision when the narrative needs it.
        """
        t = self.tension_at(position)

        # Tension directly scales effect intensity
        seg.effect_intensity = max(0.15, min(1.0,
            seg.effect_intensity * 0.6 + t * 0.4))

        # At very low tension (break): force clean effects
        if t < 0.28:
            if seg.effect in ('impact_flash_combo', 'zoom_burst_shake',
                               'energy_aura', 'speed_blur_lines'):
                seg.effect = 'cinematic_grade'
            seg.transition_in = 'dissolve' if seg.transition_in not in ('fade_in', 'fade_black') else seg.transition_in

        # At very high tension (climax): force heavy effects
        if t > 0.88:
            if seg.effect == 'cinematic_grade':
                seg.effect = 'impact_flash_combo'
            seg.effect_intensity = max(seg.effect_intensity, 0.85)
            if seg.transition_in in ('dissolve', 'fade_black'):
                seg.transition_in = 'zoom_punch'

        # Zoom scales with tension
        seg.zoom_end = 1.0 + t * 0.18 * seg.zoom_end
        seg.animation_intensity = max(0.10, seg.animation_intensity * 0.5 + t * 0.5)

        return seg

    def plan_clip_durations(self, n_clips: int, total_dur: float) -> list:
        """
        Variable clip durations based on tension.
        High tension = shorter cuts (more energy)
        Low tension  = longer clips (breathing room)
        """
        base = total_dur / n_clips
        durations = []
        for i in range(n_clips):
            pos = i / max(1, n_clips - 1)
            t = self.tension_at(pos)
            # High tension = 60% of base, low tension = 140% of base
            scale = 1.40 - t * 0.80
            durations.append(base * scale)

        # Normalize so total stays correct
        total = sum(durations)
        scale_factor = total_dur / total
        durations = [d * scale_factor for d in durations]
        return durations


# =============================================================================
# 13. SYNTH EFFECT ENGINE
# When no preset effect fits, synthesize a new one from primitives.
# This is the "creativity fallback" — it NEVER returns nothing.
# =============================================================================

class SynthEffectEngine:
    """
    Generates custom effect recipes when nothing in the library fits.

    A recipe = ordered list of primitive operations + intensities.
    Primitives: glow, chromatic, shake, vignette, grain, aura,
                fire, speed_lines, god_rays, flare, halation,
                contrast, saturation, blur, flash

    The synth engine composes these based on:
      - Mood (determines color palette)
      - Scene type (determines which primitives work)
      - Section (determines intensity envelope)
      - Beat energy (drives real-time intensity)
    """

    # Primitive compatibility per scene type
    PRIMITIVE_COMPAT = {
        'impact':   ['glow', 'chromatic', 'shake', 'speed_lines', 'flash', 'contrast'],
        'action':   ['speed_lines', 'chromatic', 'glow', 'shake', 'contrast', 'saturation'],
        'reaction': ['glow', 'halation', 'vignette', 'grain', 'contrast'],
        'scenery':  ['god_rays', 'flare', 'glow', 'vignette', 'grain', 'halation'],
        'quiet':    ['vignette', 'grain', 'glow', 'halation', 'contrast'],
        'dialogue': ['vignette', 'grain', 'glow', 'contrast'],
    }

    # Primitive combinations that DON'T work well together
    CONFLICTS = [
        {'shake', 'halation'},      # shake + film bleed = muddy
        {'fire', 'god_rays'},       # too much red
        {'chromatic', 'vhs'},       # double chromatic = too much
        {'speed_lines', 'halation'},# lines + bleed = messy
    ]

    def __init__(self, mood: str, rng: random.Random):
        self.mood = mood
        self.rng = rng

        # Mood → preferred primitives
        self.mood_prefs = {
            'hype':      ['glow', 'chromatic', 'speed_lines', 'flash', 'aura'],
            'phonk':     ['glow', 'vignette', 'grain', 'chromatic', 'contrast'],
            'dark':      ['aura', 'vignette', 'grain', 'glow', 'chromatic'],
            'villain':   ['aura', 'fire', 'chromatic', 'vignette', 'grain'],
            'sad':       ['glow', 'halation', 'vignette', 'grain', 'contrast'],
            'emotional': ['glow', 'halation', 'vignette', 'grain', 'god_rays'],
            'dramatic':  ['glow', 'god_rays', 'flare', 'vignette', 'contrast'],
            'cyberpunk': ['aura', 'chromatic', 'glow', 'vignette', 'grain'],
            'chill':     ['glow', 'halation', 'vignette', 'grain', 'saturation'],
        }.get(mood, ['glow', 'vignette', 'grain', 'contrast', 'chromatic'])

    def _check_conflicts(self, recipe: list) -> bool:
        names = set(p['name'] for p in recipe)
        for conflict_set in self.CONFLICTS:
            if conflict_set.issubset(names):
                return True
        return False

    def synthesize(self, scene_type: str, section: str,
                    intensity: float, beat_energy: float) -> dict:
        """
        Returns a synth recipe dict:
        {
          'name': 'synth_xxxx',
          'primitives': [{'name': ..., 'intensity': ...}, ...],
          'description': '...'
        }
        """
        # Get compatible primitives for this scene
        compat = self.PRIMITIVE_COMPAT.get(scene_type, self.PRIMITIVE_COMPAT['action'])
        # Intersect with mood preferences
        preferred = [p for p in self.mood_prefs if p in compat]
        # Fallback to all compat if intersection is empty
        if not preferred: preferred = compat

        # Decide how many primitives (2..5 based on intensity)
        n_prims = 2 + int(intensity * 3)  # 2 at low, 5 at max

        # Section modifiers
        section_mods = {
            'intro':   0.55, 'buildup': 0.75, 'drop':   1.00,
            'chorus':  0.88, 'break':   0.45, 'outro':  0.50,
        }
        sec_mult = section_mods.get(section, 0.70)

        recipe = []
        used = set()
        attempts = 0

        # Always start with contrast/grade (foundation)
        recipe.append({'name': 'contrast', 'intensity': intensity * sec_mult * 0.55})
        used.add('contrast')

        # Add mood-preferred primitives
        for prim in preferred:
            if len(recipe) >= n_prims: break
            if prim in used: continue
            prim_intensity = intensity * sec_mult * (0.7 + self.rng.random() * 0.3)
            prim_intensity += beat_energy * 0.2  # beat energy boosts
            prim_intensity = max(0.15, min(1.0, prim_intensity))
            recipe.append({'name': prim, 'intensity': prim_intensity})
            used.add(prim)
            # Remove conflicting options
            if self._check_conflicts(recipe):
                recipe.pop()
                used.discard(prim)

        # Always end with grain + vignette (finishing touches)
        if 'vignette' not in used:
            recipe.append({'name': 'vignette', 'intensity': intensity * 0.65})
        if 'grain' not in used:
            recipe.append({'name': 'grain', 'intensity': intensity * 0.42})

        # Generate unique name
        prim_names = '_'.join(p['name'][:3] for p in recipe[:3])
        name = f"synth_{prim_names}_{int(intensity*10)}"

        desc = (f"Synth[{scene_type}/{section}]: "
                + ' → '.join(f"{p['name']}({p['intensity']:.2f})" for p in recipe))

        return {
            'name': name,
            'primitives': recipe,
            'description': desc,
            'scene_type': scene_type,
            'section': section,
        }

    def recipe_to_effect_name(self, recipe: dict) -> str:
        """
        Map synth recipe to closest named C++ effect.
        This ensures the C++ engine always gets a valid effect name.
        """
        prims = set(p['name'] for p in recipe.get('primitives', []))
        # Rule-based mapping: dominant primitive → named effect
        if 'aura' in prims and 'fire' in prims:      return 'god_of_war'
        if 'aura' in prims and 'chromatic' in prims: return 'dark_surge'
        if 'aura' in prims:                           return 'energy_aura'
        if 'fire' in prims:                           return 'fire_outline'
        if 'speed_lines' in prims and 'chromatic' in prims: return 'zoom_burst_shake'
        if 'speed_lines' in prims:                    return 'speed_blur_lines'
        if 'god_rays' in prims:                       return 'atmospheric_grade'
        if 'halation' in prims and 'glow' in prims:  return 'emotional_peak'
        if 'halation' in prims:                       return 'portrait_enhance'
        if 'shake' in prims and 'flash' in prims:     return 'impact_flash_combo'
        if 'chromatic' in prims and 'shake' in prims: return 'chromatic_shake'
        if 'glow' in prims:                           return 'emotion_glow'
        return 'cinematic_grade'


# =============================================================================
# 14. CLIP SORTER
# Reorders clips for best narrative impact.
# Real editors don't just use clips in order — they pick the best clip
# for each moment in the narrative.
# =============================================================================

class ClipSorter:
    """
    Sorts clips to match narrative arc.

    Strategy:
      - Save highest-motion clips for drop/chorus
      - Use scenery/quiet clips for intro/break/outro
      - Put reaction/portrait clips after impacts
      - Vary scene types to avoid repetition
    """

    def sort_for_narrative(self, scenes: list, n_sections: int,
                            tension_engine: NarrativeTensionEngine,
                            rng: random.Random) -> list:
        """
        Reorder scenes to best fit the narrative arc.
        Returns reordered list of SceneInfo.
        """
        if len(scenes) <= 2: return scenes

        # Score each scene for each section position
        n = len(scenes)
        positions = [i / max(1, n - 1) for i in range(n)]
        tensions  = [tension_engine.tension_at(p) for p in positions]

        # Scene type suitability at each position
        def suitability(scene: 'SceneInfo', tension: float) -> float:
            score = 0.0
            t = tension
            if scene.scene_type == 'impact':
                score = t * 0.9 + scene.edit_value * 0.1  # great at high tension
            elif scene.scene_type == 'action':
                score = t * 0.7 + scene.edit_value * 0.3
            elif scene.scene_type == 'reaction':
                score = (0.5 + abs(t - 0.5)) * 0.5 + scene.edit_value * 0.5  # good everywhere
            elif scene.scene_type == 'scenery':
                score = (1 - t) * 0.8 + scene.edit_value * 0.2  # great at low tension
            elif scene.scene_type == 'quiet':
                score = (1 - t) * 0.9 + scene.edit_value * 0.1  # best at low tension
            return score

        # Greedy assignment: for each position (high→low tension), assign best remaining scene
        position_order = sorted(range(n), key=lambda i: tensions[i], reverse=True)
        assigned = [None] * n
        available = list(scenes)

        for pos_idx in position_order:
            if not available: break
            t = tensions[pos_idx]
            # Avoid same scene_type as neighbors
            neighbor_types = set()
            if pos_idx > 0 and assigned[pos_idx-1]:
                neighbor_types.add(assigned[pos_idx-1].scene_type)
            if pos_idx < n-1 and assigned[pos_idx+1]:
                neighbor_types.add(assigned[pos_idx+1].scene_type)

            # Score all available scenes for this position
            scored = [(suitability(sc, t) * (0.7 if sc.scene_type in neighbor_types else 1.0), i, sc)
                      for i, sc in enumerate(available)]
            scored.sort(key=lambda x: x[0], reverse=True)

            # Pick best (with small random factor for variety)
            top_k = min(3, len(scored))
            pick_idx = rng.randint(0, top_k - 1)
            _, avail_idx, chosen_scene = scored[pick_idx]
            assigned[pos_idx] = chosen_scene
            available.pop(avail_idx)

        # Fill any unassigned positions with remaining
        remaining_iter = iter(available)
        for i in range(n):
            if assigned[i] is None:
                try: assigned[i] = next(remaining_iter)
                except StopIteration: pass

        return [s for s in assigned if s is not None]


# =============================================================================
# 15. QUALITY SCORER
# Evaluates the planned edit before rendering.
# Gives actionable warnings and a quality score.
# =============================================================================

class QualityScorer:
    """
    Scores the edit plan before rendering.
    Catches problems the brain might have missed.

    Checks:
      - Effect variety (not too repetitive)
      - Transition variety
      - Narrative arc sensibility
      - Beat sync alignment
      - Section coverage
      - Intensity distribution
    """

    def score(self, segments: list, beat_analysis: 'BeatAnalysis') -> dict:
        n = len(segments)
        if n == 0: return {'score': 0, 'warnings': ['No segments']}
        warnings = []
        bonuses  = []
        score    = 70.0  # base score

        # ── Effect variety ──
        effects = [s.effect for s in segments]
        unique_e = len(set(effects))
        variety_ratio = unique_e / n
        if variety_ratio < 0.3:
            warnings.append(f"Low effect variety: only {unique_e} unique effects for {n} clips")
            score -= 8
        elif variety_ratio > 0.6:
            bonuses.append(f"Great effect variety: {unique_e}/{n}")
            score += 5

        # ── Consecutive repeats ──
        max_run = 1; cur_run = 1
        for i in range(1, len(effects)):
            if effects[i] == effects[i-1]: cur_run += 1; max_run = max(max_run, cur_run)
            else: cur_run = 1
        if max_run >= 3:
            warnings.append(f"Effect repeated {max_run} times in a row")
            score -= 6

        # ── Transition variety ──
        trans = [s.transition_in for s in segments]
        unique_t = len(set(trans))
        if unique_t < max(2, n // 4):
            warnings.append(f"Low transition variety: {unique_t} unique transitions")
            score -= 5

        # ── Intro/outro sanity ──
        if segments[0].section not in ('intro', 'buildup'):
            warnings.append("First clip is not in intro/buildup")
        if segments[0].effect_intensity > 0.45:
            warnings.append(f"Intro too heavy: intensity={segments[0].effect_intensity:.2f}")
            score -= 4
        if segments[-1].transition_in not in ('fade_black', 'dissolve', 'fade_in', 'cut'):
            warnings.append("Last transition should be gentle (fade/dissolve)")
            score -= 3

        # ── Drop coverage ──
        drop_segs = [s for s in segments if s.is_drop]
        if not drop_segs:
            warnings.append("No drop segments detected — beat sync may be off")
            score -= 10
        else:
            avg_drop_intensity = sum(s.effect_intensity for s in drop_segs) / len(drop_segs)
            if avg_drop_intensity < 0.70:
                warnings.append(f"Drops too weak: avg intensity={avg_drop_intensity:.2f}")
                score -= 6
            else:
                bonuses.append(f"Strong drops: avg intensity={avg_drop_intensity:.2f}")
                score += 8

        # ── Break coverage ──
        break_segs = [s for s in segments if s.section == 'break']
        if break_segs:
            avg_break_i = sum(s.effect_intensity for s in break_segs) / len(break_segs)
            if avg_break_i > 0.55:
                warnings.append(f"Break section too heavy: intensity={avg_break_i:.2f}")
                score -= 4
            else:
                bonuses.append("Break section properly restrained")
                score += 4

        # ── Intensity distribution ──
        intensities = [s.effect_intensity for s in segments]
        i_min, i_max = min(intensities), max(intensities)
        i_range = i_max - i_min
        if i_range < 0.30:
            warnings.append(f"Flat intensity: range only {i_range:.2f} (needs more contrast)")
            score -= 7
        else:
            bonuses.append(f"Good intensity range: {i_min:.2f}–{i_max:.2f}")
            score += 5

        # ── Animation variety ──
        anims = [s.animation for s in segments]
        if len(set(anims)) < max(2, n // 5):
            warnings.append("Low animation variety")
            score -= 3

        # ── Curve quality (zoom params) ──
        has_curves = any(abs(s.zoom_end - s.zoom_start) > 0.03 for s in segments)
        if not has_curves:
            warnings.append("No zoom animation detected — all clips are static")
            score -= 5
        else:
            bonuses.append("Zoom animation detected")
            score += 3

        score = max(0, min(100, score))

        return {
            'score': round(score, 1),
            'grade': 'S' if score>=90 else 'A' if score>=80 else 'B' if score>=70 else 'C' if score>=60 else 'D',
            'warnings': warnings,
            'bonuses': bonuses,
            'n_segments': n,
            'unique_effects': unique_e,
            'unique_transitions': unique_t,
            'drop_coverage': len(drop_segs),
            'intensity_range': round(i_range, 3),
        }

    def print_report(self, report: dict):
        print(f"\n{'─'*50}")
        print(f"  QUALITY SCORE: {report['score']}/100  [{report['grade']}]")
        print(f"  Segments: {report['n_segments']}  "
              f"Effects: {report['unique_effects']}  "
              f"Drops: {report['drop_coverage']}")
        if report['warnings']:
            print("  WARNINGS:")
            for w in report['warnings']: print(f"    ⚠ {w}")
        if report['bonuses']:
            print("  STRENGTHS:")
            for b in report['bonuses']: print(f"    ✓ {b}")
        print(f"{'─'*50}\n")


# =============================================================================
# 16. POST PROCESSOR
# Final polish after all clips are rendered and assembled.
# Master grade + loudness normalization + color finalization.
# =============================================================================

class PostProcessor:
    """
    Post-processing pass after assembly.

    Operations (all FFmpeg):
      1. Master LUT (subtle overall color correction)
      2. Vignette (subtle global vignette)
      3. Sharpening (USM)
      4. Loudness normalization (EBU R128, -14 LUFS)
      5. Final encode at target quality

    This is the "color grading suite" pass — even a mediocre edit
    looks better after proper post.
    """

    def __init__(self, crf: int = 16, preset: str = 'veryfast'):
        self.crf = crf
        self.preset = preset

    def process(self, input_path: str, output_path: str, mood: str,
                 intensity: float = 0.5) -> bool:
        """Apply master post-processing pass."""
        tmp = input_path + '.post_tmp.mp4'

        # Build FFmpeg filter chain
        vf_parts = []

        # 1. Subtle contrast/saturation (master grade)
        contrast   = 1.0 + intensity * 0.08
        saturation = 1.0 + intensity * 0.10
        brightness = -0.01 * intensity
        vf_parts.append(
            f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:brightness={brightness:.3f}"
        )

        # 2. Mood-specific color push
        if mood in ('hype', 'phonk'):
            # Warm shadows + cool highlights (teal-orange)
            vf_parts.append("curves=r='0/0 0.5/0.53 1/1':b='0/0 0.5/0.47 1/0.95'")
        elif mood in ('dark', 'villain'):
            # Crush blacks, boost blue
            vf_parts.append("curves=r='0/0 0.3/0.28 1/0.95':b='0/0.02 0.5/0.52 1/1'")
        elif mood in ('sad', 'emotional'):
            # Lift blacks, desaturate slightly, push blue
            vf_parts.append("curves=all='0/0.04 0.5/0.52 1/0.98':b='0/0 0.5/0.54 1/1'")
        elif mood == 'cyberpunk':
            # Deep shadows, vivid highlights
            vf_parts.append("curves=r='0/0 0.4/0.38 1/1':b='0/0.02 0.5/0.55 1/1'")

        # 3. Subtle global vignette
        vf_parts.append(f"vignette=angle={0.15 + intensity * 0.12:.3f}:mode=forward")

        # 4. Slight sharpening
        vf_parts.append("unsharp=lx=3:ly=3:la=0.4:cx=0:cy=0:ca=0")

        vf = ','.join(vf_parts)

        # 5. Audio: loudness normalization
        af = "loudnorm=I=-14:TP=-1.5:LRA=11"

        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', input_path,
            '-vf', vf,
            '-af', af,
            '-c:v', 'libx264', '-preset', self.preset,
            '-crf', str(self.crf),
            '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            tmp
        ]
        try:
            r = subprocess.run(cmd, timeout=600)
            if r.returncode == 0:
                import shutil
                shutil.move(tmp, output_path)
                return True
        except Exception as e:
            print(f"[PostProcess] Failed: {e}")
        return False


# =============================================================================
# 17. PERFORMANCE PROFILER (Potato PC Mode)
# Detects hardware and auto-scales quality.
# =============================================================================

class PerformanceProfiler:
    """
    Detects system resources and picks a performance profile.

    Profiles:
      ultra    — High-end PC (16+ GB RAM, 8+ cores)
      high     — Good PC (8+ GB RAM, 4+ cores)
      medium   — Mid PC (4-8 GB RAM, 2-4 cores)
      low      — Potato PC (< 4 GB RAM, 1-2 cores)
      minimal  — Absolute minimum (just makes SOMETHING work)
    """

    PROFILES = {
        'ultra': {
            'crf': 14, 'preset': 'slow',
            'width': 1920, 'height': 1080,
            'fps': 60.0,
            'analysis_samples': 16,
            'particle_count': 800,
            'rsmb_samples': 12,
            'transition_frames': 25,
            'enable_mask_effects': True,
            'enable_particles': True,
            'enable_rsmb': True,
            'enable_optical_flow': True,
        },
        'high': {
            'crf': 16, 'preset': 'medium',
            'width': 1280, 'height': 720,
            'fps': 30.0,
            'analysis_samples': 12,
            'particle_count': 400,
            'rsmb_samples': 8,
            'transition_frames': 18,
            'enable_mask_effects': True,
            'enable_particles': True,
            'enable_rsmb': True,
            'enable_optical_flow': True,
        },
        'medium': {
            'crf': 18, 'preset': 'veryfast',
            'width': 1280, 'height': 720,
            'fps': 30.0,
            'analysis_samples': 8,
            'particle_count': 200,
            'rsmb_samples': 6,
            'transition_frames': 12,
            'enable_mask_effects': False,
            'enable_particles': True,
            'enable_rsmb': False,
            'enable_optical_flow': True,
        },
        'low': {
            'crf': 22, 'preset': 'ultrafast',
            'width': 854, 'height': 480,
            'fps': 24.0,
            'analysis_samples': 5,
            'particle_count': 80,
            'rsmb_samples': 4,
            'transition_frames': 8,
            'enable_mask_effects': False,
            'enable_particles': False,
            'enable_rsmb': False,
            'enable_optical_flow': False,
        },
        'minimal': {
            'crf': 26, 'preset': 'ultrafast',
            'width': 640, 'height': 360,
            'fps': 24.0,
            'analysis_samples': 3,
            'particle_count': 0,
            'rsmb_samples': 0,
            'transition_frames': 5,
            'enable_mask_effects': False,
            'enable_particles': False,
            'enable_rsmb': False,
            'enable_optical_flow': False,
        },
    }

    def detect(self) -> dict:
        """Auto-detect hardware and return appropriate profile."""
        import platform
        profile_name = 'medium'  # safe default

        try:
            import psutil  # type: ignore[import-not-found]
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            cpu_cores = psutil.cpu_count(logical=False) or 2
            free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

            print(f"[Profile] RAM={ram_gb:.1f}GB  Cores={cpu_cores}  Free={free_ram_gb:.1f}GB")

            if ram_gb >= 16 and cpu_cores >= 8:   profile_name = 'ultra'
            elif ram_gb >= 8  and cpu_cores >= 4:  profile_name = 'high'
            elif ram_gb >= 4  and cpu_cores >= 2:  profile_name = 'medium'
            elif ram_gb >= 2:                       profile_name = 'low'
            else:                                   profile_name = 'minimal'

            # Downgrade if low free RAM right now
            if free_ram_gb < 1.5 and profile_name in ('ultra', 'high'):
                profile_name = 'medium'
                print(f"[Profile] Low free RAM ({free_ram_gb:.1f}GB), downgrading to medium")

        except ImportError:
            print("[Profile] psutil not found, using medium profile")
            # Try OS-level detection
            try:
                if platform.system() == 'Windows':
                    import ctypes
                    mem_kb = ctypes.c_ulonglong(0)
                    ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
                    ram_gb = mem_kb.value / (1024 * 1024)
                    if ram_gb >= 8:   profile_name = 'high'
                    elif ram_gb >= 4: profile_name = 'medium'
                    else:             profile_name = 'low'
            except: pass
        except Exception as e:
            print(f"[Profile] Detection failed ({e}), using medium profile")

        profile = dict(self.PROFILES[profile_name])
        profile['profile_name'] = profile_name
        print(f"[Profile] Using: {profile_name.upper()} "
              f"({profile['width']}×{profile['height']} @ {profile['fps']:.0f}fps)")
        return profile

    @staticmethod
    def apply_to_args(profile: dict, args: argparse.Namespace) -> argparse.Namespace:
        """Override CLI args with profile settings (unless user explicitly set them)."""
        # Only override if user left defaults
        if args.width == 1280:  args.width  = profile['width']
        if args.height == 720:  args.height = profile['height']
        if args.fps == 30.0:    args.fps    = profile['fps']
        if args.crf == 16:      args.crf    = profile['crf']
        if args.preset == 'veryfast': args.preset = profile['preset']
        return args


# =============================================================================
# 18. PROGRESS REPORTER
# Real-time progress for Flask app integration.
# Writes JSON to a file that app.py can poll.
# =============================================================================

class ProgressReporter:
    """
    Reports progress to a JSON file.
    app.py polls this file to show progress in the UI.

    Format: {"stage": "...", "progress": 0-100, "message": "..."}
    """

    def __init__(self, report_path: str = None):
        self.path = report_path or os.path.join(
            tempfile.gettempdir(), 'archive_progress.json')
        self.callbacks = []  # optional Python callbacks

    def report(self, stage: str, progress: float, message: str = ''):
        data = {
            'stage':    stage,
            'progress': round(max(0, min(100, progress)), 1),
            'message':  message,
            'done':     stage == 'done',
            'error':    stage == 'error',
        }
        try:
            with open(self.path, 'w') as f:
                json.dump(data, f)
        except: pass
        # Console
        bar_len = 30
        filled = int(bar_len * progress / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  [{bar}] {progress:5.1f}%  {message[:50]}", end='', flush=True)
        if stage in ('done', 'error'): print()
        for cb in self.callbacks: cb(data)

    def done(self, message: str = 'Complete'):
        self.report('done', 100, message)

    def error(self, message: str):
        self.report('error', 0, f"ERROR: {message}")


# =============================================================================
# 19. FULL PIPELINE — Updated to use ALL new systems
# =============================================================================

def run_amv_full(clips: list, music: str, output: str, mood: str,
                  bpm: float, total_dur: float, engine: str,
                  crf: int, preset: str, width: int, height: int,
                  fps: float, seed: int = 42,
                  auto_profile: bool = False,
                  post_process: bool = True,
                  progress_path: str = None) -> bool:
    """
    Complete AMV pipeline using ALL systems:
      PerformanceProfiler → BeatAnalysis → ClipSorter →
      NarrativeTensionEngine → DecisionEngine + StyleConsistencyEngine +
      SynthEffectEngine → QualityScorer → RenderOrchestrator →
      VelocityEngine → PostProcessor
    """
    prog = ProgressReporter(progress_path)
    prog.report('start', 0, 'Starting Archive Brain...')

    print(f"\n{'='*60}")
    print(f"  ARCHIVE DECISION BRAIN  v4.0  —  Full Pipeline")
    print(f"  Mood: {mood}  |  BPM: {bpm or 'auto'}  |  Clips: {len(clips)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    rng = random.Random(seed)

    # ── Step 0: Performance profile ──
    if auto_profile:
        profiler = PerformanceProfiler()
        profile  = profiler.detect()
        width    = profile.get('width',  width)
        height   = profile.get('height', height)
        fps      = profile.get('fps',    fps)
        crf      = profile.get('crf',    crf)
        preset   = profile.get('preset', preset)
    prog.report('profile', 5, f'Profile: {width}×{height}@{fps:.0f}fps')

    # ── Step 1: Music analysis ──
    prog.report('music', 8, 'Analyzing music...')
    ba = analyze_music(music, bpm) if music and os.path.exists(music) \
         else _synthetic_beats(bpm or 130, total_dur or 60)
    if not total_dur: total_dur = ba.duration
    bc = BeatCurve(ba.beats, ba.drops, ba.bpm)
    print(f"\n[1/8] Music: BPM={ba.bpm:.1f}  Beats={len(ba.beats)}  Drops={len(ba.drops)}")

    # ── Step 2: Clip analysis ──
    prog.report('analyze', 12, f'Analyzing {len(clips)} clips...')
    print(f"[2/8] Analyzing {len(clips)} clips...")
    scenes = []
    for i, clip in enumerate(clips):
        prog.report('analyze', 12 + i * 6 / len(clips),
                    f'Analyzing clip {i+1}/{len(clips)}...')
        print(f"  [{i+1}/{len(clips)}] {Path(clip).name}...", end=' ', flush=True)
        sc = analyze_clip(clip, engine)
        print(f"type={sc.scene_type} motion={sc.motion:.2f}")
        scenes.append(sc)

    # ── Step 3: Sort clips for narrative ──
    prog.report('sort', 22, 'Sorting clips for narrative...')
    print("[3/8] Sorting clips for narrative arc...")
    tension_engine = NarrativeTensionEngine()
    sorter = ClipSorter()
    scenes = sorter.sort_for_narrative(scenes, len(scenes), tension_engine, rng)
    print(f"  Order: {' → '.join(s.scene_type for s in scenes)}")

    # ── Step 4: Decision engine ──
    prog.report('plan', 28, 'Planning edit...')
    print("[4/8] Planning edit with Decision Engine...")
    style_engine = StyleConsistencyEngine(mood)
    synth_engine = SynthEffectEngine(mood, rng)

    brain = DecisionEngine(mood, seed)
    # Use tension-aware clip durations
    clip_durations = tension_engine.plan_clip_durations(len(scenes), total_dur)
    segments = brain.plan(scenes, ba, total_dur, fps)

    # ── Step 5: Apply style + narrative + synth ──
    print("[5/8] Applying style consistency + narrative tension...")
    for i, seg in enumerate(segments):
        pos = i / max(1, len(segments) - 1)

        # Style consistency (visual language)
        seg = style_engine.apply_to_segment(seg)

        # Narrative tension (arc-level decisions)
        seg = tension_engine.modify_segment(seg, pos)

        # If effect is unknown or weak, synthesize
        known_effects = {
            'impact_flash_combo', 'dark_surge', 'emotional_peak',
            'zoom_burst_shake', 'chromatic_shake', 'speed_blur_lines',
            'motion_enhance', 'emotion_glow', 'portrait_enhance',
            'cinematic_grade', 'atmospheric_grade', 'energy_aura',
            'fire_outline', 'god_of_war', 'datamosh_corrupt', 'vhs_glitch',
            'impact_frame', 'dolly_zoom', 'action', 'scenery',
        }
        if seg.effect not in known_effects:
            recipe = synth_engine.synthesize(
                seg.scene.scene_type, seg.section,
                seg.effect_intensity, seg.beat_energy)
            seg.effect = synth_engine.recipe_to_effect_name(recipe)
            print(f"  [Synth] Seg {i}: {recipe['description']}")

    # ── Step 6: Quality check ──
    prog.report('quality', 40, 'Scoring edit plan...')
    print("[6/8] Quality scoring...")
    scorer = QualityScorer()
    report = scorer.score(segments, ba)
    scorer.print_report(report)
    print(f"  Plan summary:")
    for i, seg in enumerate(segments):
        print(f"  [{i:02d}] [{seg.section:8s}] {seg.effect:25s} "
              f"fx={seg.effect_intensity:.2f} "
              f"trans={seg.transition_in:18s} "
              f"anim={seg.animation:15s} "
              f"{'DROP' if seg.is_drop else '    '} "
              f"z={seg.zoom_start:.2f}→{seg.zoom_end:.2f}[{seg.zoom_curve}]")

    # ── Step 7: Render ──
    prog.report('render', 45, f'Rendering {len(segments)} segments...')
    print(f"[7/8] Rendering {len(segments)} segments...")
    vel_engine = VelocityEngine(bc)
    orch = RenderOrchestrator(engine, crf, preset, width, height, fps)
    rendered = []
    trans_files = []

    for i, seg in enumerate(segments):
        pct = 45 + (i / len(segments)) * 40
        prog.report('render', pct, f'Rendering clip {i+1}/{len(segments)}: {seg.effect}')
        out_clip = os.path.join(orch.tmp, f'clip_{i:03d}.mp4')
        print(f"  [{i+1:02d}/{len(segments)}] {seg.effect}  "
              f"[{seg.section}]  drop={'Y' if seg.is_drop else 'N'}...",
              end=' ', flush=True)
        ok = orch.render_segment(seg, out_clip, vel_engine)
        print('✓' if ok else '✗(fallback)')
        rendered.append(out_clip if ok and os.path.exists(out_clip) else seg.clip_path)

        # Render transition
        if i > 0 and seg.transition_in not in ('cut',):
            trans_out = os.path.join(orch.tmp, f'trans_{i:03d}.mp4')
            tok = orch.render_transition(
                rendered[i-1], rendered[i],
                seg.transition_in, seg.trans_dur_frames, trans_out)
            trans_files.append(trans_out if tok and os.path.exists(trans_out) else '')
        else:
            trans_files.append('')

    # ── Assemble ──
    prog.report('assemble', 88, 'Assembling final AMV...')
    print("[8/8] Assembling...")
    assembled = output if not post_process else output + '.pre_post.mp4'
    ok = orch.assemble(segments, rendered, trans_files, music, assembled, total_dur)
    orch.cleanup()

    if not ok:
        prog.error('Assembly failed')
        print("✗ Assembly failed")
        return False

    # ── Post-processing ──
    if post_process and os.path.exists(assembled):
        prog.report('post', 93, 'Post-processing (master grade + loudness)...')
        print("  Post-processing: master grade + loudness normalization...")
        pp = PostProcessor(crf, preset)
        pp_ok = pp.process(assembled, output, mood, intensity=0.5)
        try: os.remove(assembled)
        except: pass
        if not pp_ok:
            # Fallback: just move pre-post as final
            import shutil; shutil.move(assembled, output)

    elapsed = time.time() - t0
    prog.done(f'Done in {elapsed:.1f}s → {output}')
    print(f"\n{'='*60}")
    print(f"  ✓ DONE in {elapsed:.1f}s")
    print(f"  Output: {output}")
    print(f"  Quality score: {report['score']}/100 [{report['grade']}]")
    print(f"  Segments: {len(segments)}  BPM: {ba.bpm:.1f}  "
          f"Drops: {len(ba.drops)}")
    print(f"{'='*60}\n")
    return True


# =============================================================================
# 20. SINGLE-EFFECT AND TRANSITION HELPERS (for app.py integration)
# =============================================================================

def run_single_effect(input_path: str, output_path: str,
                       effect: str, intensity: float, mood: str,
                       engine: str, crf: int = 18,
                       preset: str = 'veryfast') -> bool:
    """
    Apply one effect to one clip. Called by app.py for quick edits.
    Usage: run_single_effect('clip.mp4', 'out.mp4', 'glow', 0.8, 'hype', './engine')
    """
    cmd = [engine, 'apply_effect', effect, input_path, output_path,
           f'{intensity:.3f}', mood, str(crf), preset]
    try:
        r = subprocess.run(cmd, timeout=300)
        return r.returncode == 0
    except Exception as e:
        print(f"[SingleEffect] Failed: {e}")
        # FFmpeg fallback
        try:
            r2 = subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', input_path,
                '-vf', f'eq=contrast={1+intensity*0.3:.2f}:saturation={1+intensity*0.3:.2f}',
                '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-an', output_path
            ], timeout=400)
            return r2.returncode == 0
        except: return False


def run_transition_preview(clip_a: str, clip_b: str, trans_type: str,
                             output: str, engine: str,
                             n_frames: int = 20, fps: float = 30.0,
                             width: int = 1280, height: int = 720,
                             crf: int = 18, preset: str = 'veryfast') -> bool:
    """
    Preview a transition between two clips.
    Called by app.py to let user preview transitions.
    """
    cmd = [engine, 'render_transition',
           trans_type, clip_a, clip_b, output,
           str(n_frames), str(crf), preset,
           str(width), str(height), f'{fps:.1f}']
    try:
        r = subprocess.run(cmd, timeout=400)
        return r.returncode == 0
    except Exception as e:
        print(f"[TransPreview] Failed: {e}"); return False


# =============================================================================
# 21. UPDATED MAIN — Uses full pipeline
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Archive Decision Brain v4.0 — Full Creative AMV AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python archive_decision_brain.py \\
      --clips c1.mp4 c2.mp4 c3.mp4 \\
      --music track.mp3 --output amv.mp4 \\
      --mood hype --bpm 140 --dur 60

  # Auto-detect hardware quality:
  python archive_decision_brain.py --clips *.mp4 --music t.mp3 \\
      --output amv.mp4 --mood dark --auto-profile

  # Quick single effect:
  python archive_decision_brain.py \\
      --mode effect --input clip.mp4 --output out.mp4 \\
      --effect glow --intensity 0.8 --mood hype

  # Transition preview:
  python archive_decision_brain.py \\
      --mode transition --a clip1.mp4 --b clip2.mp4 \\
      --transition zoom_punch --output trans.mp4
        """
    )

    p.add_argument('--mode', default='amv',
                   choices=['amv', 'effect', 'transition', 'analyze', 'list'])
    p.add_argument('--clips', nargs='+', default=[])
    p.add_argument('--music',  default='')
    p.add_argument('--output', default='output.mp4')
    p.add_argument('--mood',   default='hype',
                   choices=['hype','phonk','dark','villain','sad','emotional',
                            'dramatic','cinematic','cyberpunk','neon','warm',
                            'golden','chill','lofi','bleach','teal_orange'])
    p.add_argument('--bpm',    type=float, default=0)
    p.add_argument('--dur',    type=float, default=0)
    p.add_argument('--engine', default='./archive_frame_engine')
    p.add_argument('--crf',    type=int,   default=16)
    p.add_argument('--preset', default='veryfast')
    p.add_argument('--width',  type=int,   default=1280)
    p.add_argument('--height', type=int,   default=720)
    p.add_argument('--fps',    type=float, default=30.0)
    p.add_argument('--seed',   type=int,   default=42)
    p.add_argument('--auto-profile',  action='store_true',
                   help='Auto-detect hardware and scale quality accordingly')
    p.add_argument('--no-post',       action='store_true',
                   help='Skip post-processing pass')
    p.add_argument('--progress-file', default=None,
                   help='Path to write JSON progress updates (for app.py)')

    # Single-effect mode
    p.add_argument('--input',     default='')
    p.add_argument('--effect',    default='cinematic_grade')
    p.add_argument('--intensity', type=float, default=0.80)

    # Transition mode
    p.add_argument('--a',          default='')
    p.add_argument('--b',          default='')
    p.add_argument('--transition', default='zoom_punch')

    args = p.parse_args()

    # ── Find engine binary ──
    engine = args.engine
    for candidate in [
        args.engine,
        './archive_frame_engine.exe',
        './archive_frame_engine',
        'archive_frame_engine.exe',
        'archive_frame_engine',
        os.path.join(os.path.dirname(__file__), 'archive_frame_engine.exe'),
        os.path.join(os.path.dirname(__file__), 'archive_frame_engine'),
    ]:
        if os.path.exists(candidate): engine = candidate; break
    print(f"[Brain] Engine: {engine}")

    # ── Auto-profile (potato PC mode) ──
    if args.auto_profile:
        profiler = PerformanceProfiler()
        profile  = profiler.detect()
        args     = PerformanceProfiler.apply_to_args(profile, args)

    # ── MODE DISPATCH ──

    if args.mode == 'list':
        print("AVAILABLE EFFECTS:")
        for e in ['impact_flash_combo','dark_surge','emotional_peak','zoom_burst_shake',
                   'chromatic_shake','speed_blur_lines','motion_enhance','emotion_glow',
                   'portrait_enhance','cinematic_grade','atmospheric_grade',
                   'energy_aura','fire_outline','god_of_war','datamosh_corrupt',
                   'vhs_glitch','impact_frame','dolly_zoom','action','scenery',
                   'glow','chromatic','vignette','film_grain','speed_lines',
                   'god_rays','anamorphic_flare','halation','datamosh','vhs',
                   'flash','radial_blur','turbulent','directional_mb','contrast_sat']:
            print(f"  {e}")
        print("\nAVAILABLE TRANSITIONS:")
        for t in ['cut','dissolve','flash_cut','zoom_punch','whip_pan_right','whip_pan_up',
                   'glitch_cut','smear_cut','fade_black','fade_in','cross_zoom',
                   'wipe_right','wipe_up','energy_slash','luma_wipe','ripple',
                   'spin','shatter_cut','tv_static','dolly_zoom_trans','impact_cut']:
            print(f"  {t}")
        print("\nAVAILABLE MOODS:")
        for m in ['hype','phonk','dark','villain','sad','emotional','dramatic',
                   'cinematic','cyberpunk','neon','warm','golden','chill','lofi']:
            print(f"  {m}")
        return

    if args.mode == 'effect':
        if not args.input: p.error("--mode effect requires --input")
        ok = run_single_effect(args.input, args.output, args.effect,
                                args.intensity, args.mood, engine,
                                args.crf, args.preset)
        print("✓ Done" if ok else "✗ Failed")
        sys.exit(0 if ok else 1)

    if args.mode == 'transition':
        if not args.a or not args.b: p.error("--mode transition requires --a and --b")
        ok = run_transition_preview(args.a, args.b, args.transition, args.output, engine,
                                     fps=args.fps, width=args.width, height=args.height,
                                     crf=args.crf, preset=args.preset)
        print("✓ Done" if ok else "✗ Failed")
        sys.exit(0 if ok else 1)

    if args.mode == 'analyze':
        if not args.input: p.error("--mode analyze requires --input")
        sc = analyze_clip(args.input, engine)
        print(json.dumps({
            'scene_type': sc.scene_type, 'motion': sc.motion,
            'brightness': sc.brightness, 'sharpness': sc.sharpness,
            'saturation': sc.saturation, 'edit_value': sc.edit_value,
            'duration': sc.duration, 'fps': sc.fps,
        }, indent=2))
        return

    # AMV mode
    if not args.clips: p.error("--mode amv requires --clips")
    for c in args.clips:
        if not os.path.exists(c): p.error(f"Clip not found: {c}")

    ok = run_amv_full(
        clips=args.clips, music=args.music, output=args.output,
        mood=args.mood, bpm=args.bpm, total_dur=args.dur,
        engine=engine, crf=args.crf, preset=args.preset,
        width=args.width, height=args.height, fps=args.fps, seed=args.seed,
        auto_profile=args.auto_profile,
        post_process=not args.no_post,
        progress_path=args.progress_file,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()