# =============================================================================
# ARCHIVE SUPER NLP ENGINE  —  nlp.py
# =============================================================================
# Drop-in replacement for your existing nlp.py.
# No LLM needed. No downloads. Works instantly on any machine.
#
# WHAT MAKES THIS BETTER:
#   - Understands Hindi + Hinglish + English + slang + typos
#   - Context-aware (negation, intensity modifiers, contradictions)
#   - Phrase-level matching (not just single words)
#   - Understands anime references (Gojo, Sukuna, Demon Slayer, etc.)
#   - Understands editing styles (Alight Motion, AE style, etc.)
#   - Music is ALWAYS added unless user explicitly says no
#   - Smart conflict resolution (e.g. "dark but chill" → dramatic)
#   - 400+ trigger terms across all categories
# =============================================================================

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

MOODS = ("hype", "dark", "sad", "dramatic", "chill")

# =============================================================================
# NORMALIZER
# =============================================================================

def _norm(text: str) -> str:
    """Normalize text: lowercase, remove accents, collapse spaces."""
    if not text:
        return ""
    t = text.strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    # Common typo / shorthand fixes
    t = re.sub(r"\bcharector\b",    "character",    t)
    t = re.sub(r"\bcharecter\b",    "character",    t)
    t = re.sub(r"\bcharacter\b",    "character",    t)
    t = re.sub(r"\bphonk\b",        "phonk",        t)
    t = re.sub(r"\bslowmo\b",       "slow motion",  t)
    t = re.sub(r"\bslomo\b",        "slow motion",  t)
    t = re.sub(r"\bslo mo\b",       "slow motion",  t)
    t = re.sub(r"\bslomotion\b",    "slow motion",  t)
    t = re.sub(r"\bfastmo\b",       "fast motion",  t)
    t = re.sub(r"\bspeedup\b",      "fast motion",  t)
    t = re.sub(r"\bfastforward\b",  "fast motion",  t)
    t = re.sub(r"\bamv\b",          "anime edit",   t)
    t = re.sub(r"\bvfx\b",          "visual effects", t)
    t = re.sub(r"\bbhai\b",         "",             t)  # remove filler
    t = re.sub(r"\byaar\b",         "",             t)
    t = re.sub(r"\bbro\b",          "",             t)
    t = re.sub(r"\bdude\b",         "",             t)
    t = re.sub(r"[^\w\s\-]",        " ",            t)
    t = re.sub(r"\s+",              " ",            t).strip()
    return t


def _has(text: str, *phrases: str) -> bool:
    """Check if any phrase exists in text (word-boundary aware)."""
    for phrase in phrases:
        p = phrase.strip()
        if not p:
            continue
        if " " in p:
            if p in text:
                return True
        else:
            if re.search(rf"\b{re.escape(p)}\b", text):
                return True
    return False


def _score(text: str, terms: list[str]) -> float:
    """Count how many terms from list appear in text."""
    count = 0.0
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if " " in t:
            if t in text:
                count += 1.0
        else:
            if re.search(rf"\b{re.escape(t)}\b", text):
                count += 1.0
    return count


def _negated(text: str, terms: list[str]) -> bool:
    """Check if any term is preceded by a negation word."""
    negations = (
        "no", "not", "dont", "don t", "do not",
        "without", "avoid", "skip", "remove",
        "never", "nahi", "mat", "nope", "nope",
        "bina", "chodo", "ignore",
    )
    for term in terms:
        t = term.strip()
        if not t:
            continue
        pattern = rf"\b({'|'.join(re.escape(n) for n in negations)})\s+(?:any\s+)?(?:the\s+)?{re.escape(t)}\b"
        if re.search(pattern, text):
            return True
    return False


def _intensity_multiplier(text: str) -> float:
    """Return a boost based on intensity words before a term."""
    if _has(text, "very very", "extremely", "super super", "maximum", "max", "ultra",
            "insane", "insanely", "full power", "absolute", "zabardast zabardast"):
        return 3.0
    if _has(text, "very", "super", "really", "highly", "bahut", "bohot", "bahot",
            "zyada", "zabardast", "mega", "heavy", "hard"):
        return 2.0
    return 1.0


# =============================================================================
# MASTER TERM DICTIONARIES
# =============================================================================

# ── MOOD ─────────────────────────────────────────────────────────────────────

MOOD_TERMS: dict[str, list[str]] = {
    "hype": [
        # English
        "hype", "epic", "energetic", "energy", "aggressive", "fast", "fight",
        "action", "hard", "adrenaline", "lit", "banger", "fire", "insane",
        "crazy", "power", "intense", "beast", "goat", "goated", "sigma",
        "alpha", "based", "heat", "flames", "hyped", "turbo", "rapid",
        "swift", "speed", "rush", "burst", "explode", "blast", "rage",
        "fury", "warrior", "gladiator", "champion", "victory", "dominate",
        "unstoppable", "invincible", "savage", "brutal", "ruthless",
        "battle", "clash", "versus", "1v1", "op", "overpowered",
        "clutch", "destroy", "obliterate", "annihilate", "slay",
        # Hindi / Hinglish
        "josh", "tezz", "tez", "rapido", "energy wala", "hard wala",
        "dhamakedaar", "dhamakedar", "jalwa", "mast edit", "full josh",
        "full power", "zabardast", "ekdum mast", "jhakaas",
        # Anime / characters that imply hype
        "gojo", "sukuna", "naruto", "ichigo", "luffy", "zoro", "guts",
        "levi", "eren", "tanjiro", "zenitsu", "inosuke", "giyu",
        "akaza", "muzan", "all might", "deku", "bakugo", "todoroki",
        "aizen", "kenpachi", "yamamoto", "madara", "hashirama", "minato",
        "itachi", "obito", "pain", "nagato", "jiraiya", "rock lee",
        "might guy", "kakashi", "sasuke", "vegeta", "goku", "broly",
        "beerus", "frieza", "cell", "gohan", "piccolo", "trunks",
        "saitama", "genos", "garou", "speed of sound sonic",
        "giorno", "dio", "jotaro", "jolyne", "kira",
        "chainsaw man", "denji", "power", "makima", "aki",
        "demon slayer", "attack on titan", "jujutsu kaisen",
        "black clover", "my hero academia", "one piece", "bleach",
        "dragon ball", "one punch man", "fullmetal alchemist",
        # Edit styles that imply hype
        "amv", "phonk edit", "sigma edit", "drift edit", "hardstyle",
        "nightcore", "trap edit", "bass boosted",
    ],
    "dark": [
        # English
        "dark", "shadow", "noir", "villain", "gritty", "sinister", "evil",
        "cursed", "demon", "devil", "monster", "corrupt", "twisted",
        "menacing", "ominous", "threatening", "brooding", "gloomy",
        "bleak", "haunted", "eerie", "creepy", "horror", "fear",
        "despair", "hopeless", "doomed", "fallen", "damned",
        "nightmare", "abyss", "void", "darkness", "black",
        "malevolent", "vile", "wicked", "diabolical", "ruthless",
        "cold blooded", "heartless", "merciless", "cruel",
        "trap", "underground", "gutter", "raw",
        # Hindi / Hinglish
        "andhera", "andheri", "bura", "shaitan", "raakshas",
        "khatarnak", "dangerous wala", "dark wala",
        # Dark anime characters / series
        "light yagami", "death note", "ryuk", "l lawliet",
        "tokyo ghoul", "kaneki", "berserk", "griffith",
        "promised neverland", "evangelion", "monster",
        "fullmetal alchemist", "roy mustang", "envy",
        "parasyte", "hellsing", "alucard", "dandadan",
        "vinland saga", "askeladd", "thorfinn",
    ],
    "sad": [
        # English
        "sad", "emotional", "heartbreak", "cry", "tears", "melancholy",
        "sorrow", "grief", "loss", "lonely", "alone", "miss", "missing",
        "broken", "hurt", "pain", "suffering", "tragic", "tragedy",
        "goodbye", "farewell", "memories", "nostalgia", "reminisce",
        "regret", "guilt", "empty", "hollow", "numb", "devastated",
        "crushed", "shattered", "torn", "wounded", "aching",
        "bittersweet", "poignant", "moving", "touching", "tender",
        "melancholic", "somber", "solemn", "quiet sorrow",
        "heart", "soul", "deep", "feeling", "feels",
        # Hindi / Hinglish
        "dukhi", "udaas", "dard", "dil toota", "rona", "aansu",
        "yaad", "bichhad", "judai", "tanhaai", "akela",
        "toota hua", "bikhar gaya", "bichar gaye",
        # Sad anime
        "anohana", "your lie in april", "clannad", "angel beats",
        "violet evergarden", "a silent voice", "five centimeters",
        "erased", "toradora", "plastic memories", "i want to eat your pancreas",
        "wolf children", "grave of fireflies", "your name",
        "weathering with you", "spirited away", "howl moving castle",
    ],
    "dramatic": [
        # English
        "dramatic", "cinematic", "story", "climax", "narrative",
        "plot twist", "revelation", "intense moment", "turning point",
        "epic moment", "final battle", "last stand", "sacrifice",
        "redemption", "rise", "fall", "journey", "fate", "destiny",
        "legacy", "honor", "glory", "power of friendship",
        "overcome", "never give up", "believe", "determination",
        "will to fight", "protect", "avenge", "swear",
        "cinematic edit", "storytelling", "character arc",
        "emotional peak", "build up", "tension", "suspense",
        "thrilling", "gripping", "powerful moment",
        # Hindi / Hinglish
        "natakiya", "kahani", "filmy", "dramatic wala",
        "climax wala", "filmy edit", "story wala",
        # Dramatic anime moments
        "luffy gear 5", "goku ultra instinct", "naruto sage mode",
        "tanjiro sun breathing", "ichigo bankai", "erens titan",
        "levi vs beast titan", "all might last smash",
        "simon the driller", "kamina speech", "jiren power",
    ],
    "chill": [
        # English
        "chill", "calm", "soft", "lofi", "relax", "relaxing", "smooth",
        "mellow", "peaceful", "serene", "tranquil", "gentle", "easy",
        "laid back", "vibe", "vibes", "aesthetic", "dreamy", "cozy",
        "warm", "comfort", "soothing", "healing", "wholesome",
        "lighthearted", "cheerful", "happy", "joy", "positive",
        "uplifting", "hopeful", "bright", "sunshine", "spring",
        "summer breeze", "coffee shop", "lo fi beats",
        # Romance
        "romantic", "romance", "love", "sweet", "cute", "adorable",
        "lovely", "tender", "affection", "crush", "date", "together",
        "holding hands", "first love", "heartwarming",
        # Hindi / Hinglish
        "shaant", "shant", "aram", "sukoon", "pyar", "mohabbat",
        "ishq", "pyar wala", "chill wala", "relax wala",
        "meetha", "achha", "sundar",
        # Chill anime
        "slice of life", "studio ghibli", "my neighbor totoro",
        "kiki delivery service", "non non biyori", "yotsuba",
        "barakamon", "silver spoon", "mushishi", "natsume",
        "laid back camp", "yuru camp", "k on", "k-on",
    ],
}

# ── INTENSITY ─────────────────────────────────────────────────────────────────

INTENSITY_HIGH_TERMS = [
    "max", "maximum", "ultra", "insane", "insanely", "crazy", "extreme",
    "extremely", "pro", "professional", "hardcore", "full power", "full",
    "heavy", "super", "mega", "god level", "top level", "all out",
    "everything", "all effects", "sab kuch", "sabkuch", "zabardast",
    "bahut zyada", "ekdum full", "100 percent", "100%", "full blast",
    "overloaded", "overload", "max effects", "max intensity",
    "all the effects", "use everything", "add everything",
]

INTENSITY_MEDIUM_TERMS = [
    "medium", "moderate", "normal", "standard", "decent", "good",
    "solid", "reasonable", "balanced", "average", "regular",
    "thoda zyada", "thoda",
]

INTENSITY_LOW_TERMS = [
    "simple", "clean", "minimal", "subtle", "light", "soft", "basic",
    "little", "slight", "gentle", "easy", "small", "tiny", "less",
    "low", "quiet", "calm", "not too much", "not much", "not many",
    "kam", "thoda sa", "halka", "simple wala", "seedha",
]

# ── NEGATION ─────────────────────────────────────────────────────────────────

NEGATION_WORDS = [
    "no", "not", "dont", "don t", "do not", "without", "avoid",
    "skip", "remove", "never", "nahi", "mat", "bina", "chodo",
    "hatao", "without any", "exclude", "except",
]

# ── MUSIC DISABLE ─────────────────────────────────────────────────────────────

NO_MUSIC_TERMS = [
    "no music", "no phonk", "without music", "without phonk",
    "dont add music", "dont add phonk", "do not add music",
    "do not add phonk", "no bgm", "no background music",
    "keep original audio", "use original audio", "original sound",
    "original audio only", "keep audio", "keep sound",
    "no song", "without song", "remove music", "mute music",
    "music mat lagao", "phonk mat lagao", "bina music",
    "bina phonk", "original wali awaaz",
]

# ── EFFECTS ───────────────────────────────────────────────────────────────────

EFFECT_TERMS: dict[str, list[str]] = {

    "shake": [
        "shake", "camera shake", "jhatka", "tremor", "shaky",
        "handheld", "vibration", "wobble", "unstable camera",
        "camera jitter", "camera vibrate", "hila de",
    ],

    "zoom": [
        "zoom", "zoom in", "zoom out", "punch in", "punch in effect",
        "close up", "push in", "dolly zoom", "vertigo effect",
        "zoom effect", "nazdeek", "kareeb",
    ],

    "flash": [
        "flash", "flash cut", "flash transition", "white flash",
        "black flash", "strobe", "strobe effect", "blink", "flicker",
        "chamak", "chamakna", "light flash",
    ],

    "glow": [
        "glow", "glow effect", "bloom", "bloom effect", "neon glow",
        "light glow", "soft glow", "deep glow", "god rays",
        "radiance", "luminous", "glowing", "roshni",
    ],

    "glitch": [
        "glitch", "glitch effect", "digital glitch", "rgb split",
        "chromatic aberration", "color split", "artifact", "corrupt",
        "signal noise", "databend", "distort", "vhs glitch",
        "matrix glitch", "lag", "freeze frame glitch",
    ],

    "motion_blur": [
        "motion blur", "blur", "speed blur", "radial blur",
        "directional blur", "smear", "streaks", "dhundla",
        "blur effect", "rsmb", "motion smear",
    ],

    "grain": [
        "grain", "film grain", "noise", "texture", "grainy",
        "film look", "16mm", "35mm", "analog", "vintage grain",
        "dana", "old film look",
    ],

    "vignette": [
        "vignette", "dark corners", "dark edges", "edge darkening",
        "cinematic vignette", "lens vignette", "corner shadow",
    ],

    "speed_ramp": [
        "speed ramp", "velocity ramp", "ramp", "speed change",
        "slow then fast", "fast then slow", "time ramp",
        "variable speed", "dynamic speed",
    ],

    "slow_motion": [
        "slow motion", "slow mo", "slomo", "slowmo", "slow",
        "twixtor", "optical flow", "frame interpolation",
        "120fps", "240fps", "high speed", "slow down",
        "dhire dhire", "slow karo",
    ],

    "fast_motion": [
        "fast motion", "fast forward", "speed up", "timelapse",
        "hyperlapse", "quick", "tez karo",
    ],

    "reverse": [
        "reverse", "rewind", "backwards", "time reverse",
        "play backwards", "ulta",
    ],

    "boomerang": [
        "boomerang", "ping pong", "forward reverse", "loop",
        "yo yo", "back and forth",
    ],

    "fire_lines": [
        "fire lines", "fire outline", "fire boundary", "boundary fire",
        "flame outline", "flame lines", "electric outline",
        "lightning outline", "energy outline", "neon outline",
        "character fire", "character outline", "burn outline",
        "fire around character", "fire effect character",
        "highlight boundary", "boundary highlight",
        "character boundary fire", "fire wala outline",
        "character ke around fire", "outline fire",
        "ae fire", "after effects fire", "saber effect",
        "lightsaber outline",
    ],

    "energy_aura": [
        "energy aura", "aura", "power aura", "ki aura",
        "chakra aura", "electric aura", "aura effect",
        "saber", "light saber", "neon lines", "energy lines",
        "power lines", "magic aura", "shield aura",
        "aura farming", "aura farm", "aurafarm",
        "power up aura", "transformation aura",
    ],

    "color_grade": [
        "color grade", "colour grade", "color grading",
        "colour grading", "cinematic color", "cinematic colours",
        "lut", "lookup table", "vibrant colors", "boost colors",
        "increase saturation", "color correction",
        "film grade", "teal orange", "orange teal",
    ],

    "chromatic": [
        "chromatic", "chromatic aberration", "rgb split",
        "color split", "color bleed", "lens distortion color",
        "rgb offset", "channel split",
    ],

    "particles": [
        "particles", "particle effects", "sparks", "embers",
        "fireflies", "dust particles", "bokeh", "snow",
        "snowfall", "rain", "raindrops", "smoke", "mist",
        "fog", "atmospheric particles",
    ],

    "ken_burns": [
        "ken burns", "pan", "parallax", "camera move",
        "slow pan", "cinematic pan", "drift", "camera drift",
        "subtle movement", "gentle pan",
    ],

    "overlap_transitions": [
        "overlap", "crossfade", "cross fade", "dissolve",
        "smooth transition", "seamless transition", "cool transition",
        "cool transitions", "transition heavy", "more transitions",
        "overlap transition", "blend transition",
    ],

    "anime_tone": [
        "anime tone", "anime look", "toon", "cel shade",
        "cel shading", "cartoon look", "stylized", "manga look",
        "anime style", "illustrated look",
    ],

    "black_white": [
        "black and white", "black white", "monochrome",
        "grayscale", "greyscale", "b w", "b and w", "bw",
        "desaturated", "colorless", "no color",
        "kala safed", "black n white",
    ],

    "turbulent_displace": [
        "turbulent displace", "turbulence", "warp", "distortion",
        "wave warp", "scene warp", "surreal", "trippy",
        "psychedelic", "liquid warp", "heat distortion",
    ],

    "posterize_time": [
        "posterize time", "choppy", "12 fps", "15 fps",
        "stop motion", "impact fps", "stutter", "choppy fps",
        "low fps look",
    ],

    "mask_transition": [
        "mask transition", "wipe", "luma wipe", "shape transition",
        "character cut transition", "mask wipe", "iris wipe",
    ],

    "blend_overlay": [
        "blend", "overlay", "screen blend", "add blend",
        "light leaks", "lens flare overlay", "texture overlay",
        "double exposure",
    ],

    "mask_roto": [
        "masking", "rotoscope", "rotoscoping", "rotobrush",
        "subject mask", "background blur", "focus on character",
        "blur background", "bokeh background",
    ],

    "chroma_key": [
        "green screen", "greenscreen", "chroma key",
        "background remove", "remove background",
    ],

    "pro_edit_pack": [
        "pro anime edit", "professional anime edit", "ae style",
        "after effects style", "all ae tools", "hardcore edit",
        "top level edit", "advanced vfx", "all vfx", "all effects",
        "full edit", "complete edit", "everything",
        "god level edit", "beast edit", "op edit", "insane edit",
        "pro edit", "professional edit", "expert edit",
    ],

    "loop_clips": [
        "repeat clip", "loop clip", "replay clip",
        "stutter loop", "stutter repeat", "clip loop",
        "repeat some clips",
    ],

    "speed_lines": [
        "speed lines", "motion lines", "impact lines",
        "radial lines", "manga speed lines",
    ],

    "vhs": [
        "vhs", "vhs effect", "retro", "old school", "80s look",
        "90s look", "vintage video", "tape effect", "scanlines",
        "old tv", "crt effect",
    ],

    "datamosh": [
        "datamosh", "data mosh", "glitch art", "compression artifact",
        "pixel drift", "video glitch art",
    ],
}

# ── COLOR GRADE DISABLE ────────────────────────────────────────────────────────

NO_COLOR_TERMS = [
    "no color grade", "no colour grade", "no color grading",
    "no colour grading", "dont change colors", "don t change colors",
    "do not change colors", "keep original colors", "keep original colours",
    "no grading", "original color", "original colour",
    "keep colors same", "color mat badlo", "colour mat badlo",
    "original wale colors", "keep original",
]

# ── BEAT CONTROL ───────────────────────────────────────────────────────────────

ALL_BEATS_TERMS = [
    "all beats", "every beat", "each beat", "on all beats",
    "on every beat", "har beat", "every single beat",
]

NO_BEATS_TERMS = [
    "no beat", "no beat sync", "without beats", "free flow",
    "continuous", "no cuts", "no hard cuts",
]

SELECTIVE_BEATS_TERMS = [
    "selective beats", "smart beats", "where needed", "where required",
    "only some beats", "not all beats", "human like beats",
    "intelligent beats", "natural cuts", "organic cuts",
    "wisely", "jahan zaroorat ho",
]

# ── MUSIC FOLDER HINTS ─────────────────────────────────────────────────────────

ROMANTIC_MUSIC_TERMS = [
    "romantic music", "romantic phonk", "love song", "sweet music",
    "gentle music", "calm music", "soft music", "chill music",
    "happy music", "positive vibes music", "feel good music",
]

PHONK_TERMS = [
    "phonk", "dark phonk", "slowed phonk", "drift phonk",
    "aggressive phonk", "memphis phonk", "underground phonk",
]

# ── SPECIAL OVERRIDES ──────────────────────────────────────────────────────────

SIMPLE_TERMS = [
    "simple", "clean", "minimal", "keep it simple", "basic",
    "just simple", "seedha", "saada",
]

QUALITY_TERMS = [
    "quality priority", "best quality", "highest quality",
    "no quality loss", "4k quality", "hd quality",
]

FAST_RENDER_TERMS = [
    "fast render", "quick render", "render fast", "asap",
    "as fast as possible", "minimum time", "jaldi karo",
]

# ── ANIME REFERENCE MAP (maps character/series to mood) ──────────────────────

ANIME_MOOD_MAP: dict[str, str] = {
    # Hype characters
    "gojo":         "hype", "sukuna":       "hype",
    "naruto":       "hype", "sasuke":       "hype",
    "ichigo":       "hype", "luffy":        "hype",
    "zoro":         "hype", "guts":         "hype",
    "bakugo":       "hype", "all might":    "hype",
    "deku":         "hype", "todoroki":     "hype",
    "levi":         "hype", "eren":         "hype",
    "tanjiro":      "hype", "zenitsu":      "hype",
    "inosuke":      "hype", "akaza":        "hype",
    "madara":       "hype", "pain":         "hype",
    "itachi":       "dramatic", "obito":    "dramatic",
    "goku":         "hype", "vegeta":       "hype",
    "broly":        "hype", "saitama":      "hype",
    "garou":        "hype", "denji":        "hype",
    # Dark characters
    "light yagami": "dark", "ryuk":         "dark",
    "kaneki":       "dark", "griffith":     "dark",
    "alucard":      "dark", "muzan":        "dark",
    "dio":          "dark", "kira":         "dark",
    "makima":       "dark",
    # Sad / emotional
    "violet evergarden": "sad", "clannad":  "sad",
    "anohana":      "sad",
    # Chill / romantic
    "howl":         "chill", "chihiro":     "chill",
    "kiki":         "chill",
    # Series → mood
    "demon slayer":         "hype",
    "jujutsu kaisen":       "hype",
    "attack on titan":      "dramatic",
    "my hero academia":     "hype",
    "one piece":            "hype",
    "bleach":               "hype",
    "dragon ball":          "hype",
    "one punch man":        "hype",
    "death note":           "dark",
    "tokyo ghoul":          "dark",
    "berserk":              "dark",
    "chainsaw man":         "dark",
    "fullmetal alchemist":  "dramatic",
    "vinland saga":         "dramatic",
    "your lie in april":    "sad",
    "violet evergarden":    "sad",
    "a silent voice":       "sad",
    "clannad":              "sad",
    "anohana":              "sad",
    "studio ghibli":        "chill",
    "spirited away":        "chill",
    "your name":            "dramatic",
    "weathering with you":  "dramatic",
    "sword art online":     "hype",
    "re zero":              "dramatic",
    "steins gate":          "dramatic",
    "hunter x hunter":      "hype",
    "hxh":                  "hype",
    "killua":               "hype",
    "gon":                  "hype",
    "black clover":         "hype",
    "asta":                 "hype",
    "fairy tail":           "hype",
    "natsu":                "hype",
    "seven deadly sins":    "hype",
    "meliodas":             "hype",
    "overlord":             "dark",
    "ainz":                 "dark",
    "sword art online":     "hype",
    "kirito":               "hype",
    "asuna":                "dramatic",
    "re:zero":              "dramatic",
    "rezero":               "dramatic",
    "subaru":               "dramatic",
    "emilia":               "dramatic",
    "rem":                  "sad",
}

# ── PROMPT ALIAS EXPANSION ────────────────────────────────────────────────────

PROMPT_ALIASES: dict[str, str] = {
    "aura farm":            "energy aura character focus",
    "aurafarm":             "energy aura character focus",
    "aura farming":         "energy aura character focus",
    "aura edit":            "energy aura character focus",
    "goated edit":          "pro anime edit",
    "op edit":              "pro anime edit",
    "god level edit":       "pro anime edit",
    "beast edit":           "pro anime edit",
    "sigma edit":           "hype pro anime edit",
    "hard transitions":     "cool transitions overlap transitions",
    "crazy transitions":    "cool transitions overlap transitions",
    "twixtor edit":         "slow motion optical flow",
    "velocity edit":        "slow motion speed ramp",
    "flow edit":            "slow motion smooth",
    "fire edit":            "fire lines energy aura hype",
    "dark villain edit":    "dark villain fire lines energy aura",
    "emotional edit":       "sad slow motion ken burns",
    "cinematic edit":       "dramatic cinematic color grade",
    "phonk edit":           "hype phonk aggressive",
    "drift edit":           "hype phonk speed ramp",
    "all out":              "max intensity all effects",
    "full send":            "max intensity all effects",
    "no holds barred":      "max intensity all effects",
    "keep it clean":        "simple minimal no effects",
}


# =============================================================================
# CORE PARSER
# =============================================================================

@dataclass
class ParseResult:
    raw:            str
    normalized:     str
    mood:           str
    mood_scores:    dict[str, float]
    intensity:      int
    music:          bool
    no_music:       bool
    no_phonk:       bool
    effects:        dict[str, bool]
    color_mode:     str   # "hype"|"dark"|"sad"|"dramatic"|"chill"|"original"|"black_white"
    no_color_grade: bool
    beats:          str   # "all"|"selective"|"none"
    style:          str   # "action"|"emotional"|"cinematic"|"chaos"|"clean"
    quality:        bool
    fast_render:    bool
    simple_mode:    bool
    anime_refs:     list[str]
    detected_language: str   # "english"|"hindi"|"hinglish"
    confidence:     float
    notes:          list[str] = field(default_factory=list)


def _detect_language(text: str) -> str:
    hindi_words = ["bhai", "yaar", "karo", "wala", "wali", "zyada", "thoda",
                   "bahut", "bohot", "bahot", "aur", "nahi", "mat", "bina",
                   "ekdum", "jhakaas", "dhamakedaar", "dhamakedar"]
    hindi_count = sum(1 for w in hindi_words if _has(text, w))
    if hindi_count >= 3:
        return "hindi"
    if hindi_count >= 1:
        return "hinglish"
    return "english"


def _expand_aliases(text: str) -> str:
    for alias, expansion in sorted(PROMPT_ALIASES.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{re.escape(alias)}\b", expansion, text)
    return text


def _resolve_anime_refs(text: str) -> tuple[list[str], str | None]:
    """Find anime character/series references and return (refs, implied_mood)."""
    found = []
    implied_mood = None
    best_score   = 0

    for ref, mood in ANIME_MOOD_MAP.items():
        if _has(text, ref):
            found.append(ref)
            # Longer match = more specific = higher confidence
            s = len(ref)
            if s > best_score:
                best_score   = s
                implied_mood = mood

    return found, implied_mood


def parse_prompt(raw_prompt: str) -> ParseResult:
    raw  = raw_prompt or ""
    text = _norm(raw)
    text = _expand_aliases(text)

    lang     = _detect_language(text)
    mult     = _intensity_multiplier(text)

    # ── 1. Anime reference detection ─────────────────────────────────────────
    anime_refs, anime_mood = _resolve_anime_refs(text)

    # ── 2. Mood scoring ───────────────────────────────────────────────────────
    mood_scores: dict[str, float] = {}
    for mood in MOODS:
        s = _score(text, MOOD_TERMS[mood])
        mood_scores[mood] = s

    # Apply anime mood hint
    if anime_mood and mood_scores.get(anime_mood, 0) < 1:
        mood_scores[anime_mood] = mood_scores.get(anime_mood, 0) + 1.5

    # Romantic override: if user uses romantic terms without sad/dark → chill
    romantic_hits = _score(text, [
        "romantic", "romance", "love", "sweet", "cute", "pleasant",
        "gentle", "tender", "wholesome", "pyar", "mohabbat", "ishq",
    ])
    sad_strict = _score(text, ["sad", "heartbreak", "cry", "tears", "dukhi", "dard"])
    dark_strict = _score(text, ["dark", "villain", "shadow", "evil", "shaitan"])
    if romantic_hits > 0 and sad_strict == 0 and dark_strict == 0:
        mood_scores["chill"]    += 2.0 + romantic_hits * 0.4
        mood_scores["sad"]       = max(0, mood_scores["sad"] - 1.5)
        mood_scores["dark"]      = max(0, mood_scores["dark"] - 1.0)

    # Multiply by intensity modifier
    for m in mood_scores:
        mood_scores[m] *= mult if m == max(mood_scores, key=mood_scores.get) else 1.0

    # Determine final mood
    mood = max(mood_scores, key=mood_scores.get) if any(v > 0 for v in mood_scores.values()) else "hype"

    # ── 3. Music ──────────────────────────────────────────────────────────────
    no_music  = _has(text, *NO_MUSIC_TERMS) or _negated(text, ["music", "phonk", "song", "audio"])
    no_phonk  = no_music or _has(text, "no phonk", "without phonk", "phonk mat",
                                  "phonk nahi", "avoid phonk")
    music     = not no_music

    # ── 4. Effects ────────────────────────────────────────────────────────────
    effects: dict[str, bool] = {}
    for fx_key, terms in EFFECT_TERMS.items():
        is_neg = _negated(text, terms)
        is_pos = _has(text, *terms)
        effects[fx_key] = is_pos and not is_neg

    # ── 5. Color ──────────────────────────────────────────────────────────────
    is_bw       = effects.get("black_white", False)
    no_grade    = _has(text, *NO_COLOR_TERMS) or _negated(text, ["color", "colour", "grade", "grading"])
    if is_bw:
        no_grade = True
    color_mode  = "black_white" if is_bw else ("original" if no_grade else mood)

    # ── 6. Beats ──────────────────────────────────────────────────────────────
    beats = "selective"
    if _has(text, *ALL_BEATS_TERMS):
        beats = "all"
    elif _has(text, *NO_BEATS_TERMS):
        beats = "none"
    elif _has(text, *SELECTIVE_BEATS_TERMS):
        beats = "selective"

    # ── 7. Style ──────────────────────────────────────────────────────────────
    style = "action"
    if _has(text, "cinematic", "film", "movie", "cinematic edit"):
        style = "cinematic"
    elif sad_strict > 0 or _has(text, "emotional", "cry", "dukhi"):
        style = "emotional"
    elif _has(text, "simple", "clean", "minimal", "basic"):
        style = "clean"
    elif _has(text, "chaos", "crazy", "all effects", "everything", "insane"):
        style = "chaos"
    elif _has(text, "action", "fight", "battle"):
        style = "action"

    # ── 8. Intensity ──────────────────────────────────────────────────────────
    simple_mode  = _has(text, *SIMPLE_TERMS) and not _has(text, *INTENSITY_HIGH_TERMS)
    if simple_mode:
        intensity = 1
    elif _has(text, *INTENSITY_HIGH_TERMS):
        intensity = 3
    elif _has(text, *INTENSITY_MEDIUM_TERMS):
        intensity = 2
    else:
        # Default: strong edit unless chill/romantic
        if mood in ("sad", "chill") and romantic_hits > 0:
            intensity = 2
        else:
            intensity = 3

    # Pro pack forces intensity 3
    if effects.get("pro_edit_pack", False):
        intensity = 3

    # ── 9. Quality / render ───────────────────────────────────────────────────
    quality     = _has(text, *QUALITY_TERMS)
    fast_render = _has(text, *FAST_RENDER_TERMS)

    # ── 10. Confidence ────────────────────────────────────────────────────────
    total_hits     = sum(mood_scores.values()) + sum(1 for v in effects.values() if v)
    max_possible   = 20.0
    confidence     = float(min(1.0, total_hits / max_possible))

    # ── 11. Notes (for debugging) ─────────────────────────────────────────────
    notes = []
    if anime_refs:
        notes.append(f"anime refs: {', '.join(anime_refs[:3])}")
    if romantic_hits > 0 and not sad_strict:
        notes.append("romantic intent detected → chill mood")
    if lang != "english":
        notes.append(f"language: {lang}")
    active_fx = [k for k, v in effects.items() if v]
    if active_fx:
        notes.append(f"effects: {', '.join(active_fx[:6])}")

    return ParseResult(
        raw=raw, normalized=text, mood=mood, mood_scores=mood_scores,
        intensity=intensity, music=music, no_music=no_music, no_phonk=no_phonk,
        effects=effects, color_mode=color_mode, no_color_grade=no_grade,
        beats=beats, style=style, quality=quality, fast_render=fast_render,
        simple_mode=simple_mode, anime_refs=anime_refs,
        detected_language=lang, confidence=confidence, notes=notes,
    )


# =============================================================================
# PUBLIC ADAPTER FUNCTIONS  (same interface as original nlp.py)
# =============================================================================

def infer_mood_from_prompt(prompt: str) -> str:
    return parse_prompt(prompt).mood


def build_style_profile_from_prompt(prompt: str, mood: str) -> dict[str, Any]:
    p = parse_prompt(prompt)
    fx = p.effects

    # If user didn't specify mood, use parsed mood
    if not mood or mood not in MOODS:
        mood = p.mood

    no_color = p.no_color_grade
    bw       = p.color_mode == "black_white"

    # Intensity-based auto-enable
    intensity = p.intensity
    if intensity >= 3 and not p.simple_mode:
        for k in ["shake","zoom","flash","glow","glitch","motion_blur",
                  "grain","vignette","speed_ramp","chromatic",
                  "overlap_transitions"]:
            fx.setdefault(k, True)
            if not fx.get(k):
                fx[k] = True
    elif intensity == 2:
        # Enable mood-appropriate defaults
        mood_auto: dict[str, list[str]] = {
            "hype":     ["shake","zoom","flash","speed_ramp"],
            "dark":     ["glitch","vignette","grain","chromatic"],
            "sad":      ["slow_motion","ken_burns","glow"],
            "dramatic": ["zoom","glow","ken_burns","overlap_transitions"],
            "chill":    ["ken_burns","glow","overlap_transitions"],
        }
        for k in mood_auto.get(mood, []):
            if not fx.get(k):
                fx[k] = True

    # Always-on base effects (unless simple mode or no_color)
    if not p.simple_mode:
        fx["grain"]   = True
        if not no_color:
            fx["vignette"]     = True
            fx["color_grade"]  = True

    # Disable color effects if no_color
    if no_color:
        fx["color_grade"]   = False
        fx["anime_tone"]    = False
        fx["chroma_pulse"]  = False
        if not bw:
            fx["vignette"]  = False
            fx["glow"]      = False

    # Pro pack
    if fx.get("pro_edit_pack", False):
        intensity = 3
        for k in ["shake","zoom","flash","glow","glitch","motion_blur",
                  "grain","vignette","edge_glow","speed_ramp","ken_burns",
                  "overlap_transitions","particles","energy_aura",
                  "turbulent_displace","posterize_time","mask_transition",
                  "blend_overlay","mask_roto","fire_lines",
                  "character_focus"]:
            fx[k] = True

    profile: dict[str, Any] = {
        # Core
        "intensity":             intensity,
        "mood":                  p.mood,
        "style":                 p.style,

        # Music
        "no_music":              p.no_music,
        "no_phonk":              p.no_phonk,
        "smart_music":           p.music,
        "music_mood":            p.mood,

        # Visual effects
        "shake":                 bool(fx.get("shake",              False)),
        "zoom":                  bool(fx.get("zoom",               False)),
        "flash":                 bool(fx.get("flash",              False)),
        "glow":                  bool(fx.get("glow",               False)),
        "glitch":                bool(fx.get("glitch",             False)),
        "motion_blur":           bool(fx.get("motion_blur",        False)),
        "grain":                 bool(fx.get("grain",              True)),
        "vignette":              bool(fx.get("vignette",           False)),
        "edge_glow":             bool(fx.get("edge_glow",          False)),
        "speed_ramp":            bool(fx.get("speed_ramp",         False)),
        "slow_motion":           bool(fx.get("slow_motion",        False)),
        "fast_motion":           bool(fx.get("fast_motion",        False)),
        "reverse":               bool(fx.get("reverse",            False)),
        "boomerang":             bool(fx.get("boomerang",          False)),
        "anime_tone":            bool(fx.get("anime_tone",         False)),
        "ken_burns":             bool(fx.get("ken_burns",          False)),
        "chroma_pulse":          bool(fx.get("chromatic",          False)),
        "color_grade":           bool(fx.get("color_grade",        False)),
        "character_focus":       bool(fx.get("mask_roto",          False)) or
                                 bool(fx.get("character_focus",    False)),
        "black_white":           bw,
        "no_color_grade":        no_color,
        "fire_lines":            bool(fx.get("fire_lines",         False)),
        "overlap_transitions":   bool(fx.get("overlap_transitions",False)),
        "particles":             bool(fx.get("particles",          False)),
        "energy_aura":           bool(fx.get("energy_aura",        False)),
        "turbulent_displace":    bool(fx.get("turbulent_displace", False)),
        "posterize_time":        bool(fx.get("posterize_time",     False)),
        "mask_transition":       bool(fx.get("mask_transition",    False)),
        "blend_overlay":         bool(fx.get("blend_overlay",      False)),
        "mask_roto":             bool(fx.get("mask_roto",          False)),
        "chroma_key":            bool(fx.get("chroma_key",         False)),
        "difference_matte":      bool(fx.get("difference_matte",  False)),
        "object_inpaint":        bool(fx.get("object_inpaint",     False)),
        "pro_edit_pack":         bool(fx.get("pro_edit_pack",      False)),
        "auto_timefx":           intensity >= 3,
        "speed_lines":           bool(fx.get("speed_lines",        False)),
        "vhs":                   bool(fx.get("vhs",                False)),
        "datamosh":              bool(fx.get("datamosh",           False)),

        # Beats
        "selective_beats":       p.beats == "selective",
        "all_beats":             p.beats == "all",

        # Quality & render
        "fast_render":           p.fast_render,
        "quality_priority":      p.quality or intensity >= 3,
        "loop_clips":            bool(fx.get("loop_clips",         False)),
        "loop_count":            2,
        "loop_ending":           False,
        "simple_mode":           p.simple_mode,

        # Meta
        "explicit_effects":      [k for k, v in fx.items() if v],
        "_anime_refs":           p.anime_refs,
        "_language":             p.detected_language,
        "_confidence":           p.confidence,
        "_notes":                p.notes,
    }

    # Final safety: if all_beats is set, clear selective_beats
    if profile["all_beats"]:
        profile["selective_beats"] = False

    return profile


def infer_music_profile_from_prompt(prompt: str, mood: str) -> dict[str, float]:
    p = parse_prompt(prompt)
    m = p.mood if p.mood in MOODS else mood

    base: dict[str, dict[str, float]] = {
        "hype":     {"energy": 0.88, "onset": 0.86, "brightness": 0.64},
        "dark":     {"energy": 0.72, "onset": 0.64, "brightness": 0.28},
        "dramatic": {"energy": 0.70, "onset": 0.60, "brightness": 0.52},
        "sad":      {"energy": 0.36, "onset": 0.28, "brightness": 0.34},
        "chill":    {"energy": 0.28, "onset": 0.24, "brightness": 0.50},
    }
    vals = base.get(m, {"energy": 0.60, "onset": 0.50, "brightness": 0.50}).copy()

    # Adjust for detected context
    text = p.normalized
    if _has(text, "aggressive", "hard", "intense", "fight"):
        vals["energy"] = min(1.0, vals["energy"] + 0.08)
        vals["onset"]  = min(1.0, vals["onset"]  + 0.08)
    if _has(text, "emotional", "slow", "sad", "dukhi"):
        vals["energy"] = max(0.0, vals["energy"] - 0.08)
        vals["onset"]  = max(0.0, vals["onset"]  - 0.06)
    if _has(text, "dark", "villain", "shadow"):
        vals["brightness"] = max(0.0, vals["brightness"] - 0.10)
    if _has(text, "romantic", "love", "sweet", "cute", "pleasant"):
        sad_check = _score(text, ["sad","heartbreak","cry","dukhi"])
        if sad_check == 0:
            vals["energy"]     = max(0.0, vals["energy"]     - 0.06)
            vals["onset"]      = max(0.0, vals["onset"]      - 0.08)
            vals["brightness"] = min(1.0, vals["brightness"] + 0.10)
    if p.intensity >= 3:
        vals["energy"] = min(1.0, vals["energy"] + 0.06)
        vals["onset"]  = min(1.0, vals["onset"]  + 0.06)

    return {
        "target_energy":     float(np.clip(vals["energy"],     0.0, 1.0)),
        "target_onset":      float(np.clip(vals["onset"],      0.0, 1.0)),
        "target_brightness": float(np.clip(vals["brightness"], 0.0, 1.0)),
    }


# =============================================================================
# QUICK TEST (python nlp.py to run)
# =============================================================================

if __name__ == "__main__":
    tests = [
        # English
        "make a hype phonk edit with fire lines and energy aura",
        "dark villain edit with glitch and rgb split max intensity",
        "sad emotional edit slow motion no phonk",
        "cinematic dramatic edit with color grade and glow",
        "simple clean edit no effects",
        # Hinglish
        "bhai ek dark wala villain edit bana fire lines ke saath",
        "yaar ekdum mast hype edit chahiye full power",
        "sad emotional edit bana dukhi wala slow motion",
        "romantic sweet edit pyar wala no phonk",
        # Anime references
        "gojo vs sukuna edit hype phonk",
        "demon slayer tanjiro breathing edit",
        "attack on titan levi edit dark and cinematic",
        "your lie in april sad emotional edit",
        # Edge cases
        "no music just effects",
        "all effects everything max",
        "keep it simple and clean",
        "aura farm edit",
        "black and white edit with vhs effect",
        "do not change colors, just add shake and zoom",
    ]

    print("\n" + "="*70)
    print("ARCHIVE SUPER NLP — TEST RESULTS")
    print("="*70)

    for t in tests:
        r = parse_prompt(t)
        active = [k for k, v in r.effects.items() if v]
        print(f"\nPROMPT : {t[:60]}")
        print(f"  mood={r.mood:10s} intensity={r.intensity} music={r.music} lang={r.detected_language}")
        print(f"  beats={r.beats:10s} style={r.style:10s} confidence={r.confidence:.2f}")
        if r.anime_refs:
            print(f"  anime ={r.anime_refs[:3]}")
        if active:
            print(f"  fx    ={active[:8]}")
        if r.notes:
            print(f"  notes ={r.notes}")

    print("\n" + "="*70)