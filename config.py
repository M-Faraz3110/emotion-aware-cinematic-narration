"""
Configuration file for Emotion-Aware Cinematic Narration Engine.
Contains model names, paths, and default parameters.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Google Drive paths (for Colab)
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/NarrationEngine"
DRIVE_OUTPUT_PATH = os.path.join(DRIVE_PROJECT_PATH, "outputs")

# Local output paths (fallback)
LOCAL_OUTPUT_PATH = BASE_DIR / "outputs"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# NLP Models (HuggingFace)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SPACY_MODEL = "en_core_web_trf"  # Transformer-based English model

# Voice Cloning (Coqui TTS)
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# ============================================================================
# NLP PIPELINE PARAMETERS
# ============================================================================

# Emotion Analysis
EMOTION_LABELS = ["joy", "anger", "fear", "sadness", "disgust", "surprise", "neutral"]

# Multi-emotion blending
EMOTION_BLEND_TOP_N = 3           # Keep top N emotions in blend
EMOTION_BLEND_MIN_SCORE = 0.10    # Minimum score to include in blend (10%)

# Contextual Analysis
CONTEXT_WINDOW_BEFORE = 2  # Number of lines before target
CONTEXT_WINDOW_AFTER = 2   # Number of lines after target

# Emotion blending weights (context-dominant strategy: 25-30-15-30)
LINE_WEIGHT = 0.25         # 25% weight for target line itself
DIALOGUE_WEIGHT = 0.30     # 30% weight for surrounding dialogue context
PARENTHETICAL_WEIGHT = 0.15  # 15% weight for parenthetical (direct delivery instruction)
SCENE_WEIGHT = 0.30        # 30% weight for scene context (atmospheric mood)

# Intensity thresholds
INTENSITY_HIGH_THRESHOLD = 0.8    # High emotion intensity
INTENSITY_LOW_THRESHOLD = 0.3     # Low emotion intensity

# ============================================================================
# PACING & PAUSE PARAMETERS
# ============================================================================

# Pace categories
PACE_SLOW = "slow"
PACE_NORMAL = "normal"
PACE_FAST = "fast"

# Emotion pace contributions (used for multi-emotion weighted pacing)
# Positive = faster, Negative = slower, 0 = neutral
EMOTION_PACE_CONTRIBUTIONS = {
    'anger': 0.2,      # High-energy, fast
    'fear': 0.2,       # High-energy, fast
    'joy': 0.15,       # Energetic, slightly fast
    'surprise': 0.1,   # Mild increase
    'neutral': 0.0,    # No change
    'sadness': -0.2,   # Low-energy, slow
    'disgust': -0.2,   # Low-energy, slow
}

# Base pause values (in seconds)
BASE_PAUSE = 0.3
MIN_PAUSE = 0.1
MAX_PAUSE = 2.0

# Punctuation pause values
PAUSE_ELLIPSIS = 0.6        # "..."
PAUSE_EM_DASH = 0.4         # "—"
PAUSE_QUESTION = 0.1        # "?" (slight increase)
PAUSE_EXCLAMATION = 0.0     # "!" (typically faster, no extra pause)

# Emotion-based pause modifiers
PAUSE_HIGH_INTENSITY = 0.4  # Added for intensity > 0.8
PAUSE_LOW_INTENSITY = -0.2  # Subtracted for intensity < 0.3

# Delivery emotion boosts (applied after blending to respect delivery cues)
# Maps parenthetical keywords to emotion boosts (added to blended score)
DELIVERY_EMOTION_BOOSTS = {
    'laughing': {'joy': 0.40},          # Strong joy signal
    'smiling': {'joy': 0.35},           # Moderate joy signal
    'grinning': {'joy': 0.40},          # Strong joy signal
    'crying': {'sadness': 0.40},        # Strong sadness signal
    'sobbing': {'sadness': 0.45},       # Very strong sadness signal
    'tearful': {'sadness': 0.35},       # Moderate sadness signal
    'weeping': {'sadness': 0.40},       # Strong sadness signal
    'angry': {'anger': 0.40},           # Strong anger signal
    'furious': {'anger': 0.45},         # Very strong anger signal
    'shouting': {'anger': 0.35},        # Moderate anger signal
    'yelling': {'anger': 0.35},         # Moderate anger signal
    'screaming': {'fear': 0.35},        # Moderate fear signal
    'terrified': {'fear': 0.40},        # Strong fear signal
    'panicked': {'fear': 0.40},         # Strong fear signal
}

# Parenthetical mappings
PARENTHETICAL_MAPPINGS = {
    # Pauses
    "beat": {"pace": PACE_SLOW, "pause_before": 1.2, "intensity_modifier": 0.0},
    "pause": {"pace": PACE_SLOW, "pause_before": 1.2, "intensity_modifier": 0.0},
    "long pause": {"pace": PACE_SLOW, "pause_before": 1.8, "intensity_modifier": 0.0},
    
    # Quiet/soft delivery
    "quietly": {"pace": PACE_SLOW, "pause_before": 0.0, "intensity_modifier": -0.2},
    "whispering": {"pace": PACE_SLOW, "pause_before": 0.0, "intensity_modifier": -0.2},
    "whispers": {"pace": PACE_SLOW, "pause_before": 0.0, "intensity_modifier": -0.2},
    "softly": {"pace": PACE_SLOW, "pause_before": 0.0, "intensity_modifier": -0.2},
    
    # Loud/intense delivery
    "shouting": {"pace": PACE_FAST, "pause_before": 0.0, "intensity_modifier": 0.2},
    "yelling": {"pace": PACE_FAST, "pause_before": 0.0, "intensity_modifier": 0.2},
    "angry": {"pace": PACE_FAST, "pause_before": 0.0, "intensity_modifier": 0.2},
    "screaming": {"pace": PACE_FAST, "pause_before": 0.0, "intensity_modifier": 0.25},
    
    # Emotional delivery
    "crying": {"pace": PACE_SLOW, "pause_before": 1.0, "intensity_modifier": 0.2},
    "sobbing": {"pace": PACE_SLOW, "pause_before": 1.0, "intensity_modifier": 0.2},
    "breaking": {"pace": PACE_SLOW, "pause_before": 1.0, "intensity_modifier": 0.2},
    "tearful": {"pace": PACE_SLOW, "pause_before": 0.8, "intensity_modifier": 0.15},
    
    # Speed modifiers
    "quickly": {"pace": PACE_FAST, "pause_before": 0.0, "intensity_modifier": 0.0},
    "rushed": {"pace": PACE_FAST, "pause_before": -0.2, "intensity_modifier": 0.1},
    "slowly": {"pace": PACE_SLOW, "pause_before": 0.4, "intensity_modifier": 0.0},
    "hesitant": {"pace": PACE_SLOW, "pause_before": 0.6, "intensity_modifier": -0.1},
    "hesitantly": {"pace": PACE_SLOW, "pause_before": 0.6, "intensity_modifier": -0.1},
}

# ============================================================================
# VOICE SYNTHESIS PARAMETERS
# ============================================================================

# Voice sample requirements
MIN_VOICE_SAMPLE_DURATION = 6.0  # Minimum seconds for voice cloning

# XTTS pace multipliers (relative to normal speech rate)
XTTS_PACE_MULTIPLIERS = {
    PACE_SLOW: 0.85,    # 15% slower
    PACE_NORMAL: 1.0,   # Normal speed
    PACE_FAST: 1.15,    # 15% faster
}

# Emotional prosody modulation (post-processing)
# Pitch shifts in semitones (positive = higher, negative = lower)
# Keep very subtle to avoid robotic artifacts
EMOTION_PITCH_SHIFTS = {
    'joy': 0.5,         # Very slightly brighter
    'surprise': 0.8,    # Slightly higher pitched
    'fear': 0.8,        # Slightly tense
    'anger': -0.3,      # Very slightly gruff
    'sadness': -0.5,    # Slightly subdued
    'disgust': -0.3,    # Slightly lower
    'neutral': 0.0,     # No change
}

# Energy modulation multipliers (volume adjustment)
EMOTION_ENERGY_MODS = {
    'joy': 1.15,        # 15% louder, energetic
    'anger': 1.30,      # 30% louder, forceful, intense
    'fear': 1.20,       # 20% louder, panicked
    'surprise': 1.18,   # 18% louder, expressive
    'sadness': 0.80,    # 20% quieter, subdued
    'disgust': 0.85,    # 15% quieter, withdrawn
    'neutral': 1.0,     # No change
}

# Audio processing
SAMPLE_RATE = 22050  # Target sample rate for TTS
AUDIO_FORMAT = "wav"
FADE_DURATION_MS = 20  # Fade in/out duration to avoid clicks

# ============================================================================
# GRADIO UI PARAMETERS
# ============================================================================

# UI Configuration
GRADIO_THEME = "soft"
GRADIO_SHARE = True  # Create public URL in Colab

# Example screenplay excerpts
EXAMPLE_SCRIPTS = {
    "Blade Runner - Tears in Rain": """ROY: I've seen things you people wouldn't believe.
Attack ships on fire off the shoulder of Orion.
I watched C-beams glitter in the dark near the Tannhäuser Gate.
All those moments will be lost in time, like tears in rain.
(pause)
Time to die.""",
    
    "No Country for Old Men - Opening": """NARRATOR: I was sheriff of this county when I was twenty-five years old.
Hard to believe.
My grandfather was a sheriff too.
My father...
(beat)
He never was.
I always wondered if he'd taken this job, would he have felt the same way I do.""",
    
    "The Dark Knight - Interrogation": """JOKER: (laughing) You have nothing. Nothing to threaten me with.
Nothing to do with all your strength.
BATMAN: (quietly) Don't talk like one of them. You're not!
JOKER: No, you're right. I'm not.
(beat)
To them, you're just a freak... like me.""",
}

# ============================================================================
# LOGGING AND DEBUG
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Debug flags
DEBUG_SAVE_INTERMEDIATE_AUDIO = False  # Save individual line audio files
DEBUG_PRINT_DIRECTOR_SCRIPT = True     # Print full Director's Script JSON
