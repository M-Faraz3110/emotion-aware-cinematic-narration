"""
Stage 3: Voice Pipeline
Voice cloning and per-line narration rendering using Tortoise-TTS.

This module handles:
- Voice sample preprocessing and validation
- Voice cloning from user-provided sample
- Per-line dialogue rendering with pace control
- Error handling and fallback audio generation
"""

import os
import tempfile
import logging
from typing import List, Dict, Optional, Tuple
import warnings

import torch
import librosa
import soundfile as sf
import numpy as np
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voices

import config

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Voice cloning and narration rendering pipeline using Tortoise-TTS.
    
    Clones a voice from a short audio sample and renders dialogue
    with appropriate pacing based on the Director's Script.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the voice pipeline with Tortoise-TTS model.
        
        Args:
            device: PyTorch device ("cuda" or "cpu"). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Voice Pipeline on device: {self.device}")
        
        # Load Tortoise-TTS model
        logger.info("Loading Tortoise-TTS models (this may take a moment)...")
        self.tts = TextToSpeech(device=self.device)
        
        logger.info("✓ Voice Pipeline initialized")
        
        # Voice cloning state
        self.voice_samples = []  # List of preprocessed voice sample paths
        self.is_voice_cloned = False
        self.custom_voice_name = "custom_voice"
    
    def clone_voice(self, voice_sample_path: str) -> bool:
        """
        Clone voice from an audio sample using Tortoise-TTS.
        
        Preprocesses the sample (trim silence, normalize, resample) and
        prepares it for voice cloning.
        
        Args:
            voice_sample_path: Path to voice sample audio file (WAV, MP3, etc.)
        
        Returns:
            True if voice cloning successful, False otherwise
        
        Raises:
            ValueError: If voice sample is too short or invalid
        """
        logger.info(f"Cloning voice from: {voice_sample_path}")
        
        if not os.path.exists(voice_sample_path):
            raise ValueError(f"Voice sample file not found: {voice_sample_path}")
        
        # Load and preprocess audio
        try:
            audio, original_sr = librosa.load(voice_sample_path, sr=None, mono=True)
            logger.info(f"Loaded voice sample: {len(audio)/original_sr:.2f}s @ {original_sr}Hz")
            
            # Preprocess
            processed_audio = self._preprocess_voice_sample(audio, original_sr)
            
            # Validate duration
            duration = len(processed_audio) / config.SAMPLE_RATE
            if duration < config.MIN_VOICE_SAMPLE_DURATION:
                raise ValueError(
                    f"Voice sample too short: {duration:.1f}s. "
                    f"Minimum required: {config.MIN_VOICE_SAMPLE_DURATION}s"
                )
            
            logger.info(f"✓ Voice sample preprocessed: {duration:.2f}s @ {config.SAMPLE_RATE}Hz")
            
            # Save preprocessed sample for Tortoise
            # Tortoise expects voice samples in a specific directory structure
            temp_dir = tempfile.gettempdir()
            voice_dir = os.path.join(temp_dir, "tortoise_voices", self.custom_voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            
            # Save sample (Tortoise can use multiple samples for better cloning)
            sample_path = os.path.join(voice_dir, "voice_sample_0.wav")
            sf.write(sample_path, processed_audio, config.SAMPLE_RATE)
            
            self.voice_samples = [sample_path]
            self.is_voice_cloned = True
            logger.info(f"✓ Voice cloned successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            self.is_voice_cloned = False
            return False
    
    def _preprocess_voice_sample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess voice sample audio.
        
        Steps:
        1. Trim leading/trailing silence
        2. Normalize audio levels
        3. Resample to Tortoise's sample rate (24000 Hz)
        
        Args:
            audio: Audio samples
            sr: Original sample rate
        
        Returns:
            Preprocessed audio at Tortoise's sample rate
        """
        # 1. Trim silence (using librosa's trim with default threshold)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        logger.info(f"  Trimmed silence: {len(audio)} → {len(audio_trimmed)} samples")
        
        # 2. Normalize audio to -20 dBFS (comfortable listening level)
        audio_normalized = librosa.util.normalize(audio_trimmed) * 0.1
        
        # 3. Resample to Tortoise's sample rate
        if sr != config.SAMPLE_RATE:
            audio_resampled = librosa.resample(
                audio_normalized, 
                orig_sr=sr, 
                target_sr=config.SAMPLE_RATE
            )
            logger.info(f"  Resampled: {sr}Hz → {config.SAMPLE_RATE}Hz")
        else:
            audio_resampled = audio_normalized
        
        return audio_resampled
    
    def render_script(
        self, 
        director_script: List[Dict],
        language: str = "en"
    ) -> List[Tuple[str, Dict]]:
        """
        Render entire Director's Script to audio files using Tortoise-TTS.
        
        Generates one audio file per dialogue line, applying pace control
        based on the Director's Script specifications.
        
        Args:
            director_script: Director's Script from NLP pipeline
            language: Language code (note: Tortoise is primarily English-optimized)
        
        Returns:
            List of tuples: (audio_file_path, director_script_entry)
        
        Raises:
            RuntimeError: If voice not cloned yet
        """
        if not self.is_voice_cloned:
            raise RuntimeError(
                "Voice not cloned. Call clone_voice() first with a voice sample."
            )
        
        logger.info(f"Rendering {len(director_script)} dialogue lines with Tortoise...")
        logger.info(f"Using preset: {config.TORTOISE_PRESET}")
        
        rendered_audio_files = []
        temp_dir = tempfile.gettempdir()
        
        # Load custom voice conditioning
        voice_samples, conditioning_latents = load_voices([self.custom_voice_name], 
                                                          extra_voice_dirs=[os.path.join(temp_dir, "tortoise_voices")])
        
        for i, entry in enumerate(director_script):
            line_num = entry['line_number']
            text = entry['line']
            pace = entry['pace']
            
            logger.info(f"Rendering line {line_num}/{len(director_script)}: {text[:50]}...")
            
            # Generate temp file path
            temp_audio_path = os.path.join(
                temp_dir, 
                f"narration_line_{line_num:03d}.wav"
            )
            
            try:
                # Generate audio with Tortoise using custom voice
                gen = self.tts.tts_with_preset(
                    text,
                    voice_samples=voice_samples,
                    conditioning_latents=conditioning_latents,
                    preset=config.TORTOISE_PRESET
                )
                
                # Convert to numpy array
                audio_array = gen.squeeze().cpu().numpy()
                
                # Apply pace control via time-stretching
                audio_paced = self._apply_pace_control_array(audio_array, pace)
                
                # Save to file
                sf.write(temp_audio_path, audio_paced, config.SAMPLE_RATE)
                
                rendered_audio_files.append((temp_audio_path, entry))
                logger.info(f"  ✓ Line {line_num} rendered with pace={pace}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to render line {line_num}: {e}")
                
                # Generate fallback error audio
                error_audio_path = self._generate_error_audio(line_num, text, temp_dir)
                rendered_audio_files.append((error_audio_path, entry))
        
        logger.info(f"✓ Rendering complete: {len(rendered_audio_files)} audio files")
        return rendered_audio_files
    
    def _apply_pace_control_array(self, audio: np.ndarray, pace: str) -> np.ndarray:
        """
        Apply pace control via time-stretching to audio array.
        
        Uses librosa's time_stretch to speed up or slow down audio
        while preserving pitch.
        
        Args:
            audio: Audio samples (numpy array)
            pace: Pace category ("slow", "normal", "fast")
        
        Returns:
            Time-stretched audio samples
        """
        # Get pace multiplier from config
        rate = config.TORTOISE_PACE_MULTIPLIERS.get(pace, 1.0)
        
        if rate == 1.0:
            # No time-stretching needed
            return audio
        
        # Apply time-stretch (rate > 1 = faster, rate < 1 = slower)
        audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        return audio_stretched
    
    def _generate_error_audio(
        self, 
        line_num: int, 
        failed_text: str, 
        output_dir: str
    ) -> str:
        """
        Generate fallback audio for failed line rendering.
        
        Creates a brief audio warning indicating that this line failed to render.
        
        Args:
            line_num: Line number that failed
            failed_text: The text that failed to render
            output_dir: Directory to save error audio
        
        Returns:
            Path to generated error audio file
        """
        error_message = f"Line {line_num} rendering failed."
        error_audio_path = os.path.join(
            output_dir, 
            f"narration_line_{line_num:03d}_ERROR.wav"
        )
        
        try:
            # Try to generate error message with Tortoise using a random preset
            gen = self.tts.tts(error_message, preset="ultra_fast")
            audio_array = gen.squeeze().cpu().numpy()
            sf.write(error_audio_path, audio_array, config.SAMPLE_RATE)
            logger.info(f"  Generated error audio: '{error_message}'")
            
        except Exception as e:
            # If even error audio fails, generate silence
            logger.warning(f"Error audio generation also failed: {e}. Using silence.")
            silence = np.zeros(int(config.SAMPLE_RATE * 1.0))  # 1 second silence
            sf.write(error_audio_path, silence, config.SAMPLE_RATE)
        
        return error_audio_path
    
    def cleanup_temp_files(self):
        """
        Clean up temporary voice samples.
        
        Call this when done with voice synthesis to free disk space.
        """
        if self.voice_samples:
            for sample_path in self.voice_samples:
                if os.path.exists(sample_path):
                    try:
                        # Remove the sample file
                        os.remove(sample_path)
                        # Try to remove the directory if empty
                        voice_dir = os.path.dirname(sample_path)
                        if os.path.exists(voice_dir) and not os.listdir(voice_dir):
                            os.rmdir(voice_dir)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {e}")
            logger.info("✓ Cleaned up temporary voice samples")


if __name__ == "__main__":
    # Example usage
    import json
    
    # This would normally come from the NLP pipeline
    sample_director_script = [
        {
            "line_number": 1,
            "speaker": "JOHN",
            "line": "I never wanted any of this.",
            "parenthetical": "",
            "emotion": "sadness",
            "intensity": 0.75,
            "pace": "slow",
            "pause_before": 0.5
        },
        {
            "line_number": 2,
            "speaker": "MARY",
            "line": "But here we are.",
            "parenthetical": "quietly",
            "emotion": "neutral",
            "intensity": 0.45,
            "pace": "normal",
            "pause_before": 0.8
        }
    ]
    
    # Initialize pipeline
    voice_pipeline = VoicePipeline()
    
    # Clone voice (you would provide actual voice sample path)
    # voice_pipeline.clone_voice("path/to/voice_sample.wav")
    
    # Render script
    # rendered_files = voice_pipeline.render_script(sample_director_script)
    
    print("Voice pipeline module ready. Use with actual voice sample and director script.")
