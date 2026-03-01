"""
Stage 3: Voice Pipeline
Voice cloning and per-line narration rendering using Coqui TTS XTTS-v2.

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
from TTS.api import TTS

import config

# Suppress TTS warnings
warnings.filterwarnings('ignore', category=UserWarning, module='TTS')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Voice cloning and narration rendering pipeline using XTTS-v2.
    
    Clones a voice from a short audio sample and renders dialogue
    with appropriate pacing based on the Director's Script.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the voice pipeline with XTTS-v2 model.
        
        Args:
            device: PyTorch device ("cuda" or "cpu"). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Voice Pipeline on device: {self.device}")
        
        # Load XTTS-v2 model
        logger.info(f"Loading TTS model: {config.TTS_MODEL}")
        self.tts = TTS(config.TTS_MODEL).to(self.device)
        
        logger.info("✓ Voice Pipeline initialized")
        
        # Voice cloning state
        self.speaker_wav_path: Optional[str] = None
        self.is_voice_cloned = False
    
    def clone_voice(self, voice_sample_path: str) -> bool:
        """
        Clone voice from an audio sample.
        
        Preprocesses the sample (trim silence, normalize, resample)
        and prepares it for XTTS voice cloning.
        
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
            audio, original_sr = librosa.load(voice_sample_path, sr=None)
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
            
            # Save preprocessed sample to temp file
            temp_dir = tempfile.gettempdir()
            self.speaker_wav_path = os.path.join(temp_dir, "speaker_voice_sample.wav")
            sf.write(self.speaker_wav_path, processed_audio, config.SAMPLE_RATE)
            
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
        3. Resample to target sample rate (22050 Hz)
        
        Args:
            audio: Audio samples
            sr: Sample rate
        
        Returns:
            Preprocessed audio at target sample rate
        """
        # 1. Trim silence (using librosa's trim with default threshold)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        logger.info(f"  Trimmed silence: {len(audio)} → {len(audio_trimmed)} samples")
        
        # 2. Normalize audio to -20 dBFS (comfortable listening level)
        audio_normalized = librosa.util.normalize(audio_trimmed) * 0.1
        
        # 3. Resample to target sample rate
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
        Render entire Director's Script to audio files.
        
        Generates one audio file per dialogue line, applying pace control
        based on the Director's Script specifications.
        
        Args:
            director_script: Director's Script from NLP pipeline
            language: Language code for TTS (default: "en")
        
        Returns:
            List of tuples: (audio_file_path, director_script_entry)
        
        Raises:
            RuntimeError: If voice not cloned yet
        """
        if not self.is_voice_cloned:
            raise RuntimeError(
                "Voice not cloned. Call clone_voice() first with a voice sample."
            )
        
        logger.info(f"Rendering {len(director_script)} dialogue lines...")
        
        rendered_audio_files = []
        temp_dir = tempfile.gettempdir()
        
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
                # Render with XTTS
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_audio_path,
                    speaker_wav=self.speaker_wav_path,
                    language=language
                )
                
                # Apply pace control via time-stretching
                audio_paced = self._apply_pace_control(temp_audio_path, pace)
                
                # Overwrite with paced audio
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
    
    def _apply_pace_control(self, audio_path: str, pace: str) -> np.ndarray:
        """
        Apply pace control via time-stretching.
        
        Uses librosa's time_stretch to speed up or slow down audio
        while preserving pitch.
        
        Args:
            audio_path: Path to audio file
            pace: Pace category ("slow", "normal", "fast")
        
        Returns:
            Time-stretched audio samples
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
        
        # Get pace multiplier from config
        rate = config.XTTS_PACE_MULTIPLIERS.get(pace, 1.0)
        
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
            # Try to generate error message with TTS
            self.tts.tts_to_file(
                text=error_message,
                file_path=error_audio_path,
                speaker_wav=self.speaker_wav_path,
                language="en"
            )
            logger.info(f"  Generated error audio: '{error_message}'")
            
        except Exception as e:
            # If even error audio fails, generate silence
            logger.warning(f"Error audio generation also failed: {e}. Using silence.")
            silence = np.zeros(int(config.SAMPLE_RATE * 1.0))  # 1 second silence
            sf.write(error_audio_path, silence, config.SAMPLE_RATE)
        
        return error_audio_path
    
    def cleanup_temp_files(self):
        """
        Clean up temporary voice sample file.
        
        Call this when done with voice synthesis to free disk space.
        """
        if self.speaker_wav_path and os.path.exists(self.speaker_wav_path):
            try:
                os.remove(self.speaker_wav_path)
                logger.info("✓ Cleaned up temporary voice sample")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


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
