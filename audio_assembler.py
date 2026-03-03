"""
Stage 3: Audio Assembler
Assembles individual dialogue audio files into final narration with pauses and crossfading.

This module handles:
- Insertion of pauses between dialogue lines
- Crossfading between audio segments for natural transitions
- Final audio normalization and export
- Optional cleanup of intermediate files
"""

import os
import logging
from typing import List, Tuple, Dict
import tempfile

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAssembler:
    """
    Assembles individual dialogue audio files into final narration.
    
    Handles pause insertion, crossfading, and final audio export.
    """
    
    def __init__(self):
        """Initialize the audio assembler."""
        logger.info("Audio Assembler initialized")
    
    def assemble(
        self,
        rendered_audio_files: List[Tuple[str, Dict]],
        output_path: str,
        output_filename: str = "narration.wav"
    ) -> str:
        """
        Assemble individual audio files into final narration.
        
        Process:
        1. Load each audio file
        2. Insert pause_before from Director's Script
        3. Apply crossfade between segments
        4. Normalize final audio
        5. Export to file
        
        Args:
            rendered_audio_files: List of (audio_path, director_script_entry) tuples
            output_path: Directory to save final audio
            output_filename: Name of output file
        
        Returns:
            Path to final assembled audio file
        """
        if not rendered_audio_files:
            raise ValueError("No audio files to assemble")
        
        logger.info(f"Assembling {len(rendered_audio_files)} audio segments...")
        
        # Start with empty audio
        final_audio = AudioSegment.empty()
        
        for i, (audio_path, script_entry) in enumerate(rendered_audio_files):
            line_num = script_entry['line_number']
            pause_before = script_entry['pause_before']
            
            logger.info(
                f"Adding line {line_num} with {pause_before:.2f}s pause "
                f"({i+1}/{len(rendered_audio_files)})"
            )
            print(f"  📎 Assembling line {line_num}: pause={pause_before:.2f}s, path={audio_path}")
            
            try:
                # Load audio segment
                audio_segment = AudioSegment.from_wav(audio_path)
                print(f"     Loaded audio: {len(audio_segment)}ms duration")
                
                # Add pause before this line
                if pause_before > 0:
                    pause_ms = int(pause_before * 1000)
                    silence = AudioSegment.silent(duration=pause_ms)
                    print(f"     Adding {pause_ms}ms pause")
                    
                    if len(final_audio) > 0 and config.FADE_DURATION_MS > 0:
                        # Crossfade: fade out previous segment + fade in current with silence
                        final_audio = self._crossfade_segments(
                            final_audio, 
                            silence + audio_segment,
                            config.FADE_DURATION_MS
                        )
                        print(f"     Crossfaded. Total duration now: {len(final_audio)}ms")
                    else:
                        # First segment or no crossfade - just concatenate
                        final_audio += silence + audio_segment
                        print(f"     Concatenated. Total duration now: {len(final_audio)}ms")
                else:
                    # No pause - direct crossfade if not first segment
                    if len(final_audio) > 0 and config.FADE_DURATION_MS > 0:
                        final_audio = self._crossfade_segments(
                            final_audio,
                            audio_segment,
                            config.FADE_DURATION_MS
                        )
                        print(f"     Crossfaded (no pause). Total duration now: {len(final_audio)}ms")
                    else:
                        final_audio += audio_segment
                        print(f"     Concatenated (first segment). Total duration now: {len(final_audio)}ms")
                
            except Exception as e:
                logger.error(f"Failed to add line {line_num}: {e}")
                print(f"     ✗ ERROR adding line {line_num}: {e}")
                # Continue with next segment
                continue
        
        if len(final_audio) == 0:
            raise RuntimeError("Assembly failed: no audio segments were added")
        
        print(f"  ✓ Assembly complete: {len(final_audio)}ms ({len(final_audio)/1000:.1f}s) total duration")
        
        # Normalize final audio
        logger.info("Normalizing final audio...")
        final_audio = normalize(final_audio)
        
        # Export
        os.makedirs(output_path, exist_ok=True)
        output_filepath = os.path.join(output_path, output_filename)
        
        logger.info(f"Exporting final audio to: {output_filepath}")
        final_audio.export(
            output_filepath,
            format=config.AUDIO_FORMAT,
            parameters=["-ar", str(config.SAMPLE_RATE)]
        )
        
        duration = len(final_audio) / 1000.0  # Convert ms to seconds
        logger.info(f"✓ Assembly complete: {duration:.2f}s narration saved")
        
        return output_filepath
    
    def _crossfade_segments(
        self,
        segment1: AudioSegment,
        segment2: AudioSegment,
        fade_duration_ms: int
    ) -> AudioSegment:
        """
        Crossfade between two audio segments for smooth transition.
        
        Args:
            segment1: Previous audio segment
            segment2: Next audio segment
            fade_duration_ms: Crossfade duration in milliseconds
        
        Returns:
            Combined audio with crossfade
        """
        # Apply fade out to end of segment1
        fade_duration_ms = min(fade_duration_ms, len(segment1), len(segment2))
        
        if fade_duration_ms <= 0:
            # No crossfade possible
            return segment1 + segment2
        
        # Fade out last part of segment1
        segment1_faded = segment1.fade_out(duration=fade_duration_ms)
        
        # Fade in first part of segment2
        segment2_faded = segment2.fade_in(duration=fade_duration_ms)
        
        # Overlay the faded portions
        combined = segment1_faded.overlay(segment2_faded, position=len(segment1_faded) - fade_duration_ms)
        
        return combined
    
    def cleanup_intermediate_files(
        self, 
        rendered_audio_files: List[Tuple[str, Dict]],
        keep_on_error: bool = True
    ):
        """
        Clean up intermediate audio files after assembly.
        
        Args:
            rendered_audio_files: List of (audio_path, director_script_entry) tuples
            keep_on_error: If True, keep error audio files for debugging
        """
        if config.DEBUG_SAVE_INTERMEDIATE_AUDIO:
            logger.info("DEBUG mode: keeping intermediate audio files")
            return
        
        logger.info(f"Cleaning up {len(rendered_audio_files)} intermediate files...")
        
        cleaned = 0
        kept = 0
        
        for audio_path, script_entry in rendered_audio_files:
            try:
                # Check if it's an error file
                is_error_file = "_ERROR" in os.path.basename(audio_path)
                
                if is_error_file and keep_on_error:
                    logger.info(f"Keeping error file: {os.path.basename(audio_path)}")
                    kept += 1
                else:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to remove {audio_path}: {e}")
        
        logger.info(f"✓ Cleanup complete: {cleaned} removed, {kept} kept")


def assemble_narration(
    rendered_audio_files: List[Tuple[str, Dict]],
    output_path: str,
    output_filename: str = "narration.wav",
    cleanup: bool = True
) -> str:
    """
    Convenience function to assemble and optionally cleanup narration.
    
    Args:
        rendered_audio_files: List of (audio_path, director_script_entry) tuples
        output_path: Directory to save final audio
        output_filename: Name of output file
        cleanup: If True, remove intermediate files after assembly
    
    Returns:
        Path to final assembled audio file
    
    Example:
        >>> from voice import VoicePipeline
        >>> from audio_assembler import assemble_narration
        >>> 
        >>> voice = VoicePipeline()
        >>> voice.clone_voice("sample.wav")
        >>> rendered = voice.render_script(director_script)
        >>> 
        >>> final_audio = assemble_narration(rendered, "./outputs")
        >>> print(f"Narration saved to: {final_audio}")
    """
    assembler = AudioAssembler()
    
    # Assemble audio
    final_path = assembler.assemble(rendered_audio_files, output_path, output_filename)
    
    # Cleanup if requested
    if cleanup:
        assembler.cleanup_intermediate_files(rendered_audio_files)
    
    return final_path


if __name__ == "__main__":
    # Example usage
    print("Audio Assembler module ready. Use with rendered audio files from voice pipeline.")
    
    # Example structure:
    example_rendered = [
        ("/tmp/narration_line_001.wav", {
            "line_number": 1,
            "pause_before": 0.5,
            "speaker": "JOHN"
        }),
        ("/tmp/narration_line_002.wav", {
            "line_number": 2,
            "pause_before": 0.8,
            "speaker": "MARY"
        })
    ]
    
    # Would call: assemble_narration(example_rendered, "./outputs")
