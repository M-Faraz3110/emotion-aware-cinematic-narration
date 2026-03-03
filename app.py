"""
Stage 4: Gradio UI
Interactive web interface for the Emotion-Aware Cinematic Narration Engine.

This module provides:
- Audio and text input for voice sample and screenplay
- Real-time progress tracking
- Tabbed interface for results and Director's Script
- Example screenplay buttons
- Advanced parameter controls
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Optional, Tuple
import gradio as gr

from parser import parse_screenplay
from nlp import NLPPipeline
from voice import VoicePipeline
from audio_assembler import assemble_narration
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_screenplay_file(screenplay_file) -> str:
    """
    Load screenplay from uploaded file.
    
    Args:
        screenplay_file: Uploaded screenplay file
        
    Returns:
        Content of the screenplay file
    """
    if screenplay_file is None:
        return ""
    
    try:
        # Handle different Gradio file input formats
        if isinstance(screenplay_file, dict):
            file_path = screenplay_file.get('name') or screenplay_file.get('path')
        elif hasattr(screenplay_file, 'name'):
            file_path = screenplay_file.name
        else:
            file_path = screenplay_file
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading file: {str(e)}"


def load_example_script(example_name: str) -> str:
    """
    Load example screenplay from examples directory.
    
    Args:
        example_name: Name of the example file (without .txt extension)
    
    Returns:
        Content of the example file
    """
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    example_file = f"{example_name}.txt"
    example_path = os.path.join(examples_dir, example_file)
    
    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to embedded examples from config
        for name, content in config.EXAMPLE_SCRIPTS.items():
            if example_name.lower() in name.lower():
                return content
        return f"Example '{example_name}' not found."


def process_narration(
    voice_sample,
    screenplay_text: str,
    format_hint: str,
    context_window_before: int,
    context_window_after: int,
    progress=gr.Progress()
) -> Tuple[Optional[str], str, str, str]:
    """
    Main processing function that orchestrates all pipeline stages.
    
    Args:
        voice_sample: Gradio audio input (file path or tuple)
        screenplay_text: Input screenplay or dialogue text
        format_hint: Format type ("Auto-detect", "Fountain", "Plain Dialogue")
        context_window_before: Number of context lines before target
        context_window_after: Number of context lines after target
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (audio_path, director_script_json, status_log, error_message)
    """
    status_log = []
    
    def log_status(message: str):
        """Helper to log status messages."""
        logger.info(message)
        status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    try:
        # === Stage 0: Validate inputs ===
        progress(0, desc="Validating inputs...")
        log_status("Starting narration generation...")
        
        if not screenplay_text or len(screenplay_text.strip()) < 10:
            return None, "", "\n".join(status_log), "❌ Error: Screenplay text is too short or empty."
        
        if voice_sample is None:
            return None, "", "\n".join(status_log), "❌ Error: Please provide a voice sample (record or upload audio)."
        
        # Extract audio file path from Gradio input
        if isinstance(voice_sample, dict):
            voice_sample_path = voice_sample.get('name') or voice_sample.get('path')
        elif isinstance(voice_sample, str):
            voice_sample_path = voice_sample
        else:
            voice_sample_path = voice_sample
        
        if not voice_sample_path:
            return None, "", "\n".join(status_log), "❌ Error: Invalid voice sample format."
        
        log_status(f"✓ Voice sample: {os.path.basename(voice_sample_path)}")
        log_status(f"✓ Screenplay length: {len(screenplay_text)} characters")
        
        # === Stage 1: Parse Screenplay ===
        progress(0.1, desc="📝 Parsing screenplay...")
        log_status("Stage 1: Parsing screenplay...")
        
        format_map = {
            "Auto-detect": "auto",
            "Fountain": "fountain",
            "Plain Dialogue": "plain"
        }
        format_type = format_map.get(format_hint, "auto")
        
        parsed_dialogue = parse_screenplay(screenplay_text, format_hint=format_type)
        log_status(f"✓ Parsed {len(parsed_dialogue)} dialogue lines")
        
        if len(parsed_dialogue) == 0:
            return None, "", "\n".join(status_log), "❌ Error: No dialogue found in screenplay. Check formatting."
        
        # === Stage 2: NLP Analysis ===
        progress(0.25, desc="🧠 Analyzing emotions and pacing...")
        log_status("Stage 2: Running NLP analysis...")
        
        # Apply user-specified context window settings
        original_before = config.CONTEXT_WINDOW_BEFORE
        original_after = config.CONTEXT_WINDOW_AFTER
        config.CONTEXT_WINDOW_BEFORE = context_window_before
        config.CONTEXT_WINDOW_AFTER = context_window_after
        
        nlp_pipeline = NLPPipeline()
        director_script = nlp_pipeline.analyze(parsed_dialogue)
        
        # Restore original settings
        config.CONTEXT_WINDOW_BEFORE = original_before
        config.CONTEXT_WINDOW_AFTER = original_after
        
        log_status(f"✓ Emotion analysis complete")
        log_status(f"  Detected emotions: {set(entry['emotion'] for entry in director_script)}")
        
        # === Stage 3: Voice Cloning ===
        progress(0.4, desc="🎤 Cloning voice...")
        log_status("Stage 3: Cloning voice from sample...")
        
        voice_pipeline = VoicePipeline()
        
        try:
            success = voice_pipeline.clone_voice(voice_sample_path)
            if not success:
                return None, "", "\n".join(status_log), "❌ Error: Voice cloning failed. Check voice sample quality and duration (min 6s)."
            log_status("✓ Voice cloned successfully")
        except ValueError as e:
            return None, "", "\n".join(status_log), f"❌ Error: {str(e)}"
        
        # === Stage 4: Render Dialogue ===
        progress(0.5, desc="🎙️ Rendering dialogue (this may take a few minutes)...")
        log_status(f"Stage 4: Rendering {len(director_script)} dialogue lines...")
        
        rendered_audio_files = []
        for i, entry in enumerate(director_script):
            progress(
                0.5 + (0.3 * (i / len(director_script))),
                desc=f"🎙️ Rendering line {i+1}/{len(director_script)}: {entry['line'][:40]}..."
            )
            
            temp_rendered = voice_pipeline.render_script([entry])
            rendered_audio_files.extend(temp_rendered)
            
            log_status(f"  ✓ Rendered line {entry['line_number']}/{len(director_script)}")
        
        log_status(f"✓ All lines rendered")
        
        # === Stage 5: Assemble Audio ===
        progress(0.85, desc="🔧 Assembling final narration...")
        log_status("Stage 5: Assembling audio with pauses and crossfades...")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"narration_{timestamp}.wav"
        
        # Assemble to temp directory (like voice_demo.py)
        final_audio_path = assemble_narration(
            rendered_audio_files,
            tempfile.gettempdir(),
            output_filename,
            cleanup=not config.DEBUG_SAVE_INTERMEDIATE_AUDIO
        )
        
        log_status(f"✓ Final audio assembled: {output_filename}")
        
        # === Stage 6: Generate Director's Script JSON ===
        progress(0.95, desc="📋 Generating Director's Script...")
        log_status("Generating Director's Script JSON...")
        
        director_script_json = json.dumps(director_script, indent=2, ensure_ascii=False)
        
        # Save Director's Script to temp
        script_filename = f"director_script_{timestamp}.json"
        script_path = os.path.join(tempfile.gettempdir(), script_filename)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(director_script_json)
        
        log_status(f"✓ Director's Script saved: {script_filename}")
        
        # === Complete ===
        progress(1.0, desc="✅ Complete!")
        log_status("="*60)
        log_status("✅ NARRATION GENERATION COMPLETE")
        log_status(f"📂 Output: {final_audio_path}")
        log_status(f"⏱️ Total lines: {len(director_script)}")
        log_status("="*60)
        
        # Cleanup voice pipeline temp files
        voice_pipeline.cleanup_temp_files()
        
        return final_audio_path, director_script_json, "\n".join(status_log), ""
    
    except Exception as e:
        logger.exception("Processing failed with exception")
        error_msg = f"❌ Unexpected error: {str(e)}\n\nCheck the status log for details."
        log_status(f"ERROR: {str(e)}")
        return None, "", "\n".join(status_log), error_msg


def create_gradio_interface() -> gr.Blocks:
    """
    Create the Gradio web interface.
    
    Returns:
        Gradio Blocks interface
    """
    
    # Custom CSS for better styling
    custom_css = """
    .tab-nav button { font-size: 16px; font-weight: 600; }
    .output-audio { border: 2px solid #4CAF50; border-radius: 8px; }
    .status-box { font-family: monospace; font-size: 12px; max-height: 400px; overflow-y: auto; }
    """
    
    with gr.Blocks(
        title="🎬 Emotion-Aware Cinematic Narration Engine",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown("""
        # 🎬 Emotion-Aware Cinematic Narration Engine
        
        Transform screenplays into emotionally intelligent narration. This engine:
        - 🧠 Analyzes dialogue emotion, intensity, and pacing using NLP
        - 🎤 Clones your voice from a short sample
        - 🎙️ Narrates with cinematic delivery
        
        **Instructions:**
        1. Provide a **6+ second voice sample** (record or upload)
        2. Paste your **screenplay or dialogue**
        3. Click **Generate Narration**
        4. Wait 2-5 minutes for processing
        5. Listen to your narration in the Results tab!
        """)
        
        # === Input Tab ===
        with gr.Tab("📥 Input"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎤 Voice Sample")
                    gr.Markdown("Provide a clear 6+ second recording of the voice you want to clone.")
                    
                    voice_input = gr.Audio(
                        label="Voice Sample",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                    
                    gr.Markdown("### 📝 Screenplay / Dialogue")
                    gr.Markdown("Upload a screenplay file **OR** paste text directly below.")
                    
                    screenplay_file = gr.File(
                        label="Upload Screenplay (.txt, .fountain)",
                        file_types=[".txt", ".fountain"],
                        type="filepath"
                    )
                    
                    # Example buttons
                    gr.Markdown("**Quick Examples:**")
                    with gr.Row():
                        btn_blade_runner = gr.Button("🤖 Blade Runner", size="sm")
                        btn_dark_knight = gr.Button("🦇 Dark Knight", size="sm")
                        btn_no_country = gr.Button("🔫 No Country", size="sm")
                    
                    screenplay_input = gr.Textbox(
                        label="Screenplay Text",
                        placeholder="Paste your screenplay or dialogue here...\n\nExample:\nJOHN: I never wanted this.\nMARY: (quietly) But here we are.",
                        lines=12,
                        max_lines=20
                    )
                    
                    format_input = gr.Dropdown(
                        label="Format",
                        choices=["Auto-detect", "Fountain", "Plain Dialogue"],
                        value="Auto-detect"
                    )
                    
                    # Advanced Options (Accordion)
                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        gr.Markdown("**NLP Context Window**")
                        context_before = gr.Slider(
                            label="Lines before target",
                            minimum=0,
                            maximum=5,
                            value=config.CONTEXT_WINDOW_BEFORE,
                            step=1
                        )
                        context_after = gr.Slider(
                            label="Lines after target",
                            minimum=0,
                            maximum=5,
                            value=config.CONTEXT_WINDOW_AFTER,
                            step=1
                        )
                        gr.Markdown("*Larger context windows improve emotion detection for short lines.*")
            
            generate_btn = gr.Button("🎬 Generate Narration", variant="primary", size="lg")
        
        # === Results Tab ===
        with gr.Tab("🎧 Results"):
            error_display = gr.Markdown(visible=False)
            
            audio_output = gr.Audio(
                label="Generated Narration",
                type="filepath",
                elem_classes=["output-audio"]
            )
            
            gr.Markdown("### 📊 Processing Status")
            status_output = gr.Textbox(
                label="Status Log",
                lines=15,
                max_lines=20,
                elem_classes=["status-box"]
            )
        
        # === Director's Script Tab ===
        with gr.Tab("📋 Director's Script"):
            gr.Markdown("""
            This tab shows the **Director's Script** — the enriched dialogue with emotion, 
            intensity, pace, and pause data that guides the narration engine.
            
            Each line includes:
            - **Emotion**: Detected emotion (joy, anger, sadness, etc.)
            - **Intensity**: Emotion strength (0.0 to 1.0)
            - **Pace**: Speaking speed (slow, normal, fast)
            - **Pause Before**: Silence before this line (seconds)
            """)
            
            director_script_output = gr.JSON(
                label="Director's Script JSON"
            )
        
        # === Event Handlers ===
        
        # File upload handler
        screenplay_file.change(
            fn=load_screenplay_file,
            inputs=screenplay_file,
            outputs=screenplay_input
        )
        
        # Example button clicks
        btn_blade_runner.click(
            fn=lambda: load_example_script("blade_runner_tears_in_rain"),
            outputs=screenplay_input
        )
        
        btn_dark_knight.click(
            fn=lambda: load_example_script("dark_knight_interrogation"),
            outputs=screenplay_input
        )
        
        btn_no_country.click(
            fn=lambda: load_example_script("no_country_opening"),
            outputs=screenplay_input
        )
        
        # Main generation
        generate_btn.click(
            fn=process_narration,
            inputs=[
                voice_input,
                screenplay_input,
                format_input,
                context_before,
                context_after
            ],
            outputs=[
                audio_output,
                director_script_output,
                status_output,
                error_display
            ]
        )
    
    return demo


if __name__ == "__main__":
    # Test the interface locally
    demo = create_gradio_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
