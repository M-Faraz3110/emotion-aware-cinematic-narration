"""
Voice Demo - Voice Cloning & Emotional TTS Testing
Standalone Gradio interface focused on voice cloning quality and emotional delivery.
"""

import gradio as gr
import os
import tempfile
from typing import Optional, List, Tuple

from voice import VoicePipeline
import config


class VoiceDemo:
    """Standalone voice cloning and TTS demo."""
    
    def __init__(self):
        """Initialize voice pipeline."""
        self.voice_pipeline = VoicePipeline()
        self.voice_cloned = False
    
    def clone_voice(self, voice_file) -> str:
        """
        Clone voice from uploaded sample.
        
        Args:
            voice_file: Uploaded voice sample (can be dict, str, or file object)
            
        Returns:
            Status message
        """
        if voice_file is None:
            return "❌ Please upload a voice sample first"
        
        try:
            # Handle different Gradio audio input formats
            if isinstance(voice_file, dict):
                voice_path = voice_file.get('name') or voice_file.get('path')
            elif isinstance(voice_file, str):
                voice_path = voice_file
            else:
                # Fallback for file objects
                voice_path = voice_file
            
            if not voice_path:
                return "❌ Could not extract voice sample path"
            
            success = self.voice_pipeline.clone_voice(voice_path)
            
            if success:
                self.voice_cloned = True
                return "✅ Voice cloned successfully! You can now test different emotions and paces."
            else:
                return "❌ Voice cloning failed. Ensure sample is 6+ seconds and clear audio."
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def test_emotion_variations(
        self, 
        text: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Render text with all emotion variations.
        
        Args:
            text: Text to render
            
        Returns:
            Tuple of audio paths for each emotion
        """
        if not self.voice_cloned:
            return None, None, None, None, None, None, None
        
        if not text.strip():
            return None, None, None, None, None, None, None
        
        try:
            emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']
            audio_outputs = []
            
            for idx, emotion in enumerate(emotions, start=1):
                director_script = [{
                    'line_number': idx,  # Use unique line number to avoid overwriting temp files
                    'speaker': 'SPEAKER',
                    'line': text,
                    'emotion': emotion,
                    'pace': config.PACE_NORMAL
                }]
                
                audio_files = self.voice_pipeline.render_script(director_script)
                
                if audio_files:
                    audio_outputs.append(audio_files[0][0])
                else:
                    audio_outputs.append(None)
            
            return tuple(audio_outputs)
            
        except Exception as e:
            print(f"Error rendering emotions: {e}")
            return None, None, None, None, None, None, None
    
    def test_pace_variations(
        self,
        text: str,
        emotion: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Render text with different pace variations.
        
        Args:
            text: Text to render
            emotion: Emotion to use
            
        Returns:
            Tuple of (slow_audio, normal_audio, fast_audio)
        """
        if not self.voice_cloned:
            return None, None, None
        
        if not text.strip():
            return None, None, None
        
        try:
            paces = [config.PACE_SLOW, config.PACE_NORMAL, config.PACE_FAST]
            audio_outputs = []
            
            for idx, pace in enumerate(paces, start=1):
                director_script = [{
                    'line_number': idx,  # Use unique line number to avoid overwriting temp files
                    'speaker': 'SPEAKER',
                    'line': text,
                    'emotion': emotion,
                    'pace': pace
                }]
                
                audio_files = self.voice_pipeline.render_script(director_script)
                
                if audio_files:
                    audio_outputs.append(audio_files[0][0])
                else:
                    audio_outputs.append(None)
            
            return tuple(audio_outputs)
            
        except Exception as e:
            print(f"Error rendering paces: {e}")
            return None, None, None
    
    def render_custom(
        self,
        text: str,
        emotion: str,
        pace: str
    ) -> Optional[str]:
        """
        Render text with custom emotion and pace.
        
        Args:
            text: Text to render
            emotion: Emotion selection
            pace: Pace selection
            
        Returns:
            Audio file path
        """
        if not self.voice_cloned:
            return None
        
        if not text.strip():
            return None
        
        try:
            director_script = [{
                'line_number': 1,
                'speaker': 'SPEAKER',
                'line': text,
                'emotion': emotion,
                'pace': pace
            }]
            
            audio_files = self.voice_pipeline.render_script(director_script)
            
            if audio_files:
                return audio_files[0][0]
            else:
                return None
                
        except Exception as e:
            print(f"Error rendering custom: {e}")
            return None


def create_interface():
    """Create Gradio interface for voice demo."""
    demo = VoiceDemo()
    
    with gr.Blocks(title="Voice Cloning Demo - Emotional TTS") as interface:
        gr.Markdown("""
        # 🎙️ Voice Cloning & Emotional TTS Demo
        
        Test voice cloning quality and explore emotional delivery variations.
        **Focus:** Voice cloning, emotion-based delivery, and pace control.
        
        ### Instructions:
        1. Upload a 6+ second voice sample (clear, single speaker)
        2. Click "Clone Voice"
        3. Test different emotions and paces with your text
        """)
        
        # Voice cloning section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 1️⃣ Clone Voice")
                voice_sample = gr.Audio(
                    label="Upload Voice Sample (6+ seconds recommended)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                clone_btn = gr.Button("🎭 Clone Voice", variant="primary")
                clone_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # Testing sections
        with gr.Tabs() as tabs:
            # Emotion variations tab
            with gr.Tab("😊 Emotion Variations"):
                gr.Markdown("""
                ### Test All Emotions
                Enter text and hear it rendered in all 7 emotions to compare emotional delivery.
                """)
                
                emotion_text = gr.Textbox(
                    label="Text to Render",
                    placeholder="You're in a big hurry.",
                    value="You're in a big hurry."
                )
                emotion_test_btn = gr.Button("🎬 Render All Emotions", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Joy**")
                        joy_audio = gr.Audio(label="", type="filepath")
                    with gr.Column():
                        gr.Markdown("**Anger**")
                        anger_audio = gr.Audio(label="", type="filepath")
                    with gr.Column():
                        gr.Markdown("**Sadness**")
                        sadness_audio = gr.Audio(label="", type="filepath")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Fear**")
                        fear_audio = gr.Audio(label="", type="filepath")
                    with gr.Column():
                        gr.Markdown("**Surprise**")
                        surprise_audio = gr.Audio(label="", type="filepath")
                    with gr.Column():
                        gr.Markdown("**Disgust**")
                        disgust_audio = gr.Audio(label="", type="filepath")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Neutral**")
                        neutral_audio = gr.Audio(label="", type="filepath")
            
            # Pace variations tab
            with gr.Tab("⏱️ Pace Control"):
                gr.Markdown("""
                ### Test Pacing Variations
                Compare slow, normal, and fast delivery of the same line.
                """)
                
                pace_text = gr.Textbox(
                    label="Text to Render",
                    placeholder="You know the score, pal.",
                    value="You know the score, pal."
                )
                pace_emotion = gr.Dropdown(
                    choices=['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral'],
                    value='neutral',
                    label="Emotion"
                )
                pace_test_btn = gr.Button("🎬 Render All Paces", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🐢 Slow (0.85x)**")
                        slow_audio = gr.Audio(label="", type="filepath")
                    with gr.Column():
                        gr.Markdown("**🚶 Normal (1.0x)**")
                        normal_audio = gr.Audio(label="", type="filepath")
                    with gr.Column():
                        gr.Markdown("**🏃 Fast (1.15x)**")
                        fast_audio = gr.Audio(label="", type="filepath")
            
            # Custom rendering tab
            with gr.Tab("🎨 Custom Rendering"):
                gr.Markdown("""
                ### Custom Emotion + Pace
                Full control over emotion and pacing for your text.
                """)
                
                custom_text = gr.Textbox(
                    label="Text to Render",
                    placeholder="Enter any dialogue...",
                    lines=3
                )
                
                with gr.Row():
                    custom_emotion = gr.Dropdown(
                        choices=['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral'],
                        value='neutral',
                        label="Emotion"
                    )
                    custom_pace = gr.Dropdown(
                        choices=[config.PACE_SLOW, config.PACE_NORMAL, config.PACE_FAST],
                        value=config.PACE_NORMAL,
                        label="Pace"
                    )
                
                custom_render_btn = gr.Button("🎬 Render", variant="primary")
                custom_audio = gr.Audio(label="Rendered Audio", type="filepath")
        
        # Connect buttons
        clone_btn.click(
            fn=demo.clone_voice,
            inputs=[voice_sample],
            outputs=[clone_status]
        )
        
        emotion_test_btn.click(
            fn=demo.test_emotion_variations,
            inputs=[emotion_text],
            outputs=[joy_audio, anger_audio, sadness_audio, fear_audio, surprise_audio, disgust_audio, neutral_audio]
        )
        
        pace_test_btn.click(
            fn=demo.test_pace_variations,
            inputs=[pace_text, pace_emotion],
            outputs=[slow_audio, normal_audio, fast_audio]
        )
        
        custom_render_btn.click(
            fn=demo.render_custom,
            inputs=[custom_text, custom_emotion, custom_pace],
            outputs=[custom_audio]
        )
        
        gr.Markdown("""
        ---
        **TTS Model:** Coqui XTTS-v2 (multilingual voice cloning)
        
        **Tips for Best Results:**
        - Voice sample: 6-10 seconds, clear audio, minimal background noise
        - Single speaker only
        - Natural speech patterns work best
        """)
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
