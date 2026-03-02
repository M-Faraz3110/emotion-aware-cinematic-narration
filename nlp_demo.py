"""
NLP Demo - Emotion & Pace Analysis for Screenplays
Standalone Gradio interface focused on screenplay analysis and visualization.
"""

import gradio as gr
import json
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple

from parser import ScreenplayParser
from nlp import NLPPipeline
import config


class NLPDemo:
    """Standalone NLP analysis demo with visualizations."""
    
    def __init__(self):
        """Initialize parser and NLP pipeline."""
        self.parser = ScreenplayParser()
        self.nlp = NLPPipeline()
    
    def analyze_screenplay(
        self, 
        screenplay_file
    ) -> Tuple[str, str, str, str, str, str]:
        """
        Analyze screenplay and return visualizations.
        
        Args:
            screenplay_file: Uploaded screenplay file
            
        Returns:
            Tuple of (JSON, emotion_chart, pace_chart, character_analysis, stats, line_breakdown)
        """
        try:
            # Read and parse screenplay
            with open(screenplay_file.name, 'r', encoding='utf-8') as f:
                screenplay_text = f.read()
            
            parsed = self.parser.parse(screenplay_text)
            
            if not parsed:
                return "No dialogue found in screenplay", "", "", "", "", ""
            
            # Run NLP analysis
            director_script = self.nlp.analyze(parsed)
            
            # Generate outputs
            json_output = json.dumps(director_script, indent=2)
            emotion_chart = self._create_emotion_chart(director_script)
            pace_chart = self._create_pace_chart(director_script)
            character_analysis = self._create_character_analysis(director_script)
            stats = self._create_statistics(director_script)
            line_breakdown = self._create_line_breakdown(director_script)
            
            return json_output, emotion_chart, pace_chart, character_analysis, stats, line_breakdown
            
        except Exception as e:
            error_msg = f"Error analyzing screenplay: {str(e)}"
            return error_msg, None, None, error_msg, error_msg, error_msg
    
    def _create_emotion_chart(self, director_script: List[Dict]) -> go.Figure:
        """Create emotion distribution chart."""
        emotions = [line['emotion'] for line in director_script]
        emotion_counts = Counter(emotions)
        
        # Sort by count
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [e[0] for e in sorted_emotions]
        values = [e[1] for e in sorted_emotions]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE'],
                text=values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Emotion Distribution Across Screenplay",
            xaxis_title="Emotion",
            yaxis_title="Number of Lines",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def _create_pace_chart(self, director_script: List[Dict]) -> go.Figure:
        """Create pace distribution chart."""
        paces = [line['pace'] for line in director_script]
        pace_counts = Counter(paces)
        
        # Define pace order
        pace_order = [config.PACE_SLOW, config.PACE_NORMAL, config.PACE_FAST]
        labels = [p for p in pace_order if p in pace_counts]
        values = [pace_counts[p] for p in labels]
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=['#3498DB', '#2ECC71', '#E74C3C'],
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title="Pacing Distribution",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def _create_character_analysis(self, director_script: List[Dict]) -> str:
        """Analyze characters and their emotional patterns."""
        # Group by character
        character_data = {}
        
        for line in director_script:
            char = line['speaker']
            emotion = line['emotion']
            
            if char not in character_data:
                character_data[char] = {
                    'lines': 0,
                    'emotions': []
                }
            
            character_data[char]['lines'] += 1
            character_data[char]['emotions'].append(emotion)
        
        # Build analysis text
        analysis = "# Character Analysis\n\n"
        
        # Sort by line count
        sorted_chars = sorted(character_data.items(), key=lambda x: x[1]['lines'], reverse=True)
        
        for char, data in sorted_chars:
            emotion_counts = Counter(data['emotions'])
            dominant_emotion = emotion_counts.most_common(1)[0]
            
            analysis += f"## {char}\n"
            analysis += f"- **Lines:** {data['lines']}\n"
            analysis += f"- **Dominant Emotion:** {dominant_emotion[0]} ({dominant_emotion[1]} lines, {dominant_emotion[1]/data['lines']*100:.1f}%)\n"
            analysis += f"- **Emotional Range:** {', '.join([f'{e} ({c})' for e, c in emotion_counts.most_common()])}\n\n"
        
        return analysis
    
    def _create_statistics(self, director_script: List[Dict]) -> str:
        """Create overall statistics summary."""
        total_lines = len(director_script)
        unique_characters = len(set(line['speaker'] for line in director_script))
        unique_emotions = len(set(line['emotion'] for line in director_script))
        
        emotions = [line['emotion'] for line in director_script]
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0]
        
        paces = [line['pace'] for line in director_script]
        pace_counts = Counter(paces)
        
        stats = f"""# Screenplay Statistics

## Overview
- **Total Dialogue Lines:** {total_lines}
- **Unique Characters:** {unique_characters}
- **Emotions Detected:** {unique_emotions}
- **Dominant Emotion:** {dominant_emotion[0]} ({dominant_emotion[1]} lines, {dominant_emotion[1]/total_lines*100:.1f}%)

## Pacing Breakdown
- **Slow:** {pace_counts.get(config.PACE_SLOW, 0)} lines ({pace_counts.get(config.PACE_SLOW, 0)/total_lines*100:.1f}%)
- **Normal:** {pace_counts.get(config.PACE_NORMAL, 0)} lines ({pace_counts.get(config.PACE_NORMAL, 0)/total_lines*100:.1f}%)
- **Fast:** {pace_counts.get(config.PACE_FAST, 0)} lines ({pace_counts.get(config.PACE_FAST, 0)/total_lines*100:.1f}%)

## Emotional Diversity
{self._calculate_emotional_diversity(emotions)}
"""
        
        return stats
    
    def _calculate_emotional_diversity(self, emotions: List[str]) -> str:
        """Calculate emotional diversity score."""
        emotion_counts = Counter(emotions)
        total = len(emotions)
        
        # Shannon entropy for diversity
        import math
        entropy = -sum((count/total) * math.log2(count/total) for count in emotion_counts.values())
        max_entropy = math.log2(len(emotion_counts))
        diversity_score = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
        
        interpretation = "high" if diversity_score > 70 else "moderate" if diversity_score > 40 else "low"
        
        return f"- **Diversity Score:** {diversity_score:.1f}% ({interpretation} emotional variation)"
    
    def _create_line_breakdown(self, director_script: List[Dict]) -> str:
        """Create line-by-line breakdown with all NLP analysis details."""
        breakdown = "# Line-by-Line Analysis\n\n"
        
        for line_data in director_script:
            breakdown += f"---\n\n"
            breakdown += f"**Line {line_data['line_number']}** - {line_data['speaker']}\n\n"
            breakdown += f"> {line_data['line']}\n\n"
            
            if line_data.get('parenthetical'):
                breakdown += f"*({line_data['parenthetical']})*\n\n"
            
            breakdown += f"- **Emotion:** {line_data['emotion']} (confidence: {line_data['emotion_confidence']:.2%})\n"
            breakdown += f"- **Intensity:** {line_data['intensity']:.3f}\n"
            breakdown += f"- **Pace:** {line_data['pace']} (score: {line_data['pace_score']:.3f})\n"
            breakdown += f"- **Pause Before:** {line_data['pause_before']:.2f}s\n\n"
        
        return breakdown


def create_interface():
    """Create Gradio interface for NLP demo."""
    demo = NLPDemo()
    
    # Example screenplay in proper Fountain format
    example_screenplay = """INT. BLADE RUNNER HEADQUARTERS - NIGHT

DECKARD enters the dimly lit office. GAFF sits behind a desk.

GAFF
You're in a big hurry. Sit down.

DECKARD
(frustrated)
I just came back for some information.

GAFF
(calm)
You know the score, pal. If you're not a cop, you're little people.

DECKARD
(angry)
No choice, huh?

GAFF
(threatening)
No choice at all.
"""
    
    with gr.Blocks(title="NLP Analysis Demo - Emotion & Pace Detection") as interface:
        gr.Markdown("""
        # 🎭 Screenplay NLP Analysis Demo
        
        Upload a screenplay to analyze emotional content, pacing patterns, and character dynamics.
        **Focus:** Emotion detection, pace analysis, character insights, and statistical visualizations.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                screenplay_input = gr.File(
                    label="📄 Upload Screenplay",
                    file_types=[".txt", ".fountain"]
                )
                
                analyze_btn = gr.Button("🔍 Analyze Screenplay", variant="primary")
                
                gr.Markdown("""
                ### Fountain Format:
                ```
                INT. LOCATION - TIME
                
                CHARACTER NAME
                (parenthetical)
                Dialogue text here.
                
                CHARACTER NAME
                More dialogue.
                ```
                """)
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📊 Statistics"):
                    stats_output = gr.Markdown()
                
                with gr.Tab("� Line-by-Line"):
                    line_breakdown_output = gr.Markdown()
                
                with gr.Tab("😊 Emotion Analysis"):
                    emotion_chart = gr.Plot()
                
                with gr.Tab("⏱️ Pace Analysis"):
                    pace_chart = gr.Plot()
                
                with gr.Tab("👥 Character Analysis"):
                    character_output = gr.Markdown()
                
                with gr.Tab("📋 Director's Script JSON"):
                    json_output = gr.Code(language="json")
        
        # Connect button
        analyze_btn.click(
            fn=demo.analyze_screenplay,
            inputs=[screenplay_input],
            outputs=[json_output, emotion_chart, pace_chart, character_output, stats_output, line_breakdown_output]
        )
        
        gr.Markdown("""
        ---
        **Models Used:**
        - Emotion: `j-hartmann/emotion-english-distilroberta-base`
        - Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
        - NLP: spaCy `en_core_web_trf`
        """)
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
