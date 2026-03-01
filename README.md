# 🎬 Emotion-Aware Cinematic Narration Engine

An advanced **AI-powered narration system** that analyzes screenplay dialogue using Natural Language Processing to understand emotion, intensity, and pacing — then narrates it back in a cloned voice with cinematic delivery.

Built as a **research and portfolio project** showcasing state-of-the-art NLP, voice cloning, and audio synthesis techniques.

---

## 🚀 Live Demo

**[Open in Google Colab](https://colab.research.google.com/)** *(Replace with your actual Colab notebook link)*

> **Requirements:** Google Colab Pro with A100 GPU recommended (T4 works but slower)

---

## ✨ Features

- **🧠 Contextual Emotion Analysis**: Dual-model emotion detection with surrounding context awareness
- **📊 Multi-Signal Pacing Engine**: Analyzes punctuation, parentheticals, emotion, and intensity to determine delivery speed
- **⏸️ Intelligent Pause Insertion**: Clause-boundary detection using dependency parsing
- **🎤 Voice Cloning**: High-quality voice replication using Coqui XTTS-v2 (22.05kHz, 24-bit)
- **🎙️ Dynamic Delivery Control**: Time-stretching for pace adjustment without pitch distortion
- **🔊 Professional Audio Assembly**: Crossfaded segments with emotion-driven pauses
- **📝 Multiple Format Support**: Fountain, plain dialogue, raw text with auto-detection
- **🌐 Web Interface**: Interactive Gradio UI with real-time progress tracking

---

## 🏗️ Architecture

The engine operates as a **4-stage pipeline** that communicates via a structured **Director's Script JSON**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INPUT: Screenplay + Voice Sample                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │  STAGE 1: Screenplay Parser     │
                │  • Format detection             │
                │  • Dialogue extraction          │
                │  • Parenthetical parsing        │
                └────────────────┬────────────────┘
                                 │
                     Parsed Dialogue List
                                 │
                ┌────────────────▼────────────────┐
                │  STAGE 2: NLP Pipeline          │
                │  • Contextual emotion (dual)    │
                │  • Intensity scoring            │
                │  • 4-signal pace inference      │
                │  • 4-signal pause calculation   │
                └────────────────┬────────────────┘
                                 │
                    Director's Script JSON
                                 │
                ┌────────────────▼────────────────┐
                │  STAGE 3: Voice Pipeline        │
                │  • Voice sample preprocessing   │
                │  • XTTS-v2 cloning              │
                │  • Per-line TTS rendering       │
                │  • Pace control (time-stretch)  │
                └────────────────┬────────────────┘
                                 │
                   Audio Files + Metadata
                                 │
                ┌────────────────▼────────────────┐
                │  STAGE 4: Audio Assembly        │
                │  • Pause insertion              │
                │  • Crossfade transitions        │
                │  • Normalization                │
                └────────────────┬────────────────┘
                                 │
           ┌─────────────────────▼─────────────────────┐
           │  OUTPUT: Cinematic Narration (WAV)        │
           └───────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **NLP Framework** | spaCy (en_core_web_trf) | Dependency parsing, clause detection |
| **Emotion Detection** | DistilRoBERTa (j-hartmann/emotion) | 7-class emotion classification |
| **Intensity Scoring** | RoBERTa (cardiffnlp/sentiment) | Sentiment strength analysis |
| **Voice Cloning** | Coqui TTS XTTS-v2 | Zero-shot multilingual voice synthesis |
| **Audio Processing** | librosa, pydub, soundfile | Time-stretching, crossfading, assembly |
| **Screenplay Parsing** | fountain (Python library) | Fountain format support |
| **Web Interface** | Gradio 4.0+ | Interactive UI with public URLs |
| **Compute** | Google Colab (A100 GPU) | Model inference acceleration |
| **Framework** | PyTorch, Transformers | Deep learning backend |

---

## 🧠 How It Works: The NLP Pacing Engine

### 1. Contextual Emotion Analysis

Unlike single-line emotion detection, this engine uses a **dual-inference approach**:

**Line Inference (60% weight):**
```
Input: "I can't believe you did this."
Output: {'anger': 0.78, 'sadness': 0.15, 'neutral': 0.07}
```

**Context Window Inference (40% weight):**
```
Input:
  MARY (2 lines before): I had no choice.
  JOHN (1 line before): There's always a choice.
  MARY (current): I can't believe you did this.
  JOHN (1 line after): Neither can I.
  MARY (2 line after): What happens now?

Output: {'sadness': 0.62, 'anger': 0.25, 'fear': 0.08, 'neutral': 0.05}
```

**Blended Result (60/40):**
```python
blended = {
    'anger': (0.78 * 0.6) + (0.25 * 0.4) = 0.568,
    'sadness': (0.15 * 0.6) + (0.62 * 0.4) = 0.338,
    'neutral': (0.07 * 0.6) + (0.05 * 0.4) = 0.062
}
Final Emotion: anger (0.568)
```

This prevents context loss for short lines like "No." or "What?" that are ambiguous in isolation.

---

### 2. Multi-Signal Pace Inference

Pace is determined by **4 independent signals** that vote on delivery speed:

| Signal | Logic | Example |
|--------|-------|---------|
| **Parenthetical** | Direct override | `(quickly)` → fast, `(slowly)` → slow |
| **Punctuation** | `!`, `?`, `...`, `—` patterns | `"What?! No!"` → fast |
| **Emotion-Based** | High arousal emotions | `anger`, `fear` → fast<br>`sadness` → slow |
| **Intensity Scaling** | Emotion strength | High intensity → faster delivery |

**Voting System:**
- Each signal casts a vote: `[-1, 0, +1]` (slow, normal, fast)
- Votes are summed: `total_vote = sum(signals)`
- Thresholds: `total ≥ 2` → fast, `total ≤ -2` → slow, else normal

**Example:**
```
Line: "GET OUT!" (parenthetical: None)

Signal votes:
  Parenthetical: 0 (no override)
  Punctuation:   +1 (exclamation + caps)
  Emotion:       +1 (anger detected)
  Intensity:     +1 (high intensity: 0.89)
  
Total: +3 → FAST pace
```

---

### 3. Intelligent Pause Calculation

Pauses are calculated **additively** from 4 signals with a safety cap:

| Signal | Contribution | Logic |
|--------|-------------|-------|
| **Base Pause** | +0.3s | Every line starts with a base pause |
| **Parenthetical** | +0.0s to +1.0s | `(pause)` = +1.0s, `(quickly)` = -0.2s |
| **Punctuation** | +0.1s to +0.5s | `.` = +0.1s, `...` = +0.5s, `—` = +0.3s |
| **Clause Boundaries** | +0.1s per clause | spaCy dependency parsing detects clauses |
| **Emotion-Intensity** | +0.0s to +0.4s | High sadness + high intensity = +0.4s |

**Formula:**
```python
pause = min(
    max(
        BASE_PAUSE + parenthetical + punctuation + clauses + emotion,
        MIN_PAUSE  # 0.1s
    ),
    MAX_PAUSE  # 2.0s
)
```

**Example:**
```
Line: "I never wanted this... but here we are."
Emotion: sadness (0.72)

Calculations:
  Base:           +0.30s
  Parenthetical:  +0.00s (none)
  Punctuation:    +0.50s (...) + 0.10s (.)
  Clauses:        +0.10s (1 clause: "but here we are")
  Emotion:        +0.29s (sadness scaling)
  
Total: 1.29s pause before line
```

---

### 4. Pace Control via Time-Stretching

Instead of altering TTS model parameters (which can introduce artifacts), pace is controlled **post-generation**:

```python
# librosa preserves pitch while changing speed
stretched_audio = librosa.effects.time_stretch(
    audio,
    rate=1.15  # 15% faster for "fast" pace
)
```

**Multipliers:**
- **slow**: 0.85× (15% slower)
- **normal**: 1.0× (no change)
- **fast**: 1.15× (15% faster)

---

## 📁 Project Structure

```
Emotion-Aware-Cinematic-Narration-Engine/
│
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
│
├── config.py                     # Centralized configuration
├── utils.py                      # Environment setup, helpers
│
├── parser.py                     # Stage 1: Screenplay parser
├── nlp.py                        # Stage 2: NLP pipeline
├── voice.py                      # Stage 3: Voice cloning + rendering
├── audio_assembler.py            # Stage 3: Audio assembly
├── app.py                        # Stage 4: Gradio web interface
│
├── colab_notebook.ipynb          # Google Colab launcher
│
└── examples/                     # Sample screenplays
    ├── blade_runner_tears_in_rain.txt
    ├── dark_knight_interrogation.txt
    ├── no_country_opening.txt
    └── plain_dialogue_sample.txt
```

---

## 🚀 Setup & Usage

### Option 1: Google Colab (Recommended)

1. **Open the Colab notebook**: [Link to your Colab notebook]
2. **Enable GPU**: Runtime → Change runtime type → A100 GPU
3. **Run all cells** (Ctrl+F9)
4. Wait for models to download (~10 minutes first time)
5. **Use the Gradio interface** that appears

### Option 2: Local Setup

**Requirements:**
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM for XTTS-v2)
- ~20GB disk space for models

**Installation:**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Emotion-Aware-Cinematic-Narration-Engine.git
cd Emotion-Aware-Cinematic-Narration-Engine

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_trf

# Launch Gradio interface
python app.py
```

**Usage:**
1. Open browser to `http://localhost:7860`
2. Upload a 6+ second voice sample (or record via microphone)
3. Paste your screenplay (or use example buttons)
4. Click "Generate Narration"
5. Wait 2-5 minutes for processing
6. Download your narration from the Results tab

---

## 📊 Performance

**Processing Times (on Colab A100):**
- Voice cloning: ~30 seconds (one-time per voice)
- Per-line rendering: ~3-5 seconds
- 100-line screenplay: ~5-8 minutes total

**Model Sizes:**
- XTTS-v2: ~6.4GB
- spaCy en_core_web_trf: ~500MB
- Emotion RoBERTa: ~350MB
- Sentiment RoBERTa: ~500MB

---

## 🔬 Research & Technical Details

### Director's Script Format

Each processed line is enriched with metadata:

```json
{
  "line_number": 5,
  "speaker": "MARY",
  "line": "I can't believe you did this.",
  "parenthetical": "quietly",
  "emotion": "sadness",
  "intensity": 0.72,
  "pace": "slow",
  "pause_before": 1.05,
  "original_text": "MARY: (quietly) I can't believe you did this."
}
```

### Voice Sample Preprocessing

Voice samples undergo 3-stage preprocessing before cloning:

1. **Silence Trimming**: `librosa.effects.trim(top_db=20)` removes leading/trailing silence
2. **Normalization**: Peak normalization to -20 dBFS for consistent volume
3. **Resampling**: Converted to 22,050 Hz (XTTS-v2 native sample rate)

### Audio Assembly

Final narration uses **crossfade transitions** to eliminate clicks:

```
Segment A: --------[fade out]
Segment B:         [fade in]--------
           --------XXXXXXXXXX--------  (overlay)
```

- Fade duration: 20ms (imperceptible but eliminates clicks)
- Normalization: -3dB headroom to prevent clipping

---

## 🎯 Future Enhancements

- [ ] **Batch Processing**: Parallel rendering for multiple lines
- [ ] **Background Music Integration**: Dynamic scoring based on emotion arcs
- [ ] **Multi-Voice Support**: Different clones for different characters
- [ ] **Emotion Trajectory Smoothing**: Temporal consistency across scenes
- [ ] **Fine-tuned Pace Models**: Train custom pace classifier on audiobook data
- [ ] **Real-time Streaming**: WebRTC-based live narration
- [ ] **Character Voice Matching**: Suggest voices based on character descriptions

---

## 📸 Screenshots

*Add screenshots of:*
1. Gradio interface (Input tab)
2. Processing status log
3. Director's Script JSON viewer
4. Example narration waveform

---

## 🤝 Contributing

This is a research/portfolio project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Third-Party Model Licenses:**
- **Coqui TTS (XTTS-v2)**: MPL 2.0 License
- **spaCy**: MIT License
- **Transformers Models**: Check individual model cards on HuggingFace

---

## 🙏 Acknowledgments

- **Coqui.ai** for the incredible XTTS-v2 voice cloning model
- **Explosion.ai** for spaCy's powerful NLP toolkit
- **HuggingFace** for hosting emotion and sentiment models
- **Gradio** for making ML interfaces effortless

---

## 📧 Contact

**Your Name** - [@YourTwitter](https://twitter.com/yourhandle) - your.email@example.com

Project Link: [https://github.com/YOUR_USERNAME/Emotion-Aware-Cinematic-Narration-Engine](https://github.com/YOUR_USERNAME/Emotion-Aware-Cinematic-Narration-Engine)

---

<div align="center">
  
**Built with ❤️ and 🤖 for the intersection of NLP and creative storytelling**

*If you found this helpful, consider giving it a ⭐ on GitHub!*

</div>
