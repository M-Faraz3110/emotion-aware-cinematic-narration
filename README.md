# 🎬 Emotion-Aware Cinematic Narration Engine

An advanced **AI-powered narration system** that analyzes screenplay dialogue using Natural Language Processing to understand **multi-emotion blending**, intensity, and pacing — then narrates it back in a cloned voice with **emotionally-modulated prosody** and cinematic delivery.

---

## 🚀 Live Demo

**[Open in Google Colab](https://colab.research.google.com/)** *(Replace with your actual Colab notebook link)*

> **Requirements:** Google Colab with T4 GPU or better (A100 recommended for faster processing)

---

## ✨ Features

### Core Engine
- **🎭 Multi-Emotion Blending**: Analyzes and blends top 3 emotions per line (nothing is purely one emotion)
- **🧠 Contextual Emotion Analysis**: Dual-model emotion detection with surrounding context awareness
- **🎵 Emotional Prosody**: Pitch shifting (±0.8 semitones) and energy modulation (0.80-1.30x volume) based on emotion
- **📊 Emotion-Weighted Pacing**: All detected emotions contribute to pace calculation, not just the dominant one
- **💬 Delivery Emotion Boosts**: Parentheticals like `(laughing)` boost specific emotions in the blend
- **⏸️ Intelligent Pause Insertion**: 4-signal pause calculation with clause-boundary detection
- **🎤 Voice Cloning**: High-quality voice replication using Coqui XTTS-v2 (22.05kHz, 24-bit)
- **🎙️ Dynamic Delivery Control**: Time-stretching for pace adjustment without pitch distortion
- **🔊 Professional Audio Assembly**: Crossfaded segments with emotion-driven pauses

### Interfaces
- **🌐 Full Application**: Complete screenplay-to-narration pipeline with Gradio web UI
- **🧪 NLP Demo**: Standalone emotion analysis and visualization interface
- **🎙️ Voice Demo**: Standalone voice cloning testing with emotion/pace variations
- **📝 Multiple Format Support**: Fountain, plain dialogue, raw text with auto-detection
- **📊 Real-time Progress**: Live status tracking through all pipeline stages

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
                │  • Multi-emotion blending       │
                │  • Delivery emotion boosts      │
                │  • Intensity scoring            │
                │  • Emotion-weighted pace        │
                │  • 4-signal pause calculation   │
                └────────────────┬────────────────┘
                                 │
                    Director's Script JSON
                    (with emotion_blend array)
                                 │
                ┌────────────────▼────────────────┐
                │  STAGE 3: Voice Pipeline        │
                │  • Voice sample preprocessing   │
                │  • XTTS-v2 cloning              │
                │  • Per-line TTS rendering       │
                │  • Emotional prosody:           │
                │    - Pitch shifting             │
                │    - Energy modulation          │
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

| Component              | Technology                         | Purpose                                |
| ---------------------- | ---------------------------------- | -------------------------------------- |
| **NLP Framework**      | spaCy (en_core_web_sm)             | Dependency parsing, clause detection   |
| **Emotion Detection**  | DistilRoBERTa (j-hartmann/emotion) | 7-class emotion classification         |
| **Intensity Scoring**  | RoBERTa (cardiffnlp/sentiment)     | Sentiment strength analysis            |
| **Voice Cloning**      | Coqui TTS XTTS-v2                  | Zero-shot multilingual voice synthesis |
| **Audio Processing**   | librosa, pydub, soundfile          | Time-stretching, crossfading, assembly |
| **Emotional Prosody**  | librosa (pitch_shift, normalize)   | Pitch and energy modulation            |
| **Screenplay Parsing** | fountain (Python library)          | Fountain format support                |
| **Web Interface**      | Gradio 4.0+                        | Interactive UI with public URLs        |
| **Compute**            | Google Colab (T4 GPU or better)    | Model inference acceleration           |
| **Framework**          | PyTorch, Transformers (<5.0.0)     | Deep learning backend                  |

**Critical Dependency Notes:**
- **transformers <5.0.0**: TTS library breaks with transformers 5.x+ due to API changes
  - ✅ Use: `transformers==4.45.2` (tested and working)
  - ❌ Avoid: `transformers>=5.0.0` (breaks TTS)
- **CUDA compatibility**: Ensure PyTorch CUDA version matches your system
  - Example: CUDA 11.8 → `torch==2.1.0+cu118`
- **Python version**: 3.10 or 3.11 recommended (3.12 may have TTS compatibility issues)

---

## Some notes for personal reference 
Stuff I learnt/polished:
 - We're using transformers here (DistilRoBERTa & RoBERTa-base) for emotion classifying and sentiment clasifying respectively. Self-attention is huge here and why it works like it should. 
 - Step by step process of the transformer model
   1. Tokenization
   2. Embeddings (Each token has an embedding we get after training on billions of words. We use this along with positional encoding to get a final embedding)
   3. Classification based on the 50K+ examples used in training 
   4. We basically get a list of emotions and their scores (or sentiment in the case of the sentiment classifier)
 - SpaCy was probably overkill. We just needed it for finding out pauses between dialogue. 
 - For the emotion and sentiment classifiers, we set the pipelines as:

```python
self.emotion_classifier = pipeline(
    "text-classification",
    model=config.EMOTION_MODEL,  # "j-hartmann/emotion-english-distilroberta-base"
    device=0 if self.device == "cuda" else -1,
    top_k=None  # KEY: Returns ALL emotion probabilities, not just top-1
)
```

and we call it with:

```python
line_results = self.emotion_classifier(line)[0]
```
- RoBERTa uses BPE for tokenization. 
- DistilRoBERTa seemed to be the best for emotion dectection, and RoBERTa seemed the best for sentiment detection. 

---

## 🧠 How It Works: The NLP Pacing Engine

### 1. Multi-Emotion Blending System

**Philosophy:** Nothing is purely one emotion. Real dialogue contains blends of multiple emotions simultaneously.

The engine detects emotions from **4 independent sources** and blends them:

| Source                          | Weight | Purpose                           |
| ------------------------------- | ------ | --------------------------------- |
| **Line-only model**             | 25%    | Direct emotion in the text        |
| **Context window model**        | 30%    | Emotion from surrounding dialogue |
| **Intensity model sentiment**   | 15%    | Emotional strength/valence        |
| **Context intensity sentiment** | 30%    | Contextual emotional arc          |

**Example:**
```
Line: "(sobbing) I can't do this anymore."

Source contributions:
  Line-only:     {'sadness': 0.85, 'fear': 0.10, 'neutral': 0.05}  × 0.25
  Context:       {'sadness': 0.60, 'fear': 0.25, 'surprise': 0.15} × 0.30
  Intensity:     {'sadness': 0.75, 'fear': 0.15, 'anger': 0.10}    × 0.15
  Context Int:   {'sadness': 0.70, 'fear': 0.20, 'neutral': 0.10}  × 0.30

Blended scores:
  sadness: 0.71
  fear: 0.18
  surprise: 0.05
  neutral: 0.04
  anger: 0.02

Top 3 emotion_blend:
  [
    {'emotion': 'sadness', 'score': 0.71},
    {'emotion': 'fear', 'score': 0.18},
    {'emotion': 'surprise', 'score': 0.05}
  ]
```

### 2. Delivery Emotion Boosts

Parentheticals can **boost specific emotions** in the blend to fix context misreads:

```python
DELIVERY_EMOTION_BOOSTS = {
    '(laughing)': {'joy': +0.40},      # Heavily boost joy
    '(crying)': {'sadness': +0.40},
    '(angry)': {'anger': +0.40},
    '(screaming)': {'fear': +0.35},
    '(whispering)': {'fear': +0.25, 'sadness': +0.15}
}
```

**Example:**
```
Line: "(laughing) You're in a big hurry."

Before boost:
  {'fear': 0.42, 'neutral': 0.35, 'joy': 0.15}  ← Context analysis backfired!

After boost (+0.40 to joy):
  {'joy': 0.55, 'fear': 0.28, 'neutral': 0.17}  ← Corrected!
```

### 3. Contextual Emotion Analysis

Unlike single-line emotion detection, this engine uses a **dual-inference approach**:

**Line Inference (25% weight):**
```
Input: "I can't believe you did this."
Output: {'anger': 0.78, 'sadness': 0.15, 'neutral': 0.07}
```

**Context Window Inference (30% weight):**
```
Input:
  MARY (2 lines before): I had no choice.
  JOHN (1 line before): There's always a choice.
  MARY (current): I can't believe you did this.
  JOHN (1 line after): Neither can I.
  MARY (2 line after): What happens now?

Output: {'sadness': 0.62, 'anger': 0.25, 'fear': 0.08, 'neutral': 0.05}
```

This prevents context loss for short lines like "No." or "What?" that are ambiguous in isolation.

---

### 4. Emotion-Weighted Pace Inference

Instead of using only the dominant emotion, **all emotions in the blend contribute** to pace calculation:

```python
EMOTION_PACE_CONTRIBUTIONS = {
    'anger':    +0.20,  # Faster
    'fear':     +0.20,  # Faster
    'joy':      +0.15,  # Slightly faster
    'surprise': +0.10,  # Slightly faster
    'sadness':  -0.20,  # Slower
    'disgust':  -0.15,  # Slower
    'neutral':   0.00   # No effect
}
```

**Calculation:**
```python
pace_score = 0.0
for emotion_entry in emotion_blend:
    emotion_name = emotion_entry['emotion']
    emotion_score = emotion_entry['score']
    contribution = EMOTION_PACE_CONTRIBUTIONS[emotion_name]
    pace_score += contribution * emotion_score

# Example:
# sadness (0.45) + neutral (0.28) + fear (0.15) + anger (0.12)
# = (-0.2×0.45) + (0×0.28) + (0.2×0.15) + (0.2×0.12)
# = -0.09 + 0 + 0.03 + 0.024
# = -0.036 → SLOW pace
```

Additional pace signals (parentheticals, punctuation) can override this calculation.

---

### 5. Emotional Prosody Modulation

To create emotionally distinct voices, audio is post-processed with **pitch shifting** and **energy modulation**:

#### Pitch Shifting (Subtle: 0.3-0.8 semitones)
```python
EMOTION_PITCH_SHIFTS = {
    'joy':      +0.5,   # Slightly brighter
    'surprise': +0.8,   # Higher pitched
    'fear':     +0.8,   # Tense/strained
    'anger':    -0.3,   # Slightly gruff
    'sadness':  -0.5,   # Lower/subdued
    'disgust':  -0.3,   # Slightly contemptuous
    'neutral':   0.0    # No change
}
```

Applied using librosa: `librosa.effects.pitch_shift(audio, sr=22050, n_steps=shift)`

**Note:** Kept subtle (under 1 semitone) to avoid robotic artifacts while preserving voice naturalness.

#### Energy Modulation (Volume: 0.80x - 1.30x)
```python
EMOTION_ENERGY_MODS = {
    'joy':      1.15,   # 15% louder
    'anger':    1.30,   # 30% louder (forceful)
    'fear':     1.20,   # 20% louder (panicked)
    'surprise': 1.18,   # 18% louder
    'sadness':  0.80,   # 20% quieter (withdrawn)
    'disgust':  0.85,   # 15% quieter
    'neutral':  1.0     # No change
}
```

**Processing Order:**
1. TTS generates base audio
2. Apply pitch shift (if emotion != neutral)
3. Apply energy modulation (volume * multiplier)
4. Normalize to prevent clipping (95% headroom)
5. Apply pace control (time-stretch)
6. Save processed audio

---

### 6. Multi-Signal Pace Inference (Legacy Voting System)

Pace is determined by **4 independent signals** that vote on delivery speed:

| Signal                | Logic                         | Example                                    |
| --------------------- | ----------------------------- | ------------------------------------------ |
| **Parenthetical**     | Direct override               | `(quickly)` → fast, `(slowly)` → slow      |
| **Punctuation**       | `!`, `?`, `...`, `—` patterns | `"What?! No!"` → fast                      |
| **Emotion-Based**     | High arousal emotions         | `anger`, `fear` → fast<br>`sadness` → slow |
| **Intensity Scaling** | Emotion strength              | High intensity → faster delivery           |

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

### 7. Intelligent Pause Calculation

Pauses are calculated **additively** from 4 signals with a safety cap:

| Signal                | Contribution     | Logic                                    |
| --------------------- | ---------------- | ---------------------------------------- |
| **Base Pause**        | +0.3s            | Every line starts with a base pause      |
| **Parenthetical**     | +0.0s to +1.0s   | `(pause)` = +1.0s, `(quickly)` = -0.2s   |
| **Punctuation**       | +0.1s to +0.5s   | `.` = +0.1s, `...` = +0.5s, `—` = +0.3s  |
| **Clause Boundaries** | +0.1s per clause | spaCy dependency parsing detects clauses |
| **Emotion-Intensity** | +0.0s to +0.4s   | High sadness + high intensity = +0.4s    |

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

### 8. Pace Control via Time-Stretching

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
├── nlp.py                        # Stage 2: NLP pipeline (multi-emotion blending)
├── voice.py                      # Stage 3: Voice cloning + rendering + prosody
├── audio_assembler.py            # Stage 4: Audio assembly with crossfades
│
├── app.py                        # Full Gradio application interface
├── nlp_demo.py                   # Standalone NLP analysis demo
├── voice_demo.py                 # Standalone voice cloning demo
│
├── colab_notebook.ipynb          # Google Colab launcher
│
├── test_dark_knight.py           # Example: Dark Knight scene test
└── examples/                     # Sample screenplays
    ├── blade_runner_tears_in_rain.txt
    ├── dark_knight_interrogation.txt
    ├── no_country_opening.txt
    └── plain_dialogue_sample.txt
```

---

## 🎛️ Standalone Demos

The system includes two standalone demo interfaces for testing individual components:

### NLP Demo (`nlp_demo.py`)

**Purpose:** Test the multi-emotion blending system without voice synthesis

**Features:**
- Upload screenplay files (`.txt`, `.fountain`)
- **Emotion Distribution Chart:** Visualize emotion frequencies across the entire script
- **Pace Distribution Chart:** See how pacing varies throughout the scene
- **Character Analysis:** Breakdown of emotions per character
- **Line-by-Line Table:** Detailed view with emotion blends, intensity, pace, pauses
- **Export Director's Script:** Download full JSON for external processing

**Use Cases:**
- Validate emotion detection accuracy before rendering
- Analyze screenplay pacing patterns
- Test multi-emotion blending logic
- Debug NLP pipeline issues

**Example Output:**
```
Line 42: "I can't believe you did this."
Emotions: sadness (0.71), fear (0.18), surprise (0.05)
Pace: slow (-0.12 from emotions)
Pause: 1.05s
```

### Voice Demo (`voice_demo.py`)

**Purpose:** Test voice cloning and emotional prosody without full screenplay processing

**Features:**
- Upload voice sample (6+ seconds)
- **Emotion Variation Test:** Render same text with all 7 emotions
  - Compare joy vs anger vs sadness vs fear vs disgust vs surprise vs neutral
  - Hear emotional prosody effects (pitch + volume modulation)
- **Pace Variation Test:** Render at slow / normal / fast speeds
- **Custom Input:** Enter any text and choose emotion + pace
- **A/B Testing:** Compare different emotional deliveries side-by-side

**Use Cases:**
- Validate voice clone quality before processing full screenplay
- Test if emotional prosody sounds natural (not robotic)
- Experiment with different emotion combinations
- Debug voice rendering issues

**Example Tests:**
```
Text: "This is a test of the emotional prosody system."

Emotion Tests:
  joy      → +0.5 semitones pitch, 1.10× volume
  anger    → -0.3 semitones pitch, 1.30× volume
  sadness  → -0.5 semitones pitch, 0.80× volume
  
Pace Tests:
  slow     → 0.85× speed
  normal   → 1.00× speed
  fast     → 1.15× speed
```

### When to Use Each Demo

| Scenario                                | Use This Demo  |
| --------------------------------------- | -------------- |
| "Are emotions detected correctly?"      | **NLP Demo**   |
| "Is pacing calculation working?"        | **NLP Demo**   |
| "Does the voice clone sound good?"      | **Voice Demo** |
| "Is emotional prosody too robotic?"     | **Voice Demo** |
| "Full screenplay-to-narration pipeline" | **Main App**   |

---

## 🚀 Setup & Usage

### Option 1: Google Colab (Recommended)

1. **Open the Colab notebook**: [Link to your Colab notebook]
2. **Enable GPU**: Runtime → Change runtime type → T4 GPU (or better)
3. **Run all cells** (Ctrl+F9)
4. Wait for models to download (~10 minutes first time)
5. **Choose an interface**:
   - **Full App**: Complete screenplay-to-narration pipeline
   - **NLP Demo**: Test emotion analysis and visualizations
   - **Voice Demo**: Test voice cloning and emotional delivery

**Important Compatibility Note:**
```python
# For Google Colab with TTS library, use:
pip install transformers==4.45.2  # NOT 5.x - breaks TTS compatibility
```

See troubleshooting section below for details.

### Option 2: Local Setup

**Requirements:**
- Python 3.10 or 3.11 (3.12 may have compatibility issues with TTS)
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
python -m spacy download en_core_web_sm

# Launch interface (choose one)
python app.py          # Full application
python nlp_demo.py     # NLP analysis only
python voice_demo.py   # Voice cloning only
```

### Usage: Full Application

1. Open browser to `http://localhost:7860`
2. Upload a 6+ second voice sample (or record via microphone)
3. Upload screenplay file OR paste text directly
4. Click "Generate Narration"
5. Wait 2-5 minutes for processing
6. Download your narration from the Results tab

### Usage: NLP Demo

1. Upload screenplay file
2. View emotion distribution charts
3. Analyze pace patterns
4. Export Director's Script JSON
5. See per-line emotion breakdowns with multi-emotion blends

### Usage: Voice Demo

1. Upload voice sample (6+ seconds)
2. Test emotion variations (joy, anger, sadness, etc.)
3. Test pace variations (slow, normal, fast)
4. Test custom combinations
5. Hear emotional prosody effects (pitch + volume modulation)

---

## � Troubleshooting

### "TTS breaks with transformers 5.x"

**Problem:** `AttributeError` or import errors when using TTS library

**Solution:**
```bash
pip uninstall transformers -y
pip install transformers==4.45.2
```

**Explanation:** The TTS library (Coqui XTTS-v2) is incompatible with transformers 5.x+ due to breaking API changes. Always use transformers 4.x series.

### "CUDA out of memory" errors

**Problem:** GPU runs out of VRAM during voice cloning or rendering

**Solutions:**
1. Use T4 GPU instead of K80 in Colab
2. Restart runtime to clear GPU memory
3. Process fewer lines at a time
4. Close other GPU-intensive notebooks

**Memory usage:** XTTS-v2 requires ~8-10GB VRAM for inference

### Audio only plays first few seconds

**Problem:** Generated narration is much shorter than expected (e.g., 3 seconds for 5 lines)

**Diagnosis:** This was a crossfade bug fixed in December 2024

**Solution:** Update to latest version from GitHub. If still occurring:
1. Check debug output in console for assembly duration
2. Verify all audio files were rendered (check temp directory)
3. Report issue with console logs

### Emotional prosody sounds robotic

**Problem:** Voices sound synthetic or strained, especially with high-pitched emotions

**Explanation:** Pitch shifting TTS-generated audio introduces artifacts if shifts are too large

**Current values:** Pitch shifts kept under ±0.8 semitones (subtle but natural)

**If still robotic:** You can reduce further in [config.py](config.py):
```python
EMOTION_PITCH_SHIFTS = {
    'joy': +0.3,      # Reduce from +0.5
    'surprise': +0.5, # Reduce from +0.8
    # ... etc
}
```

### Voice demo emotions overwrite each other

**Problem:** Testing multiple emotions produces same output

**Diagnosis:** Fixed bug where all tests used same line_number (1)

**Solution:** Update to latest version (fixed December 2024)

### Gradio InvalidPathError with file uploads

**Problem:** "File not found" errors when uploading screenplay files

**Solution:** Files must be in temp directory or workspace. Fixed in latest app.py (uses `/tmp` for all outputs)

---

## �📊 Performance

**Processing Times (on Colab T4 GPU):**
- Voice cloning: ~15-20 seconds (one-time per voice)
- Per-line rendering: ~2-3 seconds (includes emotional prosody processing)
- Prosody overhead: ~0.3-0.5 seconds per line (pitch shift + energy modulation)
- Example: 5-line scene (~12 seconds total processing time)
- Full screenplay (100 lines): ~8-12 minutes total

**Model Sizes:**
- XTTS-v2: ~6.4GB
- spaCy en_core_web_sm: ~12MB
- Emotion RoBERTa: ~350MB
- Sentiment RoBERTa: ~500MB

**Memory Requirements:**
- Runtime VRAM: ~8-10GB (GPU)
- Runtime RAM: ~4-6GB (CPU)
- Disk space: ~8GB (models + cache)

---

## 🔬 Research & Technical Details

### Director's Script Format

Each processed line is enriched with metadata including multi-emotion blends:

```json
{
  "line_number": 5,
  "speaker": "MARY",
  "line": "I can't believe you did this.",
  "parenthetical": "quietly",
  "emotion": "sadness",
  "emotion_blend": [
    {"emotion": "sadness", "score": 0.71},
    {"emotion": "fear", "score": 0.18},
    {"emotion": "surprise", "score": 0.05}
  ],
  "intensity": 0.72,
  "pace": "slow",
  "pause_before": 1.05,
  "original_text": "MARY: (quietly) I can't believe you did this."
}
```

**Field Descriptions:**
- `emotion`: Dominant emotion (highest score in blend) - used for voice selection
- `emotion_blend`: Top 3 emotions with scores - used for pace calculation and prosody
- `intensity`: Overall emotional intensity (0.0-1.0)
- `pace`: Computed pace category (`slow`, `normal`, `fast`)
- `pause_before`: Pause duration in seconds before this line

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
