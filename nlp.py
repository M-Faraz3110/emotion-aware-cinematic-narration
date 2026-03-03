"""
Stage 2: NLP Pipeline
Analyzes dialogue with contextual emotion detection, intensity scoring, and cinematic pacing inference.

This module enriches parsed dialogue with:
- Emotion classification (contextually aware)
- Intensity scoring (sentiment-based)
- Pace inference (slow/normal/fast)
- Pause calculation (based on multiple signals)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

import spacy
from transformers import pipeline
import torch

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    NLP Pipeline for emotion-aware dialogue analysis.
    
    Performs contextual emotion detection, sentiment intensity scoring,
    and cinematic pacing inference to create a Director's Script.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the NLP pipeline with all required models.
        
        Args:
            device: PyTorch device ("cuda" or "cpu"). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing NLP Pipeline on device: {self.device}")
        
        # Load models
        logger.info("Loading emotion classification model...")
        self.emotion_classifier = pipeline(
            "text-classification",
            model=config.EMOTION_MODEL,
            device=0 if self.device == "cuda" else -1,
            top_k=None  # Return all class probabilities
        )
        
        logger.info("Loading sentiment intensity model...")
        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("Loading spaCy model for dependency parsing...")
        self.nlp = spacy.load(config.SPACY_MODEL)
        
        logger.info("✓ All NLP models loaded successfully")
    
    def analyze(self, parsed_dialogue: List[Dict]) -> List[Dict]:
        """
        Analyze parsed dialogue and create Director's Script.
        
        This is the main entry point that orchestrates the full NLP pipeline:
        1. Contextual emotion analysis (with 60/40 blending)
        2. Sentiment intensity scoring
        3. Pace inference (4 signal types)
        4. Pause calculation (additive with cap)
        
        Args:
            parsed_dialogue: List of dialogue dicts from parser.py
                             Each has: line_number, speaker, line, parenthetical, original_text
        
        Returns:
            Director's Script: List of enriched dialogue dicts with:
                - All original fields
                - emotion: str (joy, anger, fear, sadness, disgust, surprise, neutral)
                - intensity: float (0.0 to 1.0)
                - pace: str (slow, normal, fast)
                - pause_before: float (seconds)
        """
        if not parsed_dialogue:
            logger.warning("Empty dialogue received, returning empty Director's Script")
            return []
        
        logger.info(f"Analyzing {len(parsed_dialogue)} dialogue lines...")
        
        director_script = []
        
        for i, dialogue_entry in enumerate(parsed_dialogue):
            # Extract contextual emotion and intensity
            emotion, emotion_confidence, intensity, emotion_blend = self._analyze_emotion_and_intensity(
                dialogue_entry, i, parsed_dialogue
            )
            
            # Build context window for display
            dialogue_context = self._build_context_window(i, parsed_dialogue)
            
            # Infer pace and pause
            pace_score = self._infer_pace(
                dialogue_entry, emotion, intensity, emotion_blend
            )
            
            pause_before = self._calculate_pause(
                dialogue_entry, emotion, intensity, pace_score, i, parsed_dialogue
            )
            
            # Convert numeric pace to categorical
            pace = self._pace_score_to_category(pace_score)
            
            # Create enriched entry
            enriched_entry = {
                **dialogue_entry,  # Keep all original fields
                "emotion": emotion,
                "emotion_confidence": round(emotion_confidence, 3),
                "emotion_blend": emotion_blend,  # Multi-emotion breakdown
                "intensity": round(intensity, 3),
                "pace": pace,
                "pace_score": round(pace_score, 3),  # Keep for debugging
                "pause_before": round(pause_before, 2),
                "dialogue_context": dialogue_context  # Add for UI display
            }
            
            director_script.append(enriched_entry)
            
            # Log progress
            if config.DEBUG_PRINT_DIRECTOR_SCRIPT:
                logger.info(
                    f"Line {dialogue_entry['line_number']}: "
                    f"{emotion}({intensity:.2f}) | "
                    f"pace={pace} | pause={pause_before:.2f}s"
                )
        
        logger.info(f"✓ Director's Script complete with {len(director_script)} enriched lines")
        return director_script
    
    def _analyze_emotion_and_intensity(
        self, 
        dialogue_entry: Dict, 
        index: int, 
        all_dialogue: List[Dict]
    ) -> Tuple[str, float, float, List[Dict]]:
        """
        Analyze emotion and intensity using contextual awareness and multi-source blending.
        
        Uses context-dominant blending (25-30-15-30):
        - 25% from the dialogue line itself
        - 30% from surrounding dialogue context (±2 lines)
        - 15% from parenthetical (direct delivery instruction)
        - 30% from scene context (atmospheric mood)
        
        Returns multi-emotion blend instead of single emotion for nuanced analysis.
        
        Args:
            dialogue_entry: Current dialogue entry
            index: Index in all_dialogue
            all_dialogue: Full dialogue list for context
        
        Returns:
            Tuple of (dominant_emotion, confidence, intensity, emotion_blend)
            where emotion_blend is list of dicts: [{"emotion": str, "score": float}]
        """
        target_line = dialogue_entry['line']
        
        # === Step 1: Analyze target line alone (50% weight) ===
        line_emotion_results = self.emotion_classifier(target_line)[0]
        line_emotion_scores = {r['label']: r['score'] for r in line_emotion_results}
        
        # === Step 2: Analyze surrounding dialogue context (30% weight) ===
        context_text = self._build_context_window(index, all_dialogue)
        context_emotion_results = self.emotion_classifier(context_text)[0]
        context_emotion_scores = {r['label']: r['score'] for r in context_emotion_results}
        
        # === Step 3: Analyze parenthetical (12% weight) ===
        parenthetical_scores = {}
        if dialogue_entry.get('parenthetical'):
            parenthetical_emotion = self._analyze_parenthetical_emotion(
                target_line, dialogue_entry['parenthetical']
            )
            parenthetical_scores = parenthetical_emotion
        else:
            # No parenthetical - use line's own emotion
            parenthetical_scores = line_emotion_scores
        
        # === Step 4: Analyze scene context (8% weight) ===
        scene_scores = {}
        if dialogue_entry.get('scene_context'):
            scene_emotion_results = self.emotion_classifier(dialogue_entry['scene_context'])[0]
            scene_scores = {r['label']: r['score'] for r in scene_emotion_results}
        else:
            # Neutral if no scene context
            scene_scores = {emotion: 0.0 for emotion in config.EMOTION_LABELS}
            scene_scores['neutral'] = 1.0
        
        # === Step 5: Blend all results (25% + 30% + 15% + 30% = 100%) ===
        blended_scores = {}
        for emotion in config.EMOTION_LABELS:
            line_score = line_emotion_scores.get(emotion, 0.0)
            dialogue_score = context_emotion_scores.get(emotion, 0.0)
            paren_score = parenthetical_scores.get(emotion, 0.0)
            scene_score = scene_scores.get(emotion, 0.0)
            
            blended_scores[emotion] = (
                config.LINE_WEIGHT * line_score +
                config.DIALOGUE_WEIGHT * dialogue_score +
                config.PARENTHETICAL_WEIGHT * paren_score +
                config.SCENE_WEIGHT * scene_score
            )
        
        # Sort emotions by score to get top N
        sorted_emotions = sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build emotion blend: keep top N emotions above minimum threshold
        emotion_blend = []
        for emotion, score in sorted_emotions[:config.EMOTION_BLEND_TOP_N]:
            if score >= config.EMOTION_BLEND_MIN_SCORE:
                emotion_blend.append({
                    "emotion": emotion,
                    "score": score
                })
        
        # Dominant emotion (for backward compatibility and voice synthesis)
        final_emotion = sorted_emotions[0][0]
        final_confidence = sorted_emotions[0][1]
        
        # === Step 6: Extract intensity from sentiment ===
        intensity = self._extract_intensity(target_line)
        
        # Apply parenthetical intensity modifiers
        if dialogue_entry.get('parenthetical'):
            intensity = self._apply_parenthetical_intensity_modifier(
                intensity, dialogue_entry['parenthetical']
            )
        
        return final_emotion, final_confidence, intensity, emotion_blend
    
    def _build_context_window(self, index: int, all_dialogue: List[Dict]) -> str:
        """
        Build context window with N lines before and after target.
        
        Args:
            index: Current line index
            all_dialogue: Full dialogue list
        
        Returns:
            Context string in natural language format
        """
        start_idx = max(0, index - config.CONTEXT_WINDOW_BEFORE)
        end_idx = min(len(all_dialogue), index + config.CONTEXT_WINDOW_AFTER + 1)
        
        context_lines = []
        for i in range(start_idx, end_idx):
            # Include speaker for better context
            speaker = all_dialogue[i]['speaker']
            line = all_dialogue[i]['line']
            context_lines.append(f"{speaker}: {line}")
        
        return " ".join(context_lines)
    
    def _analyze_parenthetical_emotion(self, line: str, parenthetical: str) -> Dict[str, float]:
        """
        Extract emotion signals from parenthetical stage directions in context.
        
        Analyzes the parenthetical together with the dialogue line it modifies,
        since stage directions like "(beat)" or "(quietly)" derive meaning from
        the emotional context of the line they're attached to.
        
        Args:
            line: The dialogue line being spoken
            parenthetical: Stage direction text
        
        Returns:
            Dictionary of emotion scores
        """
        # Analyze line + parenthetical together for contextual emotion
        combined_text = f"{line} ({parenthetical})"
        
        try:
            paren_emotion_results = self.emotion_classifier(combined_text)[0]
            return {r['label']: r['score'] for r in paren_emotion_results}
        except Exception as e:
            logger.warning(f"Parenthetical emotion analysis failed: {e}")
            # Fallback to line emotion only
            try:
                line_results = self.emotion_classifier(line)[0]
                return {r['label']: r['score'] for r in line_results}
            except:
                # Last resort: neutral
                return {emotion: 0.0 if emotion != 'neutral' else 1.0 
                       for emotion in config.EMOTION_LABELS}
    
    def _extract_intensity(self, text: str) -> float:
        """
        Extract emotion intensity from sentiment analysis.
        
        Uses the confidence score from sentiment classification as intensity proxy.
        
        Args:
            text: Input text
        
        Returns:
            Intensity score between 0.0 and 1.0
        """
        result = self.sentiment_classifier(text)[0]
        intensity = result['score']  # Confidence of predicted sentiment
        
        # Normalize to 0-1 range (already should be, but ensure)
        return max(0.0, min(1.0, intensity))
    
    def _apply_parenthetical_intensity_modifier(
        self, 
        base_intensity: float, 
        parenthetical: str
    ) -> float:
        """
        Apply intensity modifiers based on parenthetical stage directions.
        
        Args:
            base_intensity: Original intensity score
            parenthetical: Stage direction (e.g., "quietly", "shouting")
        
        Returns:
            Modified intensity score (clamped to 0.0-1.0)
        """
        paren_lower = parenthetical.lower().strip()
        
        if paren_lower in config.PARENTHETICAL_MAPPINGS:
            modifier = config.PARENTHETICAL_MAPPINGS[paren_lower]['intensity_modifier']
            modified = base_intensity + modifier
            return max(0.0, min(1.0, modified))
        
        return base_intensity
    
    def _infer_pace(
        self, 
        dialogue_entry: Dict, 
        emotion: str, 
        intensity: float,
        emotion_blend: List[Dict]
    ) -> float:
        """
        Infer speaking pace using multiple signals with multi-emotion blending.
        
        Signals considered:
        1. Parenthetical override (e.g., "quickly", "slowly")
        2. Punctuation patterns (!, ?, ...)
        3. Multi-emotion weighted pacing (uses emotion_blend)
        4. Intensity scaling
        
        Args:
            dialogue_entry: Current dialogue entry
            emotion: Dominant emotion (for backward compatibility)
            intensity: Emotion intensity
            emotion_blend: List of emotion dictionaries with scores
        
        Returns:
            Pace score (negative = slower, positive = faster, 0 = normal)
        """
        pace_score = 0.0  # Start at normal
        
        # === Signal 1: Parenthetical override ===
        if dialogue_entry['parenthetical']:
            paren_lower = dialogue_entry['parenthetical'].lower().strip()
            if paren_lower in config.PARENTHETICAL_MAPPINGS:
                paren_pace = config.PARENTHETICAL_MAPPINGS[paren_lower]['pace']
                if paren_pace == config.PACE_SLOW:
                    pace_score -= 0.5
                elif paren_pace == config.PACE_FAST:
                    pace_score += 0.5
        
        # === Signal 2: Punctuation patterns ===
        line = dialogue_entry['line']
        
        if '!' in line:
            pace_score += 0.3  # Exclamation = faster
        
        if '?' in line:
            pace_score += 0.1  # Question = slightly faster
        
        if '...' in line or '…' in line:
            pace_score -= 0.2  # Ellipsis = slower, trailing off
        
        if '—' in line or '--' in line:
            pace_score += 0.1  # Em dash = slight urgency
        
        # === Signal 3: Multi-emotion weighted pacing ===
        # Blend pace contributions from all emotions in emotion_blend
        for emotion_entry in emotion_blend:
            emotion_name = emotion_entry['emotion']
            emotion_score = emotion_entry['score']
            contribution = config.EMOTION_PACE_CONTRIBUTIONS.get(emotion_name, 0.0)
            pace_score += contribution * emotion_score
        
        # === Signal 4: Intensity scaling ===
        # High intensity = more pronounced (amplify current direction)
        if intensity > 0.7:
            pace_score *= 1.2
        
        return pace_score
    
    def _calculate_pause(
        self,
        dialogue_entry: Dict,
        emotion: str,
        intensity: float,
        pace_score: float,
        index: int,
        all_dialogue: List[Dict]
    ) -> float:
        """
        Calculate pause before this line using additive signals with cap.
        
        Signals considered:
        1. Parenthetical pauses (beat, pause, etc.)
        2. Punctuation signals (ellipsis, em dash)
        3. spaCy dependency parsing (clause boundaries)
        4. Emotion-intensity scaling
        
        Args:
            dialogue_entry: Current dialogue entry
            emotion: Detected emotion
            intensity: Emotion intensity
            pace_score: Calculated pace score
            index: Line index
            all_dialogue: Full dialogue list
        
        Returns:
            Pause duration in seconds (capped at MAX_PAUSE)
        """
        pause = config.BASE_PAUSE  # Start with base pause
        
        # === Signal 1: Parenthetical override ===
        if dialogue_entry['parenthetical']:
            paren_lower = dialogue_entry['parenthetical'].lower().strip()
            if paren_lower in config.PARENTHETICAL_MAPPINGS:
                paren_pause = config.PARENTHETICAL_MAPPINGS[paren_lower]['pause_before']
                pause += paren_pause
        
        # === Signal 2: Punctuation ===
        line = dialogue_entry['line']
        
        if '...' in line or '…' in line:
            pause += config.PAUSE_ELLIPSIS
        
        if '—' in line or '--' in line:
            pause += config.PAUSE_EM_DASH
        
        if '?' in line:
            pause += config.PAUSE_QUESTION
        
        # === Signal 3: spaCy dependency parsing (clause boundaries) ===
        clause_pause = self._detect_clause_boundaries(line)
        pause += clause_pause
        
        # === Signal 4: Emotion-intensity scaling ===
        if intensity > config.INTENSITY_HIGH_THRESHOLD:
            pause += config.PAUSE_HIGH_INTENSITY  # High emotion = deliberate delivery
        elif intensity < config.INTENSITY_LOW_THRESHOLD:
            pause += config.PAUSE_LOW_INTENSITY  # Low emotion = quicker (negative value)
        
        # Cap at maximum
        pause = max(config.MIN_PAUSE, min(pause, config.MAX_PAUSE))
        
        return pause
    
    def _detect_clause_boundaries(self, text: str) -> float:
        """
        Detect major syntactic boundaries using spaCy dependency parsing.
        
        Adds small pauses for:
        - Subordinate clauses
        - Conjunctions (and, but, or)
        - Appositive phrases
        
        Args:
            text: Input text
        
        Returns:
            Additional pause time based on clause count
        """
        doc = self.nlp(text)
        
        clause_count = 0
        
        for token in doc:
            # Subordinating conjunctions (because, although, if, when, etc.)
            if token.dep_ in ['mark', 'advcl']:
                clause_count += 1
            
            # Coordinating conjunctions (and, but, or)
            if token.dep_ == 'cc' and token.text.lower() in ['but', 'and', 'or']:
                clause_count += 1
            
            # Appositive phrases
            if token.dep_ == 'appos':
                clause_count += 1
        
        # Add 0.2s per major clause boundary (reasonable for natural speech)
        return clause_count * 0.2
    
    def _pace_score_to_category(self, pace_score: float) -> str:
        """
        Convert numeric pace score to categorical pace.
        
        Thresholds:
        - < -0.3: slow
        - -0.3 to 0.3: normal
        - > 0.3: fast
        
        Args:
            pace_score: Numeric pace score
        
        Returns:
            One of: "slow", "normal", "fast"
        """
        if pace_score < -0.3:
            return config.PACE_SLOW
        elif pace_score > 0.3:
            return config.PACE_FAST
        else:
            return config.PACE_NORMAL


def analyze_dialogue(parsed_dialogue: List[Dict]) -> List[Dict]:
    """
    Convenience function to analyze dialogue with NLP pipeline.
    
    Args:
        parsed_dialogue: Output from parser.parse_screenplay()
    
    Returns:
        Director's Script with emotion, intensity, pace, and pause data
    
    Example:
        >>> from parser import parse_screenplay
        >>> from nlp import analyze_dialogue
        >>> 
        >>> script = "JOHN: I never wanted this."
        >>> parsed = parse_screenplay(script)
        >>> director_script = analyze_dialogue(parsed)
        >>> 
        >>> print(director_script[0]['emotion'])
        'sadness'
    """
    pipeline = NLPPipeline()
    return pipeline.analyze(parsed_dialogue)


if __name__ == "__main__":
    # Example usage
    from parser import parse_screenplay
    import json
    
    test_script = """
    JOHN: I never wanted any of this.
    MARY: (quietly) But here we are.
    JOHN: What do we do now?
    MARY: (pause) We survive. That's what we always do.
    """
    
    # Parse screenplay
    parsed = parse_screenplay(test_script)
    
    # Analyze with NLP
    nlp = NLPPipeline()
    director_script = nlp.analyze(parsed)
    
    # Print results
    print("\n" + "="*60)
    print("DIRECTOR'S SCRIPT")
    print("="*60)
    print(json.dumps(director_script, indent=2))
