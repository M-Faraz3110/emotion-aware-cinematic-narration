"""
Stage 1: Screenplay Parser
Accepts raw text input and returns structured dialogue with speaker attribution.

Supports three input formats:
- Fountain screenplay format (industry standard)
- Plain dialogue (CHARACTER: dialogue)
- Raw text (treated as narrator monologue)
"""

import re
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreenplayParser:
    """
    Parses screenplay text into structured dialogue entries.
    Automatically detects input format and extracts speaker, dialogue, and stage directions.
    """
    
    # Fountain format markers
    FOUNTAIN_SCENE_HEADING = re.compile(r'^(INT|EXT|EST|INT\./EXT|INT/EXT|I/E)[\.\s]', re.IGNORECASE)
    FOUNTAIN_TRANSITION = re.compile(r'^(FADE IN:|FADE OUT\.|CUT TO:|DISSOLVE TO:)', re.IGNORECASE)
    FOUNTAIN_CHARACTER = re.compile(r'^[A-Z][A-Z\s\-\.]+$')
    
    # Plain dialogue format
    PLAIN_CHARACTER_LINE = re.compile(r'^([A-Z][A-Z\s\-\.]+):\s*(.+)$')
    
    # Parenthetical extraction (stage directions)
    PARENTHETICAL = re.compile(r'^\((.*?)\)\s*')
    
    def __init__(self):
        """Initialize the parser."""
        self.line_counter = 0
    
    def parse(self, text: str, format_hint: str = "auto") -> List[Dict]:
        """
        Parse screenplay text into structured dialogue entries.
        
        Args:
            text: Raw screenplay or dialogue text
            format_hint: Format type hint - "auto", "fountain", "plain", or "raw"
            
        Returns:
            List of dialogue dictionaries with structure:
            {
                "line_number": int,
                "speaker": str,
                "line": str,
                "parenthetical": str,
                "scene_context": str,
                "original_text": str
            }
            
        Raises:
            ValueError: If text is empty or too short
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Input text is too short or empty. Please provide at least 10 characters.")
        
        # Reset line counter
        self.line_counter = 0
        
        # Detect format if auto
        if format_hint == "auto":
            format_type = self._detect_format(text)
            logger.info(f"Auto-detected format: {format_type}")
        else:
            format_type = format_hint
        
        # Parse based on detected/specified format
        if format_type == "fountain":
            return self._parse_fountain(text)
        elif format_type == "plain":
            return self._parse_plain(text)
        else:  # raw
            return self._parse_raw(text)
    
    def _detect_format(self, text: str) -> str:
        """
        Auto-detect the format of the input text.
        
        Args:
            text: Raw input text
            
        Returns:
            One of: "fountain", "plain", "raw"
        """
        lines = text.split('\n')
        
        # Check for Fountain markers
        fountain_score = 0
        plain_score = 0
        
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            if not line:
                continue
                
            # Fountain indicators
            if self.FOUNTAIN_SCENE_HEADING.match(line):
                fountain_score += 3
            if self.FOUNTAIN_TRANSITION.match(line):
                fountain_score += 2
            if self.FOUNTAIN_CHARACTER.match(line) and len(line) < 40:
                fountain_score += 1
            
            # Plain dialogue indicator
            if self.PLAIN_CHARACTER_LINE.match(line):
                plain_score += 2
        
        if fountain_score >= 3:
            return "fountain"
        elif plain_score >= 2:
            return "plain"
        else:
            return "raw"
    
    def _patch_fountain_locale_bug(self):
        """
        Patch fountain library to fix Python 3.11+ LOCALE regex bug.
        
        The fountain library uses re.LOCALE flag with str patterns,
        which is deprecated in Python 3.11+. This patches the regex
        patterns to remove the LOCALE flag.
        """
        try:
            import fountain
            import re
            
            # Find and fix all regex patterns in fountain that use LOCALE
            for attr_name in dir(fountain):
                attr = getattr(fountain, attr_name)
                if isinstance(attr, type(re.compile(''))):
                    # Check if it uses LOCALE flag
                    if attr.flags & re.LOCALE:
                        # Recreate pattern without LOCALE flag
                        new_flags = attr.flags & ~re.LOCALE
                        new_pattern = re.compile(attr.pattern, new_flags)
                        setattr(fountain, attr_name, new_pattern)
            
            logger.info("Patched fountain library regex patterns for Python 3.11+ compatibility")
            return True
        except Exception as e:
            logger.debug(f"Could not patch fountain library: {e}")
            return False
    
    def _parse_fountain(self, text: str) -> List[Dict]:
        """
        Parse Fountain-formatted screenplay.
        
        Uses the screenplay-tools library for proper Fountain parsing,
        with fallback to manual parsing if library fails.
        
        Args:
            text: Fountain-formatted screenplay text
            
        Returns:
            List of structured dialogue entries
        """
        try:
            # Try using screenplay-tools library
            import fountain
            
            # Patch the LOCALE bug if needed
            self._patch_fountain_locale_bug()
            
            screenplay = fountain.Fountain(text)
            dialogue_entries = []
            
            for element in screenplay.elements:
                if element.element_type == 'dialogue':
                    character = element.character.strip() if hasattr(element, 'character') else "NARRATOR"
                    
                    # Extract parenthetical if present
                    parenthetical = ""
                    dialogue_text = element.text.strip() if hasattr(element, 'text') else ""
                    
                    if hasattr(element, 'parenthetical') and element.parenthetical:
                        parenthetical = element.parenthetical.strip('()')
                    
                    # Also check for inline parentheticals in dialogue
                    paren_match = self.PARENTHETICAL.match(dialogue_text)
                    if paren_match:
                        parenthetical = paren_match.group(1)
                        dialogue_text = dialogue_text[paren_match.end():].strip()
                    
                    if dialogue_text:  # Only add non-empty dialogue
                        self.line_counter += 1
                        dialogue_entries.append({
                            "line_number": self.line_counter,
                            "speaker": character.upper(),
                            "line": dialogue_text,
                            "parenthetical": parenthetical,
                            "original_text": f"{character}: {element.text}" if hasattr(element, 'text') else dialogue_text
                        })
            
            if dialogue_entries:
                logger.info(f"Parsed {len(dialogue_entries)} dialogue lines using fountain library")
                return dialogue_entries
            else:
                logger.info("Fountain library returned no dialogue, using manual parsing")
                
        except ImportError:
            logger.info("screenplay-tools/fountain library not installed, using manual parsing")
        except Exception as e:
            logger.info(f"Using manual Fountain parser (fountain library has Python 3.11+ compatibility issue)")
            logger.debug(f"Fountain error details: {e}")
        
        # Fallback: Manual Fountain parsing
        return self._parse_fountain_manual(text)
    
    def _parse_fountain_manual(self, text: str) -> List[Dict]:
        """
        Manual Fountain parser as fallback.
        
        Simple state machine that looks for:
        1. ALL CAPS character name
        2. Optional parenthetical on next line
        3. Dialogue on subsequent lines until blank line or next element
        4. Action lines for scene context
        
        Args:
            text: Fountain-formatted text
            
        Returns:
            List of structured dialogue entries with scene context
        """
        lines = text.split('\n')
        dialogue_entries = []
        scene_context = []  # Track recent action lines for environmental mood
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Scene headings update context but don't add to action lines
            if self.FOUNTAIN_SCENE_HEADING.match(line):
                scene_context = [line]  # Reset context with new scene
                i += 1
                continue
            
            # Skip transitions
            if self.FOUNTAIN_TRANSITION.match(line):
                i += 1
                continue
            
            # Check if this is a character name (ALL CAPS, short line)
            if self.FOUNTAIN_CHARACTER.match(line) and len(line) < 40 and line == line.upper():
                character = line.strip()
                i += 1
                
                # Capture current scene context for this dialogue block
                current_scene_context = " ".join(scene_context[-3:])  # Last 3 action lines
                
                # Collect dialogue lines with their parentheticals
                # Each entry is (dialogue_text, parenthetical)
                dialogue_with_parens = []
                pending_parenthetical = ""
                
                while i < len(lines):
                    next_line = lines[i].strip()
                    
                    # Empty line ends dialogue block (strict Fountain)
                    if not next_line:
                        break
                    
                    # Check if it's a parenthetical (wrapped in parens)
                    if next_line.startswith('(') and next_line.endswith(')'):
                        # Store parenthetical for the NEXT dialogue line
                        pending_parenthetical = next_line.strip('()')
                        i += 1
                        continue
                    
                    # Check if next character/scene heading (ends dialogue)
                    if (self.FOUNTAIN_CHARACTER.match(next_line) or 
                        self.FOUNTAIN_SCENE_HEADING.match(next_line) or
                        self.FOUNTAIN_TRANSITION.match(next_line)):
                        break
                    
                    # It's dialogue - check for inline parenthetical first
                    inline_paren_match = self.PARENTHETICAL.match(next_line)
                    if inline_paren_match:
                        # Inline parenthetical takes precedence
                        paren = inline_paren_match.group(1)
                        dialogue_text = next_line[inline_paren_match.end():].strip()
                        dialogue_with_parens.append((dialogue_text, paren))
                        pending_parenthetical = ""  # Clear pending
                    elif next_line:
                        # Use pending parenthetical if any
                        dialogue_with_parens.append((next_line, pending_parenthetical))
                        pending_parenthetical = ""  # Clear after use
                    
                    i += 1
                
                # Create separate entry for each dialogue line
                for dialogue_text, parenthetical in dialogue_with_parens:
                    if dialogue_text.strip():
                        self.line_counter += 1
                        dialogue_entries.append({
                            "line_number": self.line_counter,
                            "speaker": character.upper(),
                            "line": dialogue_text.strip(),
                            "parenthetical": parenthetical,
                            "scene_context": current_scene_context,
                            "original_text": f"{character}: {dialogue_text.strip()}"
                        })
            else:
                # This is an action line - add to scene context
                # (Not a character name, scene heading, or transition)
                scene_context.append(line)
                # Keep only last 5 action lines to avoid too much context
                if len(scene_context) > 5:
                    scene_context = scene_context[-5:]
                i += 1
        
        logger.info(f"Manual Fountain parsing extracted {len(dialogue_entries)} dialogue lines")
        return dialogue_entries
    
    def _parse_plain(self, text: str) -> List[Dict]:
        """
        Parse plain dialogue format (CHARACTER: dialogue).
        
        Args:
            text: Plain dialogue text with "CHARACTER: dialogue" format
            
        Returns:
            List of structured dialogue entries
        """
        lines = text.split('\n')
        dialogue_entries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match CHARACTER: dialogue pattern
            match = self.PLAIN_CHARACTER_LINE.match(line)
            if match:
                character = match.group(1).strip().upper()
                dialogue_text = match.group(2).strip()
                
                # Extract parenthetical from dialogue
                parenthetical = ""
                paren_match = self.PARENTHETICAL.match(dialogue_text)
                if paren_match:
                    parenthetical = paren_match.group(1)
                    dialogue_text = dialogue_text[paren_match.end():].strip()
                
                if dialogue_text:
                    self.line_counter += 1
                    dialogue_entries.append({
                        "line_number": self.line_counter,
                        "speaker": character,
                        "line": dialogue_text,
                        "parenthetical": parenthetical,
                        "scene_context": "",
                        "original_text": line
                    })
        
        logger.info(f"Plain dialogue parsing extracted {len(dialogue_entries)} dialogue lines")
        
        # If we found no matches, fall back to raw parsing
        if not dialogue_entries:
            logger.warning("No CHARACTER: dialogue patterns found, falling back to raw text parsing")
            return self._parse_raw(text)
        
        return dialogue_entries
    
    def _parse_raw(self, text: str) -> List[Dict]:
        """
        Parse raw text as narrator monologue.
        
        Splits text into sentences and treats each as a separate narration line.
        
        Args:
            text: Raw text without specific formatting
            
        Returns:
            List of structured dialogue entries (all assigned to NARRATOR)
        """
        # Simple sentence splitting (will be enhanced by spaCy in Stage 2 if needed)
        # For now, use basic splitting by sentence-ending punctuation
        
        # Clean up the text
        text = text.strip()
        
        # Split by sentence-ending punctuation followed by space and capital letter
        # This is a simple heuristic; spaCy will do better in Stage 2
        sentences = re.split(r'([.!?]+\s+)(?=[A-Z"])', text)
        
        # Reconstruct sentences
        dialogue_entries = []
        current_sentence = ""
        
        for i, part in enumerate(sentences):
            if re.match(r'[.!?]+\s+$', part):
                # This is punctuation/space
                current_sentence += part.strip()
                if current_sentence.strip():
                    self.line_counter += 1
                    dialogue_entries.append({
                        "line_number": self.line_counter,
                        "speaker": "NARRATOR",
                        "line": current_sentence.strip(),
                        "parenthetical": "",
                        "scene_context": "",
                        "original_text": current_sentence.strip()
                    })
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add final sentence if exists
        if current_sentence.strip():
            self.line_counter += 1
            dialogue_entries.append({
                "line_number": self.line_counter,
                "speaker": "NARRATOR",
                "line": current_sentence.strip(),
                "parenthetical": "",
                "scene_context": "",
                "original_text": current_sentence.strip()
            })
        
        # If sentence splitting failed, treat whole text as one entry
        if not dialogue_entries:
            self.line_counter += 1
            dialogue_entries.append({
                "line_number": self.line_counter,
                "speaker": "NARRATOR",
                "line": text,
                "parenthetical": "",
                "scene_context": "",
                "original_text": text
            })
        
        logger.info(f"Raw text parsing created {len(dialogue_entries)} narration lines")
        return dialogue_entries


def parse_screenplay(text: str, format_hint: str = "auto") -> List[Dict]:
    """
    Convenience function to parse screenplay text.
    
    Args:
        text: Raw screenplay or dialogue text
        format_hint: Format type hint - "auto", "fountain", "plain", or "raw"
        
    Returns:
        List of structured dialogue entries
        
    Example:
        >>> script = '''
        ... JOHN: I never wanted any of this.
        ... MARY: (quietly) But here we are.
        ... '''
        >>> result = parse_screenplay(script)
        >>> len(result)
        2
    """
    parser = ScreenplayParser()
    return parser.parse(text, format_hint)


if __name__ == "__main__":
    # Example usage
    test_script = """
    JOHN: I never wanted any of this.
    MARY: (quietly) But here we are.
    JOHN: What do we do now?
    """
    
    result = parse_screenplay(test_script)
    
    print("Parsed dialogue:")
    for entry in result:
        print(f"\nLine {entry['line_number']}:")
        print(f"  Speaker: {entry['speaker']}")
        print(f"  Line: {entry['line']}")
        if entry['parenthetical']:
            print(f"  Parenthetical: ({entry['parenthetical']})")
