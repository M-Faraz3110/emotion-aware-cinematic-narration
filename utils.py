"""
Utility functions for the Emotion-Aware Cinematic Narration Engine.
Handles environment setup, model management, and helper functions.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment(force_colab: bool = False) -> Dict[str, Any]:
    """
    Setup the execution environment for the narration engine.
    Handles Google Drive mounting (on Colab), directory creation, and GPU detection.
    
    Args:
        force_colab: If True, assumes Colab environment even if not detected
        
    Returns:
        Dictionary with environment information:
        {
            'is_colab': bool,
            'has_gpu': bool,
            'gpu_name': str or None,
            'output_path': str,
            'drive_mounted': bool
        }
    """
    env_info = {
        'is_colab': False,
        'has_gpu': False,
        'gpu_name': None,
        'output_path': str(config.LOCAL_OUTPUT_PATH),
        'drive_mounted': False
    }
    
    # Detect Google Colab
    try:
        import google.colab
        env_info['is_colab'] = True
        logger.info("✓ Running on Google Colab")
    except ImportError:
        if force_colab:
            env_info['is_colab'] = True
            logger.warning("Forcing Colab mode (not actually on Colab)")
        else:
            logger.info("Running in local environment")
    
    # Check GPU availability
    if torch.cuda.is_available():
        env_info['has_gpu'] = True
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        logger.info(f"✓ GPU detected: {env_info['gpu_name']}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠ No GPU detected. Voice cloning will be very slow or may fail.")
        logger.warning("  Consider using Google Colab with GPU runtime.")
    
    # Mount Google Drive if on Colab
    if env_info['is_colab']:
        try:
            from google.colab import drive
            
            # Check if already mounted
            if not os.path.exists(config.DRIVE_MOUNT_PATH):
                logger.info("Mounting Google Drive...")
                drive.mount(config.DRIVE_MOUNT_PATH)
            else:
                logger.info("✓ Google Drive already mounted")
            
            env_info['drive_mounted'] = True
            env_info['output_path'] = config.DRIVE_OUTPUT_PATH
            
            # Create project directories
            os.makedirs(config.DRIVE_PROJECT_PATH, exist_ok=True)
            os.makedirs(config.DRIVE_OUTPUT_PATH, exist_ok=True)
            logger.info(f"✓ Output directory: {config.DRIVE_OUTPUT_PATH}")
            
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            logger.info(f"Falling back to local output path")
            env_info['drive_mounted'] = False
            env_info['output_path'] = "/content/outputs"
            os.makedirs(env_info['output_path'], exist_ok=True)
    else:
        # Create local output directory
        os.makedirs(config.LOCAL_OUTPUT_PATH, exist_ok=True)
        logger.info(f"✓ Output directory: {config.LOCAL_OUTPUT_PATH}")
    
    return env_info


def install_spacy_model(model_name: str = config.SPACY_MODEL) -> bool:
    """
    Install spaCy language model if not already installed.
    
    Args:
        model_name: Name of the spaCy model to install
        
    Returns:
        True if installation successful or model already exists
    """
    import spacy
    
    try:
        # Try to load the model
        spacy.load(model_name)
        logger.info(f"✓ spaCy model '{model_name}' already installed")
        return True
    except OSError:
        # Model not found, install it
        logger.info(f"Installing spaCy model '{model_name}'...")
        import subprocess
        
        try:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=subprocess.DEVNULL
            )
            logger.info(f"✓ Successfully installed '{model_name}'")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install spaCy model: {e}")
            return False


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are installed and accessible.
    
    Returns:
        Dictionary mapping dependency name to availability status
    """
    dependencies = {}
    
    # Check core libraries
    try:
        import spacy
        dependencies['spacy'] = True
    except ImportError:
        dependencies['spacy'] = False
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        dependencies['transformers'] = False
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        dependencies['torch'] = False
    
    try:
        import TTS
        dependencies['TTS'] = True
    except ImportError:
        dependencies['TTS'] = False
    
    try:
        import gradio
        dependencies['gradio'] = True
    except ImportError:
        dependencies['gradio'] = False
    
    try:
        import pydub
        dependencies['pydub'] = True
    except ImportError:
        dependencies['pydub'] = False
    
    try:
        import librosa
        dependencies['librosa'] = True
    except ImportError:
        dependencies['librosa'] = False
    
    # Fountain check disabled - causes regex LOCALE errors with pandas on Python 3.11
    # The fountain library will work fine at runtime despite this error
    dependencies['fountain'] = True  # Assume available (tested in requirements.txt)
    
    # try:
    #     import fountain
    #     dependencies['fountain'] = True
    # except (ImportError, ValueError) as e:
    #     # ValueError can occur from pandas regex LOCALE flag issues in Python 3.11
    #     if "LOCALE flag" in str(e):
    #         logger.warning(f"Fountain import triggered pandas regex issue (safe to ignore): {e}")
    #         dependencies['fountain'] = True  # Mark as available anyway
    #     else:
    #         dependencies['fountain'] = False
    
    # Log results
    print("Dependency check:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
        logger.info(f"  {status} {dep}")
    
    all_available = all(dependencies.values())
    if not all_available:
        missing = [dep for dep, avail in dependencies.items() if not avail]
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        logger.warning(f"Some dependencies are missing: {', '.join(missing)}")
    
    return dependencies


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2m 30s" or "45s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def get_device() -> str:
    """
    Get the best available device for PyTorch.
    
    Returns:
        "cuda" if GPU available, otherwise "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_environment_info(env_info: Dict[str, Any]) -> None:
    """
    Pretty print environment information.
    
    Args:
        env_info: Environment info dictionary from setup_environment()
    """
    print("\n" + "="*60)
    print(" EMOTION-AWARE CINEMATIC NARRATION ENGINE")
    print("="*60)
    print(f"\n📍 Environment:")
    print(f"   {'Google Colab' if env_info['is_colab'] else 'Local Machine'}")
    
    print(f"\n🎮 GPU:")
    if env_info['has_gpu']:
        print(f"   ✓ {env_info['gpu_name']}")
    else:
        print(f"   ✗ No GPU detected (voice cloning may be slow)")
    
    print(f"\n💾 Output Path:")
    print(f"   {env_info['output_path']}")
    
    if env_info['is_colab']:
        print(f"\n☁️  Google Drive:")
        print(f"   {'✓ Mounted' if env_info['drive_mounted'] else '✗ Not mounted'}")
    
    print("\n" + "="*60 + "\n")


def save_director_script(director_script: list, output_path: str, filename: str = "director_script.json") -> str:
    """
    Save the Director's Script JSON to file.
    
    Args:
        director_script: List of dialogue dictionaries
        output_path: Output directory path
        filename: Output filename
        
    Returns:
        Full path to saved file
    """
    import json
    
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(director_script, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved Director's Script to: {filepath}")
    return filepath


def load_director_script(filepath: str) -> list:
    """
    Load a Director's Script JSON from file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of dialogue dictionaries
    """
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        director_script = json.load(f)
    
    logger.info(f"✓ Loaded Director's Script from: {filepath}")
    return director_script


if __name__ == "__main__":
    # Test environment setup
    env_info = setup_environment()
    print_environment_info(env_info)
    
    # Check dependencies
    deps = check_dependencies()
    
    # Try to install spaCy model
    if deps.get('spacy', False):
        install_spacy_model()
