import os
import urllib.request
import torch
import numpy as np
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_analysis.log'),
        logging.StreamHandler()
    ]
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class EmotionModelCache:
    """Singleton class to preload and cache emotion models for fast predictions."""
    
    _instance = None
    _models_loaded = False
    
    # Model configuration
    MODELS_INFO = {
        "cardiffnlp/twitter-roberta-base-emotion-latest": {
            "labels": ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                       'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
        },
        "bhadresh-savani/bert-base-go-emotion": {
            "labels": [
                "admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"
            ]
        },
        "j-hartmann/emotion-english-distilroberta-base": {
            "labels": ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        }
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmotionModelCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self.models = {}
            self._load_models()
            self._models_loaded = True
    
    def _load_models(self):
        """Load all emotion models into memory."""
        print("Initializing emotion model cache...\n")
        
        for model_name, meta in self.MODELS_INFO.items():
            try:
                print(f"- Loading {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.eval()  # Set to evaluation mode
                
                self.models[model_name] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "labels": meta["labels"]
                }
                print(f"  Successfully cached")
                
            except Exception as err:
                print(f"  Failed to load {model_name}: {err}")
        
        print(f"\nModel cache ready! {len(self.models)} models loaded and cached.\n")
    
    def get_models(self):
        """Get the cached models dictionary."""
        return self.models
    
    def is_ready(self):
        """Check if models are loaded and ready."""
        return self._models_loaded and len(self.models) > 0


# Initialize the global model cache (loads models once at startup)
model_cache = EmotionModelCache()



def predict_emotions(text):
    """
    Predict emotions from text using cached models for fast inference.
    
    Args:
        text (str): The text to analyze for emotions
        
    Returns:
        dict: Results from all models with emotion and confidence scores
    """
    if not model_cache.is_ready():
        raise RuntimeError("Model cache is not ready. Please check model loading.")
    
    cached_models = model_cache.get_models()
    results = {}
    
    for name, parts in cached_models.items():
        tokenizer = parts["tokenizer"]
        model = parts["model"]
        labels = parts["labels"]

        # Tokenize input text
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get predictions (no gradient calculation for faster inference)
        with torch.no_grad():
            output = model(**tokens)

        # Calculate probabilities
        scores = torch.softmax(output.logits, dim=1)[0].numpy()
        top_idx = np.argmax(scores)

        results[name] = {
            "emotion": labels[top_idx],
            "confidence": float(scores[top_idx])
        }
    
    return results

def get_best_guess(results):
    """Get the model prediction with highest confidence."""
    return max(results.items(), key=lambda x: x[1]["confidence"])


def analyze_audio_file(audio_url, temp_path="/tmp/test.wav"):
    """
    Complete pipeline to analyze emotion from audio file.
    
    Args:
        audio_url (str): URL to the audio file
        temp_path (str): Temporary path to save downloaded audio
        
    Returns:
        dict: Analysis results including transcript and emotions
    """
    try:
        # Download audio
        print(f"Downloading audio from: {audio_url}")
        download_audio_from_gcs(audio_url, temp_path)
        print(f"Audio saved to: {temp_path}")

        # Transcribe
        print("Transcribing speech...")
        text = transcribe_audio(temp_path)
        print(f"Transcript: {text}\n")

        # Check transcription success
        if not text or text.startswith("Could not") or "error" in text.lower():
            return {
                "success": False,
                "error": "Transcription failed or was empty",
                "transcript": text
            }

        # Predict emotions using cached models (fast!)
        print("Analyzing emotions...")
        predictions = predict_emotions(text)
        best_model, best_result = get_best_guess(predictions)

        # Generate summary using Gemini
        print("Generating summary with Gemini...")
        summary = generate_summary_with_gemini(text)

        return {
            "success": True,
            "transcript": text,
            "summary": summary,
            "predictions": predictions,
            "best_prediction": {
                "model": best_model,
                "emotion": best_result['emotion'],
                "confidence": best_result['confidence']
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Speech recognition
def transcribe_audio(file_path):
    """Convert audio file to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"


# download
def download_audio_from_gcs(url, save_path):
    """Download audio file from Google Cloud Storage."""
    urllib.request.urlretrieve(url, save_path)

# Summary
def generate_summary_with_gemini(text):
    """
    Generate a bullet point summary of the transcribed text using Gemini 1.5 Flash.
    
    Args:
        text (str): The transcribed text to summarize
        
    Returns:
        str: Bullet point summary or error message
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not found in environment variables"
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for bullet point summary
        prompt = f"""
        Please provide a concise bullet point summary of the following speech transcript. 
        Focus on the main points, key messages, and important details mentioned.
        Format as clear bullet points (use • or -).
        
        Transcript:
        {text}
        
        Summary:
        """
        
        # Generate summary
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def get_supported_emotions():
    """
    Get list of all supported emotions across all models.
    
    Returns:
        dict: Dictionary mapping model names to their supported emotions
    """
    if not model_cache.is_ready():
        return {}
    
    emotions_by_model = {}
    for model_name, model_data in model_cache.get_models().items():
        model_short = model_name.split('/')[-1]
        emotions_by_model[model_short] = model_data['labels']
    
    return emotions_by_model


def get_all_unique_emotions():
    """
    Get set of all unique emotions supported across all models.
    
    Returns:
        set: Set of all unique emotion labels
    """
    all_emotions = set()
    for model_data in model_cache.get_models().values():
        all_emotions.update(model_data['labels'])
    return sorted(all_emotions)


def validate_environment():
    """
    Validate that all required environment variables are set.
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    audio_url = os.getenv("AUDIO_URL")
    if not audio_url:
        errors.append("AUDIO_URL not found in environment variables")
    elif not audio_url.startswith(('http://', 'https://')):
        errors.append("AUDIO_URL must be a valid HTTP/HTTPS URL")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        errors.append("GEMINI_API_KEY not found (AI summary will be disabled)")
    
    return len(errors) == 0, errors


# Main
if __name__ == "__main__":
    print("Speech Emotion Recognition System")
    print("=================================\n")
    
    # Validate environment variables
    is_valid, validation_errors = validate_environment()
    if not is_valid:
        print("Environment validation failed:")
        for error in validation_errors:
            print(f"- {error}")
        exit(1)

    # Get configuration from environment
    audio_url = os.getenv("AUDIO_URL")
    temp_audio_path = os.getenv("TEMP_AUDIO_PATH", "/tmp/test.wav")
    
    # Check if models are ready
    if not model_cache.is_ready():
        print("Error: Models failed to load properly.")
        exit(1)

    print(f"Models cached and ready for fast inference!")
    print(f"Target audio: {audio_url}\n")

    # Analyze the audio file
    result = analyze_audio_file(audio_url, temp_audio_path)
    
    if result["success"]:
        print("EMOTION ANALYSIS RESULTS:")
        print("=" * 40)
        print(f"Transcript: {result['transcript']}")
        print()
        print(f"Top Prediction:")
        print(f"  Model: {result['best_prediction']['model']}")
        print(f"  Emotion: {result['best_prediction']['emotion'].upper()}")
        print(f"  Confidence: {result['best_prediction']['confidence']:.1%}")
        print()
        
        # Show all model predictions
        print("All Model Predictions:")
        for model_name, pred in result['predictions'].items():
            model_short = model_name.split('/')[-1]
            print(f"  {model_short}: {pred['emotion']} ({pred['confidence']:.1%})")
        
        # Show Gemini summary
        print()
        print("GEMINI AI SUMMARY:")
        print("=" * 40)
        print(result['summary'])
            
    else:
        print(f"Analysis failed: {result.get('error', 'Unknown error')}")
        if 'transcript' in result:
            print(f"Transcript: {result['transcript']}")

    print("\nAnalysis complete!")