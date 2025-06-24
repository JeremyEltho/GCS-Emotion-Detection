import os
import urllib.request
import torch
import numpy as np
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Model Config ===
models_info = {
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

# === Load Models ===
print("Loading models...\n")
models = {}
for model_name, meta in models_info.items():
    try:
        print(f"- {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        models[model_name] = {
            "tokenizer": tokenizer,
            "model": model,
            "labels": meta["labels"]
        }
    except Exception as err:
        print(f"Can't load {model_name}: {err}")
print("\nAll models are ready.\n")

# === Predict Emotions ===
def predict_emotions(text):
    results = {}
    for name, parts in models.items():
        tokenizer = parts["tokenizer"]
        model = parts["model"]
        labels = parts["labels"]

        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**tokens)

        scores = torch.softmax(output.logits, dim=1)[0].numpy()
        top_idx = np.argmax(scores)

        results[name] = {
            "emotion": labels[top_idx],
            "confidence": float(scores[top_idx])
        }
    return results

# === Get Best Guess ===
def get_best_guess(results):
    return max(results.items(), key=lambda x: x[1]["confidence"])

# === Transcribe Audio ===
def transcribe_audio(file_path):
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

# === Download audio from GCS ===
def download_audio_from_gcs(url, save_path):
    urllib.request.urlretrieve(url, save_path)

# === Main Pipeline ===
if __name__ == "__main__":
    audio_url = os.getenv("AUDIO_URL")
    temp_audio_path = os.getenv("TEMP_AUDIO_PATH", "/tmp/test.wav")
    
    if not audio_url:
        print("Error: AUDIO_URL not found in environment variables.")
        print("Please create a .env file with AUDIO_URL=<your_audio_url>")
        exit(1)

    print(f"\nDownloading audio from GCS...")
    download_audio_from_gcs(audio_url, temp_audio_path)
    print(f"Audio saved to {temp_audio_path}")

    print("\nTranscribing...")
    text = transcribe_audio(temp_audio_path)
    print(f"Transcript:\n{text}\n")

    if not text or text.startswith("Could not") or "error" in text.lower():
        print("Transcription failed or was empty.")
    else:
        predictions = predict_emotions(text)
        best_model, best_result = get_best_guess(predictions)

        print("Top Emotion Prediction:")
        print(f"- Model: {best_model}")
        print(f"- Emotion: {best_result['emotion']}")
        print(f"- Confidence: {best_result['confidence']:.2f}")

    # Optional: remove temp file
    os.remove(temp_audio_path)
