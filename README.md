# Speech Emotion Recognition

A Python application that analyzes emotional content in speech by downloading audio files, transcribing speech to text, and predicting emotions using multiple pre-trained machine learning models.

## Features

- Downloads audio files from cloud storage (Google Cloud Storage)
- Transcribes speech to text using Google's Speech Recognition API
- Analyzes emotions using three different pre-trained models:
  - Cardiff NLP Twitter RoBERTa (11 emotions)
  - BERT Go Emotion (28 emotions)
  - DistilRoBERTa English (7 emotions)
- Returns the best emotion prediction with confidence scores

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd gcstetsing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your audio file URL.

## Usage

Run the script:
```bash
python gcs.py
```

The script will:
1. Download the audio file from the specified URL
2. Transcribe the speech to text
3. Analyze emotions in the transcribed text
4. Display the best emotion prediction with confidence score

## Environment Variables

- `AUDIO_URL`: URL to your audio file (required)
- `TEMP_AUDIO_PATH`: Temporary path for downloaded audio (optional, defaults to `/tmp/test.wav`)

## Dependencies

- PyTorch: Neural network framework
- Transformers: Hugging Face pre-trained models
- SpeechRecognition: Audio transcription
- NumPy: Numerical operations
- python-dotenv: Environment variable management

## Supported Emotions

Depending on the model, the system can detect:
- Basic emotions: anger, joy, sadness, fear, surprise, disgust
- Complex emotions: love, optimism, trust, anticipation, and more
- Social emotions: admiration, gratitude, pride, embarrassment

## License

MIT License - feel free to use this project for your own purposes.
