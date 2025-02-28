import os
import re
import glob
from transformers import pipeline

# Load the sentiment analysis pipeline with RoBERTa
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def get_latest_transcript(directory="recordings/text"):
    """Find the latest transcript based on timestamp in filename."""
    files = glob.glob(os.path.join(directory, "transcription_*.txt"))
    
    if not files:
        raise FileNotFoundError(f"No transcripts found in {directory}")
    
    latest_file = max(files, key=lambda f: int(re.search(r'(\d{8})_(\d{6})', f).group(0)))
    return latest_file

def read_transcript(file_path):
    """Read transcript safely with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()  # Remove extra spaces/newlines
        if not text:
            raise ValueError("Transcript is empty.")
        return text
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

def chunk_text(text, max_length=512):
    """Split text into chunks of max_length tokens to avoid model input limits."""
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

# Main script
try:
    file = get_latest_transcript()
    text = read_transcript(file)

    # Handle long text
    text_chunks = chunk_text(text)
    results = [sentiment_pipeline(chunk) for chunk in text_chunks]

    # Aggregate results
    print("\nSentiment Analysis Results:")
    for i, res in enumerate(results):
        print(f"Chunk {i+1}: {res}")

except Exception as e:
    print(f"Error: {e}")
