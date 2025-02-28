import os
import re
import glob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER's lexicon is available
nltk.download("vader_lexicon")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

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
    """Split text into chunks of max_length words to avoid processing issues."""
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def analyze_sentiment(text):
    """Analyze sentiment using VADER and return human-readable results."""
    scores = sia.polarity_scores(text)
    
    # Determine sentiment label
    if scores['compound'] >= 0.05:
        sentiment_label = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return {
        "Sentiment": sentiment_label,
        "Confidence": abs(scores['compound']),  # Compound score as confidence
        "Scores": scores  # Include detailed scores
    }

# Main script
try:
    file = get_latest_transcript()
    text = read_transcript(file)

    # Handle long text
    text_chunks = chunk_text(text)
    results = [analyze_sentiment(chunk) for chunk in text_chunks]

    # Print results
    print("\nSentiment Analysis Results:")
    for i, res in enumerate(results):
        print(f"Chunk {i+1}: Sentiment: {res['Sentiment']}, Confidence: {res['Confidence']:.2f}")
        print(f"  Detailed Scores: {res['Scores']}\n")

except Exception as e:
    print(f"Error: {e}")
