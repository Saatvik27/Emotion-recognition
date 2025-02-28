# predict_emotion.py
import tensorflow as tf
import librosa
import soundfile as sf
import numpy as np
import argparse
import glob
import os
import re
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
import logging
import tensorflow.keras.layers as layers

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize Wav2Vec2 components once with PyTorch weights
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base", from_pt=True)

# Add this to predict_emotion.py before load_model()
@tf.keras.utils.register_keras_serializable()
class EmotionLSTM(tf.keras.Model):
    def __init__(self, num_classes=7, **kwargs):  # Add **kwargs here
        super(EmotionLSTM, self).__init__(**kwargs)  # Pass kwargs to parent
        self.masking = layers.Masking(mask_value=0.0)
        self.lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dropout1 = layers.Dropout(0.3)
        self.lstm2 = layers.Bidirectional(layers.LSTM(64))
        self.dense1 = layers.Dense(32, activation='relu')
        self.classifier = layers.Dense(num_classes, activation='softmax')
        self.num_classes = num_classes  # Store for config

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)
        print(f"\nModel input shape: {input_shape}")
        
    def call(self, inputs):
        x = self.masking(inputs)
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        return self.classifier(x)

def load_model(model_path="models/speech_emotion_model.keras"):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'EmotionLSTM': EmotionLSTM}
    )

def get_latest_recording(directory="recordings/audio"):
    """Find the latest recording based on timestamp in filename"""
    files = glob.glob(os.path.join(directory, "recording_*.wav"))
    
    if not files:
        raise FileNotFoundError(f"No recordings found in {directory}")
    
    latest_file = None
    latest_timestamp = None
    
    for file in files:
        match = re.search(r'recording_(\d{8})_(\d{6})\.wav$', os.path.basename(file))
        if match:
            timestamp_str = match.group(1) + match.group(2)  # Combine date and time
            timestamp = int(timestamp_str)
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_file = file
    
    if not latest_file:
        raise ValueError("No valid recordings found with timestamp pattern")
    
    return latest_file


def extract_features(audio_path):
    """Extract features from audio file using Wav2Vec2"""
    try:
        # Load and preprocess audio
        audio, sr = sf.read(audio_path)
        audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=16000)
        audio = librosa.util.fix_length(audio, size=16000*3)  # 3 seconds
        
        # Process through Wav2Vec2
        inputs = processor(
            audio, 
            sampling_rate=16000,
            return_tensors="tf",
            padding="max_length",
            max_length=16000*3,
            truncation=True
        )
        features = wav2vec2_model(**inputs).last_hidden_state.numpy()
        return features.squeeze(0)  # Remove batch dimension
        
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def predict_emotion(audio_path, model):
    """Make emotion prediction for a single audio file"""
    try:
        # Extract and pad features
        features = extract_features(audio_path)
        padded_features = tf.keras.preprocessing.sequence.pad_sequences(
            [features], 
            maxlen=500, 
            dtype='float32', 
            padding='post', 
            truncating='post'
        )
        
        # Make prediction
        prediction = model.predict(padded_features)
        return prediction
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

def main():
    # Define emotion labels
    emotion_labels = {
        0: "angry", 1: "disgust", 2: "fear", 3: "happy",
        4: "neutral", 5: "pleasant_surprise", 6: "sad"
    }

    try:
        # Get latest recording automatically
        audio_path = get_latest_recording()
        print(f"Analyzing latest recording: {os.path.basename(audio_path)}")
        
        # Load model and predict
        model = load_model()
        prediction = predict_emotion(audio_path, model)
        predicted_class = np.argmax(prediction)
        
        print(f"\nPredicted emotion: {emotion_labels[predicted_class]}")
        print("Confidence scores:")
        for emotion, score in zip(emotion_labels.values(), prediction[0]):
            print(f"{emotion}: {score:.4f}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()