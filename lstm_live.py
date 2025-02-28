import torch
import pyaudio
import numpy as np
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
import tensorflow as tf
import librosa
import soundfile as sf
from tensorflow.keras import layers


wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


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

# Load trained LSTM model
lstm_model = tf.keras.models.load_model("models/speech_emotion_model.keras", custom_objects={'EmotionLSTM': EmotionLSTM})

class LiveCaptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Captioning & Emotion Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # GUI Components
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.emotion_label = tk.Label(root, text="Speech Emotion: Unknown", font=("Arial", 14), fg="blue")
        self.emotion_label.pack()
        
        self.sentiment_label = tk.Label(root, text="Text Sentiment: Unknown", font=("Arial", 14), fg="purple")
        self.sentiment_label.pack()

        self.status_label = tk.Label(root, text="Status: Stopped", fg="red")
        self.status_label.pack()

        self.start_button = tk.Button(root, text="Start Listening", command=self.toggle_listening)
        self.start_button.pack(pady=5)
        
        # Audio Variables
        self.p = pyaudio.PyAudio()
        self.listening = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_size = 5  # Seconds
       
        self.initialize_audio()
        self.init_models()

    def init_models(self):
        """Initialize all ML models and tokenizers"""
        # Speech-to-Text model
        self.stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # Sentiment Analysis components
        self.sentiment_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.sentiment_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def update_text(self, text):
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)  # Auto-scroll to the latest text

    def update_sentiment(self, sentiment):
        self.sentiment_label.config(text=f"Text Sentiment: {sentiment}")
        
    def update_emotion(self, emotion):
        self.emotion_label.config(text=f"Speech Emotion: {emotion}")
    
    def initialize_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def toggle_listening(self):
        if not self.listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        self.listening = True
        self.status_label.config(text="Status: Listening", fg="green")
        self.start_button.config(text="Stop Listening")
        
        try:
            # Reinitialize audio components
            self.initialize_audio()
            
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback,
                start=False
            )
            
            self.stream.start_stream()
            self.processing_thread = threading.Thread(target=self.process_audio)
            self.processing_thread.start()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            self.listening = False

    def stop_listening(self):
        self.listening = False
        self.status_label.config(text="Status: Stopped", fg="red")
        self.start_button.config(text="Start Listening")
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.initialize_audio()

    def on_closing(self):
        """Proper cleanup when closing window"""
        self.stop_listening()
        self.p.terminate()
        self.root.destroy()
    
    def process_audio(self):
        while self.listening:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_float32 = audio_np.astype(np.float32) / 32768.0
                self.audio_buffer = np.append(self.audio_buffer, audio_float32)
                
                if len(self.audio_buffer) >= self.RATE * self.buffer_size:
                    self.process_buffer()

    def process_buffer(self):
        try:
            input_values = wav2vec_processor(
                self.audio_buffer,
                sampling_rate=self.RATE,
                return_tensors="pt"
            ).input_values

            with torch.no_grad():
                logits = wav2vec_model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = wav2vec_processor.batch_decode(predicted_ids)[0]

            if transcription.strip():
                self.root.after(0, self.update_text, transcription)
                self.detect_sentiment(transcription)
            else:
                print("No speech detected, skipping sentiment analysis.")

            self.detect_speech_emotion(self.audio_buffer)

            self.audio_buffer = np.array([], dtype=np.float32)

        except Exception as e:
            print(f"Processing error: {str(e)}")

    def detect_sentiment(self, text):
        try:
            # Use the correct sequence classification model
            inputs = self.sentiment_tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=50
            )

            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)

            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

            sentiments = ["Negative", "Neutral", "Positive"]
            if 0 <= predicted_class < len(sentiments):
                self.root.after(0, self.update_sentiment, sentiments[predicted_class])
            else:
                print("Invalid sentiment prediction")

        except Exception as e:
            print(f"Sentiment error: {str(e)}")

    def detect_speech_emotion(self, audio):
        try:
            # Ensure fixed input size
            if len(audio) < 16000 * 3:
                audio = np.pad(audio, (0, 16000 * 3 - len(audio)))
    
            # Process through Wav2Vec2
            inputs = wav2vec_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=16000 * 3,
                truncation=True
            )
            with torch.no_grad():
                outputs = wav2vec_model(**inputs)
            features = outputs.logits.numpy()
            features = features.squeeze(0)  # Remove batch dimension
    
            # Pad features to match the input shape expected by the LSTM model
            padded_features = tf.keras.preprocessing.sequence.pad_sequences(
                [features],
                maxlen=500,
                dtype='float32',
                padding='post',
                truncating='post'
            )
    
            # Make prediction
            prediction = lstm_model.predict(padded_features)
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"]
            detected_emotion = emotions[np.argmax(prediction)]
            self.root.after(0, self.update_emotion, detected_emotion)
    
        except Exception as e:
            print(f"Emotion detection error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveCaptionApp(root)
    root.mainloop()
