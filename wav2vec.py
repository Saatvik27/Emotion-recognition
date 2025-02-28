from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor
import tensorflow as tf
import sounddevice as sd
import numpy as np
import tkinter as tk
from queue import Queue
from threading import Thread
import soundfile as sf  # New import for audio file saving
from datetime import datetime
import os

# Initialize global variables
is_recording = False
audio_queue = Queue()
audio_data = None
sample_rate = 16000

# Create directory for recordings if it doesn't exist
if not os.path.exists('recordings'):
    os.makedirs('recordings')

# Load pre-trained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
stt_model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def audio_callback(indata, frames, time, status):
    if is_recording:
        audio_queue.put(indata.copy())

def start_recording():
    global is_recording, audio_data
    is_recording = True
    audio_data = []
    print("Recording started...")

def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped")
    process_audio()

def process_audio():
    global audio_data
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine audio chunks
    audio_array = np.concatenate(list(audio_queue.queue), axis=0)
    audio_data = audio_array.squeeze().astype(np.float32)
    
    # Save audio file
    audio_filename = f"recordings/audio/recording_{timestamp}.wav"
    sf.write(audio_filename, audio_data, sample_rate)
    print(f"Audio saved as {audio_filename}")
    
    # Convert to Wav2Vec2 input format
    input_values = processor(audio_data, return_tensors="tf", 
                           sampling_rate=sample_rate).input_values
    
    # Get transcription
    logits = stt_model(input_values).logits
    predicted_ids = tf.argmax(logits, axis=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Save text file
    text_filename = f"recordings/text/transcription_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write(transcription)
    print(f"Transcription saved as {text_filename}")
    
    # Update GUI
    result_label.config(text=f"Transcription: {transcription}\n"
                             f"Audio saved to: {audio_filename}\n"
                             f"Text saved to: {text_filename}")

def replay_recording():
    if audio_data is not None:
        print("Playing back recording...")
        sd.play(audio_data, sample_rate)
        sd.wait()
    else:
        result_label.config(text="No recording to play!")

# Create GUI
root = tk.Tk()
root.title("Voice Recorder")

# Create buttons
start_button = tk.Button(root, text="Start Recording", command=start_recording)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
replay_button = tk.Button(root, text="Replay Recording", command=replay_recording)
result_label = tk.Label(root, text="Transcription will appear here", wraplength=400)

# Layout
start_button.pack(pady=5)
stop_button.pack(pady=5)
replay_button.pack(pady=5)
result_label.pack(pady=10)

# Start audio stream
def audio_stream():
    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=sample_rate,
                       blocksize=1024):
        while True:
            sd.sleep(1000)

stream_thread = Thread(target=audio_stream, daemon=True)
stream_thread.start()

root.mainloop()