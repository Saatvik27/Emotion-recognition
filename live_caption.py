import torch
import pyaudio
import numpy as np
import wave
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class LiveCaptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Captioning")
        
        # GUI Components
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.text_area.pack(padx=10, pady=10)
        
        self.status_label = tk.Label(root, text="Status: Stopped", fg="red")
        self.status_label.pack()
        
        self.start_button = tk.Button(root, text="Start Listening", command=self.toggle_listening)
        self.start_button.pack(pady=5)
        
        # Audio processing variables
        self.listening = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_size = 5  # Seconds of audio to buffer before processing
        
        # Initialize model and audio once
        self.initialize_model()
        self.initialize_audio()

    def initialize_model(self):
        MODEL_NAME = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

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
        
        # Start audio stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

    def stop_listening(self):
        self.listening = False
        self.status_label.config(text="Status: Stopped", fg="red")
        self.start_button.config(text="Start Listening")
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def process_audio(self):
        while self.listening:
            if not self.audio_queue.empty():
                # Collect audio data
                audio_data = self.audio_queue.get()
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_float32 = audio_np.astype(np.float32) / 32768.0
                self.audio_buffer = np.append(self.audio_buffer, audio_float32)
                
                # Process when buffer has enough data
                if len(self.audio_buffer) >= self.RATE * self.buffer_size:
                    self.process_buffer()

    def process_buffer(self):
        try:
            # Process audio buffer
            input_values = self.processor(
                self.audio_buffer,
                sampling_rate=self.RATE,
                return_tensors="pt"
            ).input_values
            
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Update GUI
            self.root.after(0, self.update_text, transcription)
            
            # Reset buffer
            self.audio_buffer = np.array([], dtype=np.float32)
            
        except Exception as e:
            print(f"Processing error: {str(e)}")

    def update_text(self, text):
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)  # Auto-scroll to bottom

    def on_closing(self):
        self.stop_listening()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveCaptionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()