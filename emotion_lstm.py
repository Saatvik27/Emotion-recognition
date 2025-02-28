import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, TFWav2Vec2Model, logging
from tqdm import tqdm
import multiprocessing

# Suppress transformer warnings
logging.set_verbosity_error()

# Load Wav2Vec2 components once
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base", from_pt=True)

@tf.keras.utils.register_keras_serializable()
class EmotionLSTM(Model):
    def __init__(self, num_classes=7):
        super(EmotionLSTM, self).__init__()
        self.masking = layers.Masking(mask_value=0.0)
        self.lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dropout1 = layers.Dropout(0.3)
        self.lstm2 = layers.Bidirectional(layers.LSTM(64))
        self.dense1 = layers.Dense(32, activation='relu')
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def build(self, input_shape):
        # Explicitly build the model to verify input dimensions
        super().build(input_shape)
        print(f"\nModel input shape: {input_shape}")
        
    def call(self, inputs):
        x = self.masking(inputs)
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        return self.classifier(x)


def extract_features(audio_path):
    try:
        audio = librosa.load(audio_path, sr=16000, duration=3)[0]
        audio = librosa.util.fix_length(audio, size=16000*3)
        inputs = processor(audio, return_tensors="tf", sampling_rate=16000, 
                          padding="max_length", max_length=16000*3)
        return wav2vec2_model(**inputs).last_hidden_state.numpy().squeeze()
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def prepare_dataset(audio_paths, labels, max_length=500):
    # Parallel processing with progress bar
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        features = list(tqdm(pool.imap(extract_features, audio_paths), total=len(audio_paths)))

    # Filter failed extractions
    valid_features = []
    valid_labels = []
    for f, l in zip(features, labels):
        if f is not None:
            valid_features.append(f[:max_length])  # Truncate instead of pad
            valid_labels.append(l)
    
    return np.array(valid_features), np.array(valid_labels)

# Train model
def train_emotion_model(X_train, y_train, X_val, y_val, num_classes=7):
    # Verify input shapes
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    model = EmotionLSTM(num_classes=num_classes)
    
    # Use lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add TensorBoard callback
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("models/speech_emotion_model.keras", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    # Train with validation
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,  # Increased batch size
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Main script
if __name__ == "__main__":
    tess_dataset_path = "./TESS"

    audio_paths = []
    labels = []
    emotion_map = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
        "neutral": 4, "pleasant_surprise": 5, "sad": 6
    }

    for folder in os.listdir(tess_dataset_path):
        folder_path = os.path.join(tess_dataset_path, folder)
        if os.path.isdir(folder_path):
            emotion = folder.split("_")[-1].lower()
            if emotion in emotion_map:
                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        audio_paths.append(os.path.join(folder_path, file))
                        labels.append(emotion_map[emotion])

    X, y = prepare_dataset(audio_paths, labels)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model, history = train_emotion_model(X_train, y_train, X_val, y_val)

    def predict_emotion(audio_path, model):
        features = extract_features(audio_path)
        padded_features = tf.keras.preprocessing.sequence.pad_sequences(
            [features], maxlen=500, dtype='float32', padding='post', truncating='post'
        )
        prediction = model.predict(padded_features)
        return np.argmax(prediction)

    test_audio = "test_audio.wav"
    predicted_class = predict_emotion(test_audio, model)
    emotion_labels = {v: k for k, v in emotion_map.items()}
    print(f"Predicted emotion: {emotion_labels[predicted_class]}")