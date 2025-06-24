#!/usr/bin/env python3
"""
Emotion Classification Test Script
This script loads the trained model and tests it on new audio files.
"""

import numpy as np
import librosa
import pickle
import tensorflow as tf
import joblib
import argparse
import os
import sys
import glob

class Config:
    """Configuration class for audio processing parameters"""
    # RAVDESS emotion labels
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm', 
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Audio processing parameters
    SAMPLE_RATE = 22050
    DURATION = 3.0
    N_MFCC = 13
    N_MELS = 128
    HOP_LENGTH = 512
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    def __init__(self):
        self.SAMPLE_RATE = 22050
        self.DURATION = 3.0
        self.N_MFCC = 13
        self.N_MELS = 128
        self.HOP_LENGTH = 512

class EmotionClassifier:
    def __init__(self, model_path, scaler_path, encoder_path, config_path):
        """
        Initialize the emotion classifier with trained model and preprocessors
        """
        # Load configuration
        try:
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
        except Exception as e:
            # If config loading fails, use default config
            print(f"Warning: Could not load config file ({e}), using default configuration")
            self.config = Config()
        
        # Ensure all required attributes exist with fallback values
        if not hasattr(self.config, 'SAMPLE_RATE'):
            self.config.SAMPLE_RATE = 22050
        if not hasattr(self.config, 'DURATION'):
            self.config.DURATION = 3.0
        if not hasattr(self.config, 'N_MFCC'):
            self.config.N_MFCC = 13
        if not hasattr(self.config, 'N_MELS'):
            self.config.N_MELS = 128
        if not hasattr(self.config, 'HOP_LENGTH'):
            self.config.HOP_LENGTH = 512
        
        print(f"Configuration loaded - Sample Rate: {self.config.SAMPLE_RATE}, Duration: {self.config.DURATION}s")
        
        # Load preprocessors
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load model
        if model_path.endswith('.h5'):
            self.model = tf.keras.models.load_model(model_path)
            self.model_type = 'keras'
        else:
            self.model = joblib.load(model_path)
            self.model_type = 'sklearn'
        
        print("Model loaded successfully!")
        print(f"Model type: {self.model_type}")
        print(f"Available emotions: {list(self.label_encoder.classes_)}")
    
    def extract_features(self, file_path):
        """
        Extract features from audio file (same as training pipeline)
        """
        try:
            # Get configuration values with fallbacks
            sample_rate = getattr(self.config, 'SAMPLE_RATE', 22050)
            duration = getattr(self.config, 'DURATION', 3.0)
            n_mfcc = getattr(self.config, 'N_MFCC', 13)
            n_mels = getattr(self.config, 'N_MELS', 128)
            hop_length = getattr(self.config, 'HOP_LENGTH', 512)
            
            # Load audio file
            y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            
            # Pad or truncate to fixed length
            max_len = int(sr * duration)
            if len(y) < max_len:
                y = np.pad(y, (0, max_len - len(y)), mode='constant')
            else:
                y = y[:max_len]
            
            # Extract features
            features = {}
            
            # 1. MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. Chroma features - Fixed function call
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            except AttributeError:
                # Fallback for older versions
                stft = librosa.stft(y, hop_length=hop_length)
                chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr)
            except:
                # Create dummy chroma features if all else fails
                chroma = np.random.random((12, int(len(y) / hop_length) + 1))
                
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 3. Mel-spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
            features['mel_mean'] = np.mean(mel, axis=1)
            features['mel_std'] = np.std(mel, axis=1)
            
            # 4. Spectral features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # 5. Rhythm features
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
            except:
                features['tempo'] = 120.0  # Default tempo
            
            # 6. Harmonic and percussive components
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                features['harmonic_mean'] = np.mean(y_harmonic)
                features['percussive_mean'] = np.mean(y_percussive)
            except:
                # Fallback calculations
                features['harmonic_mean'] = np.mean(y)
                features['percussive_mean'] = np.std(y)
            
            # Flatten all features
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def predict_emotion(self, file_path):
        """
        Predict emotion for a single audio file
        """
        # Extract features
        features = self.extract_features(file_path)
        if features is None:
            return None, None
        
        # Reshape and scale features
        features = features.reshape(1, -1)
        
        try:
            features_scaled = self.scaler.transform(features)
        except Exception as e:
            print(f"Error scaling features: {e}")
            return None, None
        
        # Make prediction
        try:
            if self.model_type == 'keras':
                predictions = self.model.predict(features_scaled, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)
            else:
                predicted_class = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            
            # Convert to emotion label
            emotion = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return emotion, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None
    
    def batch_predict(self, file_paths):
        """
        Predict emotions for multiple audio files
        """
        results = []
        for file_path in file_paths:
            emotion, confidence = self.predict_emotion(file_path)
            results.append({
                'file': os.path.basename(file_path),
                'emotion': emotion,
                'confidence': confidence
            })
        return results

def main():
    parser = argparse.ArgumentParser(description='Test emotion classification model')
    parser.add_argument('--audio', type=str, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, default='emotion_model.h5', help='Path to model file')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help='Path to scaler file')
    parser.add_argument('--encoder', type=str, default='label_encoder.pkl', help='Path to label encoder file')
    parser.add_argument('--config', type=str, default='config.pkl', help='Path to config file')
    
    args = parser.parse_args()
    
    # Check if all required files exist
    required_files = [args.model, args.scaler, args.encoder]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return
    
    # Config file is optional now
    if not os.path.exists(args.config):
        print(f"Warning: Config file not found: {args.config}, using default configuration")
    
    # Initialize classifier
    try:
        classifier = EmotionClassifier(args.model, args.scaler, args.encoder, args.config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process audio input
    if not args.audio:
        print("Error: Please provide audio file or directory path using --audio")
        return
    
    if os.path.isfile(args.audio):
        # Single file prediction
        print(f"Analyzing: {args.audio}")
        emotion, confidence = classifier.predict_emotion(args.audio)
        if emotion:
            print(f"Predicted Emotion: {emotion}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("Failed to process audio file")
    
    elif os.path.isdir(args.audio):
        # Batch prediction for directory
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(glob.glob(os.path.join(args.audio, ext)))
        
        if not audio_files:
            print("No audio files found in the directory")
            return
        
        print(f"Found {len(audio_files)} audio files")
        results = classifier.batch_predict(audio_files)
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 60)
        for result in results:
            if result['emotion']:
                print(f"{result['file']:<30} {result['emotion']:<12} {result['confidence']:.4f}")
            else:
                print(f"{result['file']:<30} {'ERROR':<12} {'N/A'}")
    
    else:
        print("Error: Audio path is neither a file nor a directory")

if __name__ == "__main__":
    # If no arguments provided, show usage example
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print("python test_model.py --audio sample.wav")
        print("python test_model.py --audio /path/to/audio/directory")
        print("python test_model.py --audio sample.wav --model emotion_model.pkl")
        print("\nFor help: python test_model.py -h")
    else:
        main()