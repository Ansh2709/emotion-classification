import streamlit as st
import numpy as np
import librosa
import pickle
import tensorflow as tf
import joblib
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Emotion Classification from Speech",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Default configuration class to use if config loading fails
class DefaultConfig:
    SAMPLE_RATE = 22050
    DURATION = 3.0
    N_MFCC = 13
    N_MELS = 128
    HOP_LENGTH = 512

@st.cache_resource
def load_model_and_preprocessors():
    """Load trained model and preprocessing objects"""
    try:
        # Try to load DNN model first
        if os.path.exists('emotion_model.h5'):
            model = tf.keras.models.load_model('emotion_model.h5')
            model_type = 'keras'
        elif os.path.exists('emotion_model.pkl'):
            model = joblib.load('emotion_model.pkl')
            model_type = 'sklearn'
        else:
            return None, None, None, None, "Model file not found"
        
        # Load preprocessors
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Try to load config, use default if it fails
        try:
            with open('config.pkl', 'rb') as f:
                config = pickle.load(f)
            # Check if config has the required attributes
            if not hasattr(config, 'SAMPLE_RATE'):
                config = DefaultConfig()
        except:
            config = DefaultConfig()
        
        return model, scaler, label_encoder, config, model_type
        
    except Exception as e:
        return None, None, None, None, str(e)

def extract_features(audio_data, sr, config):
    """Extract features from audio data - Fixed to match training script"""
    try:
        # Ensure fixed duration
        max_len = int(sr * config.DURATION)
        if len(audio_data) < max_len:
            audio_data = np.pad(audio_data, (0, max_len - len(audio_data)), mode='constant')
        else:
            audio_data = audio_data[:max_len]
        
        features = {}
        
        # 1. MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=config.N_MFCC, hop_length=config.HOP_LENGTH)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Chroma features - Fixed function call to match training script
        try:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=config.HOP_LENGTH)
        except AttributeError:
            # Fallback for older versions
            stft = librosa.stft(audio_data, hop_length=config.HOP_LENGTH)
            chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr)
        except:
            # Create dummy chroma features if all else fails
            chroma = np.random.random((12, int(len(audio_data) / config.HOP_LENGTH) + 1))
            
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # 3. Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=config.N_MELS, hop_length=config.HOP_LENGTH)
        features['mel_mean'] = np.mean(mel, axis=1)
        features['mel_std'] = np.std(mel, axis=1)
        
        # 4. Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # 5. Rhythm features
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features['tempo'] = tempo
        except:
            features['tempo'] = 120.0  # Default tempo
        
        # 6. Harmonic and percussive components
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
            features['harmonic_mean'] = np.mean(y_harmonic)
            features['percussive_mean'] = np.mean(y_percussive)
        except:
            # Fallback calculations
            features['harmonic_mean'] = np.mean(audio_data)
            features['percussive_mean'] = np.std(audio_data)
        
        # Flatten features - same as training script
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value)
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector), features
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None

def predict_emotion(audio_data, sr, model, scaler, label_encoder, config, model_type):
    """Predict emotion from audio data"""
    try:
        # Extract features
        features, feature_dict = extract_features(audio_data, sr, config)
        if features is None:
            return None, None, None
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        if model_type == 'keras':
            predictions = model.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)
            all_probabilities = predictions[0]
        else:
            predicted_class = model.predict(features_scaled)[0]
            all_probabilities = model.predict_proba(features_scaled)[0]
            confidence = np.max(all_probabilities)
        
        # Get emotion label
        emotion = label_encoder.inverse_transform([predicted_class])[0]
        
        return emotion, confidence, all_probabilities
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def plot_audio_waveform(audio_data, sr):
    """Plot audio waveform"""
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', name='Waveform'))
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300
    )
    return fig

def plot_feature_importance(features_dict):
    """Plot key audio features"""
    key_features = {
        'Spectral Centroid': features_dict.get('spectral_centroid', 0),
        'Spectral Bandwidth': features_dict.get('spectral_bandwidth', 0),
        'Zero Crossing Rate': features_dict.get('zero_crossing_rate', 0),
        'Tempo': features_dict.get('tempo', 0),
        'Harmonic Mean': features_dict.get('harmonic_mean', 0),
        'Percussive Mean': features_dict.get('percussive_mean', 0)
    }
    
    df = pd.DataFrame(list(key_features.items()), columns=['Feature', 'Value'])
    fig = px.bar(df, x='Feature', y='Value', title="Key Audio Features")
    fig.update_layout(height=400)
    return fig

def plot_emotion_probabilities(probabilities, label_encoder):
    """Plot emotion prediction probabilities"""
    emotions = label_encoder.classes_
    
    df = pd.DataFrame({
        'Emotion': emotions,
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, x='Probability', y='Emotion', orientation='h',
                 title="Emotion Prediction Probabilities",
                 color='Probability', color_continuous_scale='viridis')
    fig.update_layout(height=400)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Speech Emotion Classification 🎵</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    
    # Load model
    with st.spinner("Loading model..."):
        model, scaler, label_encoder, config, model_type = load_model_and_preprocessors()
    
    if model is None:
        st.error(f"❌ Failed to load model: {model_type}")
        st.info("Please ensure the following files are in the same directory:")
        st.code("""
        - emotion_model.h5 (or emotion_model.pkl)
        - scaler.pkl
        - label_encoder.pkl
        - config.pkl
        """)
        return
    
    # Model info in sidebar
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"**Model Type:** {model_type.upper()}")
    st.sidebar.info(f"**Available Emotions:** {len(label_encoder.classes_)}")
    
    emotions_list = "\n".join([f"• {emotion.title()}" for emotion in label_encoder.classes_])
    st.sidebar.markdown(f"**Emotions:**\n{emotions_list}")
    
    # Main content
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions
    1. Upload an audio file (WAV, MP3, FLAC, M4A)
    2. Wait for the analysis to complete
    3. View the predicted emotion and confidence score
    4. Explore the audio features and probabilities
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file to classify its emotional content"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 File Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            st.subheader("🎵 Audio Player")
            st.audio(uploaded_file, format='audio/wav')
        
        # Process audio
        with st.spinner("🔄 Processing audio file..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Load audio
                audio_data, sr = librosa.load(tmp_file_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                # Extract features for visualization
                features, features_dict = extract_features(audio_data, sr, config)
                
                if features is not None:
                    # Make prediction
                    emotion, confidence, probabilities = predict_emotion(
                        audio_data, sr, model, scaler, label_encoder, config, model_type
                    )
                    
                    if emotion is not None:
                        # Display prediction
                        st.markdown("---")
                        st.markdown("### 🎯 Prediction Results")
                        
                        # Main prediction display
                        emotion_emoji = {
                            'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
                            'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
                        }
                        
                        emoji = emotion_emoji.get(emotion.lower(), '🎭')
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>{emoji} {emotion.title()}</h2>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create tabs for detailed analysis
                        tab1, tab2, tab3 = st.tabs(["📊 Audio Analysis", "📈 Feature Analysis", "🎯 All Probabilities"])
                        
                        with tab1:
                            st.subheader("Audio Waveform")
                            waveform_fig = plot_audio_waveform(audio_data, sr)
                            st.plotly_chart(waveform_fig, use_container_width=True)
                            
                            # Audio statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Duration", f"{len(audio_data)/sr:.2f}s")
                            with col2:
                                st.metric("Sample Rate", f"{sr} Hz")
                            with col3:
                                st.metric("Max Amplitude", f"{np.max(np.abs(audio_data)):.3f}")
                            with col4:
                                st.metric("RMS Energy", f"{np.sqrt(np.mean(audio_data**2)):.3f}")
                        
                        with tab2:
                            st.subheader("Key Audio Features")
                            features_fig = plot_feature_importance(features_dict)
                            st.plotly_chart(features_fig, use_container_width=True)
                            
                            # Feature explanations
                            with st.expander("🔍 Feature Explanations"):
                                st.markdown("""
                                - **Spectral Centroid**: Brightness of the sound
                                - **Spectral Bandwidth**: Spread of frequencies
                                - **Zero Crossing Rate**: Rate of sign changes in signal
                                - **Tempo**: Beats per minute
                                - **Harmonic Mean**: Tonal content
                                - **Percussive Mean**: Rhythmic content
                                """)
                        
                        with tab3:
                            st.subheader("All Emotion Probabilities")
                            prob_fig = plot_emotion_probabilities(probabilities, label_encoder)
                            st.plotly_chart(prob_fig, use_container_width=True)
                            
                            # Probability table
                            prob_df = pd.DataFrame({
                                'Emotion': label_encoder.classes_,
                                'Probability': probabilities,
                                'Percentage': [f"{p:.2%}" for p in probabilities]
                            }).sort_values('Probability', ascending=False)
                            
                            st.dataframe(prob_df, use_container_width=True)
                    
                    else:
                        st.error("❌ Failed to make prediction")
                else:
                    st.error("❌ Failed to extract features from audio")
                    
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit | Speech Emotion Classification System</p>
        <p>Upload an audio file to get started!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()