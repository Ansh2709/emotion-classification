# emotion-classification
Based on Machine Learning and Deep Learning

Emotion Classification from Speech - README
Project Overview
This repository provides a comprehensive pipeline for speech emotion classification using deep learning and ensemble machine learning models. The system processes raw audio files, extracts relevant acoustic features, and classifies them into one of several emotional states. The solution includes:

Jupyter notebooks for model training (DNN, XGBoost, etc.)

A Python script for model testing on new audio files

A Streamlit web application for interactive emotion prediction

All necessary model artifacts and configuration files

![Project Flow](https://github.com/user-attachments/assets/edd713f2-e662-43df-9c7e-c0f87ec96f7f)




Directory Structure




![Screenshot 2025-06-25 023125](https://github.com/user-attachments/assets/2fd8bd26-7843-4b4c-a997-73649a352dd6)


# 🎤 Speech Emotion Recognition System

> A sophisticated machine learning system for classifying emotions from speech signals with multiple model architectures and robust performance across diverse emotional classes.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

---

## 🌟 Features

- **Multi-Architecture Support**: DNN, XGBoost, Random Forest, and ensemble methods
- **Advanced Feature Extraction**: MFCCs, Chroma, Mel Spectrogram, and spectral features
- **Real-time Processing**: Command-line interface and interactive web application
- **Robust Performance**: Optimized for diverse emotional classes with data augmentation
- **Production Ready**: Complete pipeline from preprocessing to deployment

---

## 🏗️ System Architecture

```
📁 Raw Audio Input
    ↓
🔧 Audio Preprocessing (22.05kHz, 3s duration)
    ↓
🎯 Feature Extraction (MFCCs, Chroma, Spectral)
    ↓
⚡ Model Training (DNN/XGBoost/RF)
    ↓
🚀 Deployment (CLI/Web Interface)
```

---

## 🔬 Preprocessing Pipeline

### Audio Processing Chain

| Step | Description | Configuration |
|------|-------------|---------------|
| **🎵 Audio Loading** | Consistent sample rate loading | 22,050 Hz |
| **✂️ Duration Normalization** | Trim/pad to fixed length | 3 seconds |
| **📊 Feature Extraction** | Multi-domain feature extraction | 13 MFCCs + spectral |
| **⚖️ Standardization** | Zero mean, unit variance scaling | StandardScaler |
| **🏷️ Label Encoding** | Numerical emotion mapping | LabelEncoder |

### Extracted Features

<details>
<summary><strong>📈 Spectral Features</strong></summary>

- **MFCCs**: Mean & std of Mel-frequency cepstral coefficients
- **Chroma Features**: Pitch class information (mean & std)
- **Mel Spectrogram**: Detailed spectral representation
- **Spectral Centroid**: Brightness indicator
- **Spectral Bandwidth**: Frequency range measure
- **Spectral Rolloff**: 85% energy threshold
- **Zero Crossing Rate**: Speech/music discrimination

</details>

<details>
<summary><strong>🎼 Musical Features</strong></summary>

- **Tempo Estimation**: Rhythm analysis
- **Harmonic Components**: Tonal content
- **Percussive Components**: Transient detection

</details>

---

## 🤖 Model Architectures

### 🧠 Deep Neural Network (DNN)
```python
Architecture: Multi-layer dense network
Regularization: Dropout layers
Optimization: Early stopping + learning rate scheduling
Performance: ~65% validation accuracy
```

### 🌲 Ensemble Methods
```python
Models: XGBoost, Gradient Boosting, Enhanced Random Forest
Features: Automated feature selection
Optimization: Class weighting for imbalanced data
```

### 📊 Baseline
```python
Model: Random Forest Classifier
Purpose: Initial benchmarking and comparison
```

---

## 📈 Performance Metrics

### 🎯 Evaluation Criteria

| Metric | Target | Status |
|--------|---------|---------|
| **Overall Accuracy** | > 80% | 🔄 In Progress |
| **Weighted F1-Score** | > 80% | 🔄 In Progress |
| **Per-Class Accuracy** | > 75% | 🔄 In Progress |

### 🏆 Current Results (DNN)

| Emotion | Accuracy | Performance |
|---------|----------|-------------|
| **Calm** | 88% | ✅ Excellent |
| **Angry** | 70% | ⚠️ Good |
| **Happy** | 67% | ⚠️ Good |
| **Others** | Varies | 🔄 Improving |

> **Note**: Enhanced pipelines with feature selection and ensemble methods show improved performance across all emotion classes.

---

## 🚀 Quick Start

### 1. 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd speech-emotion-recognition

# Install dependencies
pip install -r requirements.txt
```

### 2. 🎓 Training

#### Option A: Deep Neural Network
```bash
# Open Jupyter notebook
jupyter notebook "complete ipynb(DNN).ipynb"
```

#### Option B: XGBoost Ensemble
```bash
# Open Jupyter notebook
jupyter notebook "complete ipynb(Xgboost).ipynb"
```

### 3. 🧪 Testing

#### Single File Prediction
```bash
python test_model.py \
    --audio path/to/audio.wav \
    --model trained_models/emotion_model.h5 \
    --scaler trained_models/scaler.pkl \
    --encoder trained_models/label_encoder.pkl \
    --config trained_models/config.pkl
```

#### Batch Processing
```bash
python test_model.py --audio path/to/directory/
```

### 4. 🌐 Web Application

```bash
# Launch interactive web interface
streamlit run app.py
```

**Features:**
- 📤 Audio file upload
- 📊 Real-time emotion prediction
- 📈 Probability visualization
- 🔍 Feature analysis dashboard

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── 📓 complete ipynb(DNN).ipynb          # Deep learning training
├── 📓 complete ipynb(Xgboost).ipynb     # Ensemble methods training
├── 🐍 test_model.py                      # CLI testing interface
├── 🌐 app.py                            # Streamlit web application
├── 📋 requirements.txt                   # Dependencies
├── 📁 trained_models/                    # Model artifacts
│   ├── emotion_model.h5                 # Trained model
│   ├── scaler.pkl                       # Feature scaler
│   ├── label_encoder.pkl                # Label encoder
│   └── config.pkl                       # Configuration
├── 📁 demo/                             # Demonstration materials
│   └── demo_video.mp4                   # Usage demonstration
└── 📖 README.md                         # This file
```

---

## 🎬 Demo

🎥 **Check out our demonstration video**: `demo/demo_video.mp4`

The demo showcases:
- Real-time emotion classification
- Web interface functionality
- Model performance across different emotions
- Feature visualization capabilities

---

## 🔬 Technical References

Our methodology aligns with cutting-edge research in speech emotion recognition:

1. **"Searching for Effective Preprocessing Method and CNN-based Architecture with Efficient Channel Attention on Speech Emotion Recognition"**
   - Advanced preprocessing techniques
   - CNN architecture optimization

2. **"Speech emotion classification using attention based network and regularized feature selection"**
   - Attention mechanisms
   - Feature selection strategies

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

### Development Guidelines
- Follow PEP 8 style conventions
- Add unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- RAVDESS dataset contributors
- Open source community for amazing libraries
- Research community for advancing speech emotion recognition

---

<div align="center">

**Built with ❤️ for the speech processing community**

[⭐ Star this repo](../../stargazers) • [🐛 Report Bug](../../issues) • [💡 Request Feature](../../issues)

</div>




