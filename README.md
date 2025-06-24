# emotion-classification
Based on Machine Learning and Deep Learning

Emotion Classification from Speech - README
Project Overview
This repository provides a comprehensive pipeline for speech emotion classification using deep learning and ensemble machine learning models. The system processes raw audio files, extracts relevant acoustic features, and classifies them into one of several emotional states. The solution includes:

Jupyter notebooks for model training (DNN, XGBoost, etc.)

A Python script for model testing on new audio files

A Streamlit web application for interactive emotion prediction

All necessary model artifacts and configuration files

Directory Structure
text
emotion-classification/
├── complete ipynb(DNN).ipynb
├── complete ipynb(Xgboost).ipynb         # Main training notebooks
├── test_model.py                         # Testing script for batch/single audio
├── app.py                                # Streamlit web application
├── README.md                             # This file
├── requirements.txt                      # Dependencies
├── trained models/                       # Trained models and preprocessors
│   ├── model used in test_model.py and app.py/
│   │   ├── emotion_model.h5              # Keras model
│   │   ├── scaler.pkl                    # Feature scaler
│   │   ├── label_encoder.pkl             # Label encoder
│   │   └── config.pkl                    # Configuration
│   ├── model can be used in test_model.py and app.py/
│   │   ├── best_emotion_model.h5         # Enhanced Keras model
│   │   ├── enhanced_scaler.pkl           # Enhanced feature scaler
│   │   ├── enhanced_label_encoder.pkl    # Enhanced label encoder
│   │   ├── enhanced_config.pkl           # Enhanced configuration
│   │   └── feature_selector.pkl          # Feature selection object
└── demo/
    └── demo_video.mp4                    # Demo video
Project Description
This project aims to accurately classify emotions from speech signals. The system supports multiple model architectures (DNN, XGBoost, Random Forest, etc.) and is designed for robust performance across diverse emotional classes. The workflow includes:

Feature extraction from raw audio

Data preprocessing and augmentation

Model training and evaluation

Deployment via command-line and web interfaces

Pre-processing Methodology
Audio Pre-processing Steps:

Audio Loading & Trimming:

All audio files are loaded at a consistent sample rate (default: 22050 Hz) and trimmed/padded to a fixed duration (default: 3 seconds).

Feature Extraction:

MFCCs: Mean and standard deviation of Mel-frequency cepstral coefficients.

Chroma Features: Mean and standard deviation to capture pitch class information.

Mel Spectrogram: Mean and standard deviation for detailed spectral representation.

Spectral Features: Centroid, bandwidth, rolloff, and zero-crossing rate.

Rhythm Features: Estimated tempo.

Harmonic/Percussive Components: Mean values for both.

Feature Scaling:

Features are standardized using a StandardScaler to ensure zero mean and unit variance.

Label Encoding:

Emotion labels are encoded numerically using a LabelEncoder.

Feature Selection (Enhanced Models):

Advanced pipelines use feature selection techniques to retain the most informative features, improving model efficiency and accuracy.

Data Splitting:

Data is split into training, validation, and test sets with stratification to maintain class balance.

Reference:
These steps are consistent with best practices in speech emotion recognition and are supported by recent literature, which highlights the importance of log-Mel spectrograms and multi-feature extraction for robust emotion classification.

Model Pipeline
1. Training Notebooks (complete ipynb(DNN).ipynb, complete ipynb(Xgboost).ipynb):

Data Preparation: Loads and preprocesses the RAVDESS dataset (or similar), extracts features, and encodes labels.

Baseline Model: Random Forest classifier for initial benchmarking.

Deep Neural Network (DNN):

Multi-layer dense network with dropout regularization.

Trained with early stopping and learning rate scheduling.

Ensemble Models:

XGBoost, Gradient Boosting, and enhanced Random Forest.

Feature selection and class weighting for improved performance.

Evaluation:

Accuracy, F1-score, per-class accuracy, and confusion matrix.

Models are compared against strict criteria (see below).

2. Testing Script (test_model.py):

Loads trained model and preprocessors.

Accepts single audio files or directories for batch prediction.

Outputs predicted emotion and confidence score for each file.

3. Web Application (app.py):

Built with Streamlit for interactive use.

Allows users to upload audio files and view predictions, probabilities, and feature visualizations.

4. Model Artifacts:

All trained models, scalers, encoders, and configuration objects are saved in the trained models/ directory for reproducibility and deployment.

Accuracy Metrics
Evaluation Criteria
Overall Accuracy: > 80%

Weighted F1-Score: > 80%

Per-Class Accuracy: > 75% for every emotion class

Sample Results (DNN Example)
Validation Accuracy: ~65%

Test Accuracy: ~63%

Validation F1-Score: ~65%

Per-Class Accuracy:

calm: 88%

angry: 70%

happy: 67%

(lower for some classes; see confusion matrix in notebook)

Note: While the DNN model achieves strong performance on several classes, some classes remain challenging due to data imbalance or overlapping features. Enhanced pipelines using feature selection and ensemble methods can further improve these metrics, as discussed in the advanced notebook.

How to Use
1. Install Dependencies
bash
pip install -r requirements.txt
2. Train a Model
Open and run complete ipynb(DNN).ipynb or complete ipynb(Xgboost).ipynb in Jupyter or Colab.

Follow the notebook instructions for data preparation, training, and evaluation.

Trained models and preprocessors will be saved in the trained models/ directory.

3. Test the Model
bash
python test_model.py --audio path/to/audio.wav \
    --model trained models/model used in test_model.py and app.py/emotion_model.h5 \
    --scaler trained models/model used in test_model.py and app.py/scaler.pkl \
    --encoder trained models/model used in test_model.py and app.py/label_encoder.pkl \
    --config trained models/model used in test_model.py and app.py/config.pkl
For batch prediction, provide a directory to --audio.

4. Run the Web App
bash
streamlit run app.py
Upload an audio file and view the predicted emotion, probabilities, and feature analysis.

Demo
A demonstration video is available in the demo/ folder (demo_video.mp4).

References
[Searching for Effective Preprocessing Method and CNN-based Architecture with Efficient Channel Attention on Speech Emotion Recognition]

[Speech emotion classification using attention based network and regularized feature selection]

Project code and methodology are consistent with current best practices in the field.

License
This project is released under the MIT License.
