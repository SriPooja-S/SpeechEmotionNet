# SpeechEmotionNet (Speech Emotion Recognition)

This project performs **Speech Emotion Recognition (SER)** using the **RAVDESS dataset**, extracting audio features (MFCC, Chroma, Spectral Contrast, ZCR) and training an **MLP classifier** with **5-Fold Stratified Cross-Validation**.  
The best-performing model is saved along with evaluation visuals such as **ROC Curve**, **Confusion Matrix**, and **Cross-Validation Accuracy**.

---

## Project Objective
- Build a machine learning model that can **detect emotions from speech audio**.  
- Extract robust audio features using Librosa.  
- Train and evaluate a neural network on the RAVDESS dataset.  
- Provide visual insights into model performance.

---

## Dataset  
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**  
- Official Source: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- (Dataset not included; download manually)

---
## Running the project

    pip install -r requirements.txt
    python mlp_feature_extraction.py
    python mlp_model_training.py
    streamlit run app.py
