import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


DATASET_PATH = "AUDIO DATASET 24"  
OUTPUT_CSV = "ravdess_features.csv"


def extract_features(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr])

def parse_emotion_from_filename(fname):
    parts = fname.split('-')
    emo_code = int(parts[2])
    emotions = {1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fear',7:'disgust',8:'surprise'}
    return emotions.get(emo_code, 'unknown')

rows = []
for root, _, files in os.walk(DATASET_PATH):
    for f in tqdm(files, desc="Extracting features"):
        if f.endswith(".wav"):
            full_path = os.path.join(root, f)
            features = extract_features(full_path)
            emotion = parse_emotion_from_filename(f)
            rows.append([f, emotion] + features.tolist())

# Column names
cols = ['file', 'emotion'] + [f'mfcc_{i}' for i in range(40)] + \
       [f'chroma_{i}' for i in range(12)] + \
       [f'contrast_{i}' for i in range(7)] + ['zcr']

# Save CSV
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Features saved to {OUTPUT_CSV}")
