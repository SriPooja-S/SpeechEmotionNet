import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


FEATURE_CSV = "ravdess_features.csv"
MODEL_DIR = "models"
VISUALS_DIR = "visuals"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

df = pd.read_csv(FEATURE_CSV)
print("Dataset loaded:", df.shape)

X = df.drop(['file', 'emotion'], axis=1).values
y = df['emotion'].values

# Encoding
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
y_cat = to_categorical(y_enc)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified K-Fold training
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_acc = []
best_val_acc = 0
best_model = None
best_fold_data = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_enc)):
    print(f"\nFold {fold+1}/{k}")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_cat[train_idx], y_cat[val_idx]

    # Build model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        verbose=1,
        callbacks=[es]
    )

    val_acc = max(hist.history['val_accuracy'])
    fold_acc.append(val_acc)
    print(f"Fold {fold+1} Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_fold_data = (X_val, y_val)

print("\nBest Fold Accuracy:", best_val_acc)

# save model
best_model.save(os.path.join(MODEL_DIR, "SpeechEmotionNet_best.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.joblib"))
print("Saved model, scaler, and encoder to 'models/'")

# Plot accuracy per fold
plt.figure(figsize=(6,4))
plt.plot(range(1, k+1), fold_acc, marker='o')
plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig(os.path.join(VISUALS_DIR, "cv_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved accuracy plot.")

# ROC curve and confusion matrix
print("\nGenerating ROC and Confusion Matrix...")

X_val, y_val = best_fold_data
y_true = np.argmax(y_val, axis=1)

# Binarize labels for ROC
y_true_bin = label_binarize(y_true, classes=np.arange(len(encoder.classes_)))

y_score = best_model.predict(X_val)
n_classes = y_true_bin.shape[1]

# -------- ROC CURVE --------
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"{encoder.classes_[i]} (AUC={roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.legend()
plt.title("ROC Curve (Best Fold)")
plt.xlabel("FPR")
plt.ylabel("TPR")

ROC_SAVE = f"{VISUALS_DIR}/roc_curve.png"
plt.savefig(ROC_SAVE, dpi=300, bbox_inches='tight')
plt.close()
print("ROC saved:", ROC_SAVE)

# -------- CONFUSION MATRIX --------
y_pred = np.argmax(y_score, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)

plt.title("Confusion Matrix (Best Fold)")
plt.xlabel("Predicted")
plt.ylabel("True")

CM_SAVE = f"{VISUALS_DIR}/confusion_matrix.png"
plt.savefig(CM_SAVE, dpi=300, bbox_inches='tight')
plt.close()

print("Confusion Matrix saved:", CM_SAVE)
