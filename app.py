import streamlit as st
import numpy as np
import pandas as pd
import librosa, joblib, shap, base64, os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ID = "Pooja001/SpeechEmotionNet"

st.set_page_config(page_title="SpeechEmotionNet — SER", layout="wide")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    margin-top: 25px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 16px !important;
    padding: 8px 16px !important;
}
.custom-image {
    width: 75%;
    margin-left: auto;
    margin-right: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

st.title(" SpeechEmotionNet — Speech Emotion Recognition")

# Load models from Hugging face
@st.cache_resource
def load_assets():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="SpeechEmotionNet_best.keras")
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.joblib")
    encoder_path = hf_hub_download(repo_id=REPO_ID, filename="encoder.joblib")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    return model, scaler, encoder

model, scaler, encoder = load_assets()

def extract_features_from_array(y, sr=16000):
    y = librosa.util.normalize(y)

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfccs, chroma, contrast, zcr])


tab_pred, tab_explain, tab_eval = st.tabs([
    "Prediction",
    "Explainability (SHAP)",
    "Evaluation Metrics"
])

# Prediction
with tab_pred:

    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file is not None:

        y, sr = librosa.load(uploaded_file, sr=16000)
        y = librosa.util.normalize(y)

        st.audio(uploaded_file)

        features = extract_features_from_array(y, sr)
        X = scaler.transform([features])

        probs = model.predict(X, verbose=0)[0]
        top_two = np.argsort(probs)[-2:][::-1]
        top1, top2 = top_two[0], top_two[1]
        p1, p2 = probs[top1], probs[top2]

        pred_label = encoder.inverse_transform([top1])[0]

        st.subheader("Prediction Result")

        if p1 < 0.6 and (p1 - p2) < 0.15:
            st.warning(
                f"Ambiguous prediction between {encoder.inverse_transform([top1])[0].upper()} ({p1:.2f}) "
                f"and {encoder.inverse_transform([top2])[0].upper()} ({p2:.2f})"
            )
        else:
            st.success(f"Predicted Emotion: {pred_label.upper()} ({p1:.2f} confidence)")

        # Probability bar plot
        prob_df = pd.DataFrame({"Emotion": encoder.classes_, "Probability": probs}).set_index("Emotion")
        st.bar_chart(prob_df)

        # CSV Download
        csv_buffer = BytesIO()
        prob_df.to_csv(csv_buffer)
        b64 = base64.b64encode(csv_buffer.getvalue()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="emotion_probs.csv">Download Prediction CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


# SHAP - explainability
with tab_explain:

    st.subheader("SHAP Explainability for Uploaded Audio")

    uploaded_file2 = st.file_uploader("Upload WAV for SHAP", type=["wav"], key="shap_upload")

    if uploaded_file2 is not None:

        y, sr = librosa.load(uploaded_file2, sr=16000)
        y = librosa.util.normalize(y)

        features = extract_features_from_array(y, sr)
        X = scaler.transform([features])

        probs = model.predict(X)[0]
        pred_class = np.argmax(probs)
        pred_label = encoder.inverse_transform([pred_class])[0]

        st.info(f"SHAP explanation for: {pred_label.upper()}")

        background = X + np.random.normal(0, 0.01, X.shape)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[pred_class]

        shap_arr = np.array(shap_values).reshape(-1)

        feature_names = (
            [f"mfcc_{i}" for i in range(40)] +
            [f"chroma_{i}" for i in range(12)] +
            [f"contrast_{i}" for i in range(7)] +
            ["zcr"]
        )

        min_len = min(len(feature_names), len(shap_arr))
        df_shap = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "ABS_SHAP": np.abs(shap_arr[:min_len])
        }).sort_values("ABS_SHAP", ascending=False)

        st.subheader("Top 10 Most Influential Features")
        st.dataframe(df_shap.head(10))

        # Plot
        fig, ax = plt.subplots(figsize=(6,5))
        ax.barh(df_shap.head(10)["Feature"], df_shap.head(10)["ABS_SHAP"])
        ax.invert_yaxis()
        ax.set_title("Top 10 Influential Features")
        st.pyplot(fig, use_container_width=True)


# Visuals
with tab_eval:

    st.subheader("Model Evaluation Metrics")

    def show_hf_visual(filename, caption):
        try:
            file_path = hf_hub_download(repo_id=REPO_ID, filename=f"visuals/{filename}")
            encoded = base64.b64encode(open(file_path, "rb").read()).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{encoded}" class="custom-image">',
                unsafe_allow_html=True
            )
            st.caption(caption)
        except:
            st.warning(f"{filename} not found in HuggingFace repo.")

    with st.expander("Technical Plots", expanded=False):
        show_hf_visual("roc_curve.png", "ROC Curve (Best Fold)")
        show_hf_visual("confusion_matrix.png", "Confusion Matrix")
        show_hf_visual("cv_accuracy.png", "Cross-Validation Accuracy")
