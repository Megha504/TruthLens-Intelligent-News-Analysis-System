import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Safe stopwords loading
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load model artifacts
model = joblib.load("category_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

bias_model = joblib.load("bias_model.pkl")
bias_vectorizer = joblib.load("bias_vectorizer.pkl")
bias_encoder = joblib.load("bias_encoder.pkl")

# Load clustering model and vectorizer
clustering_model = joblib.load("kmeans_model.pkl")  # trained KMeans model
clustering_vectorizer = joblib.load("vectorizer.pkl")  # same TF-IDF used for clustering
# Load dataset for opinion label lookup
df = pd.read_excel("data/bias_dataset.xlsx")
df["full_text"] = df["sentence"].fillna("") + " " + df["article"].fillna("")
df["clean_text"] = df["full_text"].apply(lambda text: " ".join([
    word for word in re.sub(r"[^a-z\s]", "", text.lower()).split()
    if word not in stop_words
]))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([word for word in text.split() if word not in stop_words])

# Streamlit UI
st.set_page_config(page_title="News Category & Bias Classifier", layout="centered")
st.title("📰 News Category & Bias Classifier")
st.markdown("Enter a **headline** and optional **short description** to predict the news category and bias.")

# Input
headline = st.text_input("Headline")
description = st.text_area("Short Description (optional)")

if st.button("Classify"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        full_text = headline + " " + description
        cleaned = clean_text(full_text)

        # Category Prediction
        X_input = vectorizer.transform([cleaned])
        pred = model.predict(X_input)[0]
        label = encoder.inverse_transform([pred])[0]

        try:
            proba = model.predict_proba(X_input)[0]
            confidence = np.max(proba) * 100
            st.success(f"**Predicted Category:** {label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        except:
            st.success(f"**Predicted Category:** {label}")
            st.info("Confidence score not available for this model.")

        # Bias Detection — ✅ FIXED LINE
        bias_input = bias_vectorizer.transform([cleaned])
        bias_pred = bias_model.predict([cleaned])[0]


        try:
            bias_proba = bias_model.predict_proba([cleaned])[0]
            bias_confidence = np.max(bias_proba) * 100
            st.info(f"**Bias Prediction:** {bias_pred} ({bias_confidence:.2f}%)")
        except:
            st.info(f"**Bias Prediction:** {bias_pred}")

        # Opinion Label based on bias prediction
        bias_opinion_map = {
            "biased": "Contains subjective or emotionally charged language.",
            "non-biased": "Factual and neutral in tone.",
            "neutral": "Balanced reporting with minimal bias.",
            "unknown": "Bias could not be determined."
        }

        opinion_label = bias_opinion_map.get(bias_pred.lower(), "Bias could not be interpreted.")

        # Color-coded bias styling
        bias_colors = {
            "biased": "#e74c3c",
            "non-biased": "#2ecc71",
            "neutral": "#3498db"
        }
        bias_color = bias_colors.get(bias_pred.lower(), "#888")
        
        # --- Factuality Detection ---
        factuality_model = joblib.load("factuality_model.pkl")
        factuality_vectorizer = joblib.load("factuality_vectorizer.pkl")
        factuality_encoder = joblib.load("factuality_encoder.pkl")

        factuality_input = factuality_vectorizer.transform([cleaned])
        factuality_pred = factuality_model.predict(factuality_input)[0]
        factuality_label = factuality_encoder.inverse_transform([factuality_pred])[0]

        try:
            factuality_proba = factuality_model.predict_proba(factuality_input)[0]
            factuality_confidence = np.max(factuality_proba) * 100
            st.info(f"**Factuality Prediction:** {factuality_label} ({factuality_confidence:.2f}%)")
        except:
            st.info(f"**Factuality Prediction:** {factuality_label}")

        factuality_colors = {
            "factual": "#2ecc71",
            "non-factual": "#e74c3c",
            "mixed": "#f39c12"
        }
        factuality_opinion_map = {
        "factual": "Likely accurate and supported by evidence.",
        "non-factual": "Contains false or misleading information.",
        "mixed": "Partially true but may include inaccuracies.",
        "unknown": "Factuality could not be determined."
    }
        factuality_opinion = factuality_opinion_map.get(factuality_label.lower(), "Factuality could not be interpreted.")
        factuality_color = factuality_colors.get(factuality_label.lower(), "#888")
        # Final result card
        st.markdown(
            f"""
            <div style='background-color:#f0f2f6;padding:20px;border-radius:10px;border-left:5px solid {bias_color}'>
                <h3 style='color:#4CAF50'>Prediction: {label}</h3>
                <p><strong>Headline:</strong> {headline}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Bias:</strong> {bias_pred}</p>
                <p><strong>Bias Interpretation:</strong> {opinion_label}</p>
                <p><strong>Factuality:</strong> {factuality_label}</p>
                <p><strong>Factuality Interpretation:</strong> {factuality_opinion}</p>
            </div>
            """,
            unsafe_allow_html=True
        )