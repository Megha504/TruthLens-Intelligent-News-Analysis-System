# TruthLens: Intelligent News Analysis System

TruthLens is a state-of-the-art Machine Learning and NLP application built to elevate digital transparency. Designed as an academic capstone, the system processes news headlines and text descriptions to automatically predict topic categories, estimate linguistic factuality, detect subjective bias, and group news articles inside a streamlined portal

### Key Capabilities

* **News Categorization (Supervised NLP):** Classifies complex articles into core high-level themes (e.g., Politics, Business/Finance, Technology, Entertainment, Science, Lifestyle, Sports) using TF-IDF bigrams and calibrated vector support.
* **Factuality & Veracity Prediction:** Evaluates linguistic patterns, hedges, and lexical structures to predict whether content is *Factual*, *Non-Factual*, or *Mixed*.
* **Subjective Bias Analysis:** Identifies whether an article's tone is *Neutral* or *Biased* (subjective/polarizing) to highlight potential media framing.
* **Topical Clustering (Unsupervised KMeans):** Automates semantic discovery by placing articles into mathematical topic clusters, displaying high-frequency keywords for contextual framing.

### Machine Learning & Preprocessing Pipeline

* **Text Standardization:** Converts input payloads to lowercase, strips URLs/hyperlinks, eliminates numerical/special punctuation, and filters out common English stopwords.
* **Feature Engineering:** Extracts structural patterns using standard $N$-gram ranges (Unigrams and Bigrams) with a maximum feature threshold of 5,000 using `TfidfVectorizer` ($min\_df = 3$).
*   **Supervised Model Core:** Leverages a `LinearSVC` (Support Vector Classifier) wrapped inside a `CalibratedClassifierCV` to enable robust probabilistic outputs alongside hard-margin category boundaries.
* **Unsupervised Topic Modeling:** Uses a trained KMeans model coupled with custom centroid extraction to generate keyword highlights for semantic context.

### Tech Stack & Core Libraries

* **User Interface:** Streamlit Engine (Dynamic Data App Framework)
* **Model Management:** Joblib (Binary serialization & deserialization)
* **Scientific Computing:** NumPy, Pandas, Scikit-Learn
* **Natural Language Processing:** NLTK (Natural Language Toolkit)

## Project Structure
```text
D:\AIML_PROJECT\
│
├── .venv/
│
├── data/
│   ├── bias_dataset.xlsx
│   ├── combined_dataset.json
│   ├── News_Category_Dataset_v3.json
│   ├── tech_articles.jsonl
│   ├── test.tsv
│   ├── train.tsv
│   └── valid.tsv
│
├── .gitattributes
├── 0.26.0'
├── app.py
├── bias_encoder.pkl
├── bias_model.pkl
├── bias_vectorizer.pkl
├── category_model.pkl
├── cluster_keywords.pkl
├── clustered_df.pkl
├── clustering_vectorizer.pkl
├── factuality_encoder.pkl
├── factuality_model.pkl
├── factuality_vectorizer.pkl
├── kmeans_model.pkl
├── label_encoder.pkl
├── news.ipynb
├── requirements.txt
├── tfidf_vectorizer.pkl
└── vectorizer.pkl
```
## Installation & Local Setup
1. Clone the Repository
```text
git clone https://github.com/Megha504/TruthLens-Intelligent-News-Analysis-System.git
cd TruthLens-Intelligent-News-Analysis-System
```
2. Set Up a Virtual Environment
```text
# Windows
python -m venv venv
venv\Scripts\activate
```
3. Install Project Dependencies
Install all required libraries mapped in your requirements manifest:
```text
pip install -r requirements.txt
```
4. Download NLTK Text Corpora
Ensure your local environment has the required NLTK stopword dataset:
```text
python -c "import nltk; nltk.download('stopwords')"
```
5. Start the Web Dashboard
Fire up the local Streamlit application server using Python:
```text
python -m streamlit run app.py
```
Open your web browser and navigate to http://localhost:8501 to interact with TruthLens!
