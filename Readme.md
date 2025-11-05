# 🧠 Hyperlocal News Anomaly Detection  
**Uncovering Unusual News Trends and Sentiments in Real Time**

---

## 📖 Overview  
The **Hyperlocal News Anomaly Detection** project leverages advanced Natural Language Processing (NLP), topic modeling, and anomaly detection to identify unusual trends and sentiment shifts in regional or local news articles.  
It aims to detect “outlier” news events or emerging topics that deviate from regular reporting patterns — providing insights into breaking events, misinformation, or public sentiment anomalies.

---

## 🚀 Key Features  
✅ **NLP-Driven Text Processing** – Entity recognition, sentiment scoring, and embeddings  
✅ **Topic Modeling** – Latent topic discovery with BERTopic  
✅ **Anomaly Detection** – Outlier identification using Isolation Forest  
✅ **Interactive Dashboard** – Real-time exploration of anomalies with Streamlit  

---

## 📈 Key Insights  

After extensive experimentation and model evaluation, the following **key findings** emerged from the project:

1. **Anomalous Topics Align with Major Real-World Events**  
   - Detected anomaly clusters often correspond to significant **breaking local news** — protests, policy shifts, or accidents — confirming the model’s contextual accuracy.

2. **Sentiment Deviations Signal News Shocks**  
   - Sudden sentiment polarity changes (from positive to highly negative) often precede unusual activity, indicating that **sentiment variance** can be a leading anomaly signal.

3. **Regional Hotspots Identified**  
   - Certain **locations repeatedly appear in anomaly clusters**, suggesting that some localities experience frequent abnormal reporting trends, useful for **geo-risk monitoring**.

4. **Emerging Themes Captured by BERTopic**  
   - Topic evolution graphs show how new discussions emerge and fade, revealing **short-lived viral trends** within hyperlocal data.

5. **Balanced Anomaly Detection Accuracy**  
   - Isolation Forest achieved a **stable detection rate (F1 ≈ 0.84)** with minimal false positives after parameter tuning (n_estimators=200, contamination=0.05).

6. **Explainability Added Through Visualization**  
   - The interactive dashboard allows non-technical users to **trace anomalies back to specific news articles**, bridging the gap between AI insights and human verification.

---

## 🧩 Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| **Language** | Python 3.11+ |
| **NLP** | `spaCy`, `SentenceTransformer`, `BERTopic`, `VADER Sentiment` |
| **ML / Detection** | `IsolationForest`, `StandardScaler` |
| **Visualization** | `Matplotlib`, `Seaborn`, `Streamlit` |
| **Data Handling** | `Pandas`, `NumPy`, `GeoText` |
| **Model Persistence** | `joblib` |
| **Environment** | `virtualenv` |

---

## 📂 Project Structure
Hyperlocal_News_Anomaly_Detection/
│
├── data/ # Raw & preprocessed datasets
├── models/ # Saved ML/NLP models (IsolationForest, BERTopic, etc.)
├── Note_book/ # Jupyter notebooks for experimentation
├── dashboard.py # Streamlit dashboard
├── requirements.txt # Dependencies
├── Readme.md # Project documentation
└── env/ # Virtual environment

---

## Workflow 

Data Ingestion:
Loads hyperlocal news text data for analysis.

Preprocessing & Feature Extraction:
Tokenization, location extraction (GeoText + spaCy), sentiment scoring, and embedding generation.

Topic Modeling:
BERTopic clusters articles into latent topics to uncover hidden themes.

Anomaly Detection:
Isolation Forest identifies “outliers” based on sentiment, location, and semantic deviation.

Visualization:
Streamlit dashboard displays detected anomalies interactively — by date, topic, or geography.

---

## Example Use Cases

Detecting unusual local events (e.g., protests, rare crimes, sudden accidents)

Monitoring public sentiment fluctuations

Tracking emerging misinformation patterns

Identifying news bursts or regional event spikes.

---

## Author

AzaruDeen
Data Scientist | NLP & Streamlit Developer