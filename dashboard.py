import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config and Style 
st.set_page_config(page_title="Hyperlocal News Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        /* General Background and Font */
        body {
            background-color: #f9fafc;
            color: #222;
        }
        .main {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #004080;
        }
        .sidebar .sidebar-content {
            background-color: #f1f3f6;
            border-right: 2px solid #e0e0e0;
        }
        .metric-box {
            background-color: #e9f5ff;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            color: #004080;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        /* --- Article Reader Styling --- */
        .article-reader {
            background: linear-gradient(135deg, #fdfdfd 0%, #e6f0ff 100%);
            color: #1a1a1a;
            border-left: 6px solid #4a90e2;
            padding: 18px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 15px;
            line-height: 1.6;
            box-shadow: 0 3px 8px rgba(0,0,0,0.05);
            opacity: 0;
            animation: fadeIn 1s forwards;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Title 
st.markdown("<h1 style='text-align:center;'>🌐 Hyperlocal News Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Interactive Analysis of Topics, Sentiment, and Anomalies in Local News</p>", unsafe_allow_html=True)
st.markdown("---")

#  Load Data 
DATA_OUT = r"C:\Users\HP\Documents\Data science\Hyperlocal_News_Anomaly_Detection\data\Articles_processed.csv"
df = pd.read_csv(DATA_OUT)

df['geotext_countries_list'] = df['geotext_countries'].fillna("").apply(lambda x: [c.strip() for c in x.split(",") if c.strip()])
df['geotext_cities_list']    = df['geotext_cities'].fillna("").apply(lambda x: [c.strip() for c in x.split(",") if c.strip()])

#  Sidebar Filters 
st.sidebar.header("🎯 Filter News Data")

def add_all_option(options):
    return ["All"] + sorted(list(set([opt for opt in options if opt and pd.notna(opt)])))

unique_cities      = add_all_option(df['geotext_cities'].dropna().unique())
unique_countries   = add_all_option(df['geotext_countries'].dropna().unique())
unique_topics      = add_all_option(df['topic_label'].dropna().unique())
unique_news_types  = add_all_option(df['NewsType_final'].dropna().unique())

selected_city      = st.sidebar.multiselect("🏙️ Select City", unique_cities, default=["All"])
selected_country   = st.sidebar.multiselect("🌍 Select Country", unique_countries, default=["All"])
selected_topic     = st.sidebar.multiselect("🧩 Select Topic", unique_topics, default=["All"])
selected_newstype  = st.sidebar.multiselect("📰 Select News Type", unique_news_types, default=["All"])

search_keyword     = st.sidebar.text_input("🔎 Search by keyword in article")
search_topic       = st.sidebar.text_input("🔖 Search by topic label")

show_anomaly       = st.sidebar.checkbox("⚠️ Show Anomalies Only", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🕵️ Advanced Filters")

min_score    = float(df['anomaly_score'].min())
max_score    = float(df['anomaly_score'].max())
score_range  = st.sidebar.slider("Anomaly Score Range", min_score, max_score, (min_score, max_score))

sent_min     = float(df['sentiment_compound'].min())
sent_max     = float(df['sentiment_compound'].max())
sentiment_range = st.sidebar.slider("Sentiment Score Range", sent_min, sent_max, (sent_min, sent_max))

#  Apply Filters 
df_filtered = df.copy()

if "All" not in selected_city:
    df_filtered = df_filtered[df_filtered['geotext_cities_list'].apply(lambda L: any(c in L for c in selected_city))]

if "All" not in selected_country:
    df_filtered = df_filtered[df_filtered['geotext_countries_list'].apply(lambda L: any(c in L for c in selected_country))]

if "All" not in selected_topic:
    df_filtered = df_filtered[df_filtered['topic_label'].isin(selected_topic)]

if "All" not in selected_newstype:
    df_filtered = df_filtered[df_filtered['NewsType_final'].isin(selected_newstype)]

if search_keyword:
    df_filtered = df_filtered[df_filtered['raw_article'].str.contains(search_keyword, case=False, na=False)]

if search_topic:
    df_filtered = df_filtered[df_filtered['topic_label'].str.contains(search_topic, case=False, na=False)]

df_filtered = df_filtered[
    (df_filtered['anomaly_score'].between(score_range[0], score_range[1])) &
    (df_filtered['sentiment_compound'].between(sentiment_range[0], sentiment_range[1]))
]

if show_anomaly:
    df_filtered = df_filtered[df_filtered['is_anomaly'] == True]

st.markdown(f"<b>Displaying {df_filtered.shape[0]} filtered articles</b>", unsafe_allow_html=True)

#  Metrics 
st.markdown("### 📊 Quick Stats")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-box'><h3>Total Articles</h3><h2>{len(df_filtered)}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box'><h3>Unique Topics</h3><h2>{df_filtered['topic_label'].nunique()}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-box'><h3>Anomalies</h3><h2>{df_filtered['is_anomaly'].sum()}</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-box'><h3>Avg Sentiment</h3><h2>{round(df_filtered['sentiment_compound'].mean(),3)}</h2></div>", unsafe_allow_html=True)

#  Tabs 
tabs = st.tabs(["Overview","Topic Analysis","Sentiment Analysis","Anomaly Analysis","NER Insights","Geo Map"])

# Overview + Article Reader
with tabs[0]:
    st.subheader("📋 Filtered Articles")
    st.dataframe(df_filtered[['raw_article','geotext_cities','geotext_countries','topic_label','NewsType_final','is_anomaly']], height=300)

    st.markdown("---")
    st.subheader("📰 Sample Article Reader")

    if not df_filtered.empty:
        # Dropdown to choose a sample article
        sample_titles = df_filtered['raw_article'].head(10).tolist()
        selected_article = st.selectbox("Select an Article to Read", sample_titles)

        # Display selected article content with improved style
        article_text = df_filtered[df_filtered['raw_article'] == selected_article].iloc[0]['raw_article']
        st.markdown(f"<div class='article-reader'>{article_text}</div>", unsafe_allow_html=True)
    else:
        st.warning("No articles available to display.")

#  Remaining Tabs (Unchanged) 
with tabs[1]:
    st.subheader("📈 Topic Distribution")
    if not df_filtered.empty:
        topic_counts = df_filtered['topic_label'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=topic_counts.values, y=topic_counts.index, palette="Blues_d", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Topic")
        st.pyplot(fig)
    else:
        st.warning("No data available for selected filters.")

with tabs[2]:
    st.subheader("💬 Sentiment Distribution")
    if not df_filtered.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(df_filtered['sentiment_compound'], bins=30, kde=True, color="#80c7ff")
        ax.set_xlabel("Compound Sentiment Score")
        st.pyplot(fig)
        st.write("Average Sentiment Score:", round(df_filtered['sentiment_compound'].mean(),3))
    else:
        st.warning("No data to display.")

with tabs[3]:
    st.subheader("🚨 Anomalies vs Normal")
    if not df_filtered.empty:
        anomaly_counts = df_filtered['is_anomaly'].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=anomaly_counts.index.map({True:"Anomaly", False:"Normal"}), y=anomaly_counts.values, palette=["#ff6b6b","#4caf50"], ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("No anomaly data to show.")

with tabs[4]:
    st.subheader("🔍 Named Entity Recognition Insights")
    if not df_filtered.empty:
        for col in ['ner_gpe','ner_loc','ner_org','ner_person']:
            st.markdown(f"**Top Entities in {col}**")
            top_entities = pd.Series(",".join(df_filtered[col].dropna()).split(",")).value_counts().head(10)
            st.bar_chart(top_entities)
    else:
        st.warning("No NER data available.")

with tabs[5]:
    st.subheader("🗺️ Geographic Visualization (Original Map Preserved)")
    if not df_filtered.empty:
        map_data = df_filtered[['geotext_cities','anomaly_score']].dropna()
        map_data['latitude']  = np.random.uniform(low=20.0, high=30.0, size=len(map_data))
        map_data['longitude'] = np.random.uniform(low=75.0, high=85.0, size=len(map_data))
        st.map(map_data)
    else:
        st.warning("No geographic data to display.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;'>✨ Developed by Azar Deen — Bright Dashboard Edition v5 (Enhanced Reader) ✨</p>", unsafe_allow_html=True)
