import streamlit as st
import pandas as pd
import re
from PyPDF2 import PdfReader
from textblob import TextBlob
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

st.title("📊 Product Review Sentiment Analysis")

# -------------------------------
# Function: Extract text from PDF
# -------------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------------
# Function: Split reviews
# -------------------------------
def split_reviews(text):
    reviews = re.split(r'\n\s*\n', text)
    reviews = [r.strip() for r in reviews if r.strip() and len(r.strip()) > 20]
    return reviews

# -------------------------------
# Function: Sentiment Analysis
# -------------------------------
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF file with product reviews", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    reviews = split_reviews(text)

    st.success(f"✅ Extracted {len(reviews)} reviews")

    # -------------------------------
    # Create DataFrame
    # -------------------------------
    data = []
    for i, review in enumerate(reviews):
        label, polarity = get_sentiment(review)

        data.append({
            "ID": i + 1,
            "Preview": review[:200],
            "Review": review,
            "Sentiment": label,
            "Polarity": round(polarity, 3)
        })

    df = pd.DataFrame(data)

    # -------------------------------
    # Summary Metrics
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    total = len(df)
    pos = len(df[df["Sentiment"] == "Positive"])
    neg = len(df[df["Sentiment"] == "Negative"])
    neu = len(df[df["Sentiment"] == "Neutral"])

    col1.metric("Positive", f"{pos}", f"{(pos/total)*100:.1f}%")
    col2.metric("Negative", f"{neg}", f"{(neg/total)*100:.1f}%")
    col3.metric("Neutral", f"{neu}", f"{(neu/total)*100:.1f}%")

    # -------------------------------
    # Charts Row 1
    # -------------------------------
    col1, col2 = st.columns(2)

    color_map = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "blue"
    }

    with col1:
        pie = px.pie(df, names="Sentiment", title="Sentiment Distribution",
                     color="Sentiment", color_discrete_map=color_map)
        st.plotly_chart(pie, use_container_width=True)

    with col2:
        bar = px.bar(df["Sentiment"].value_counts().reset_index(),
                     x="Sentiment", y="count",
                     title="Sentiment Count",
                     color="Sentiment", color_discrete_map=color_map)
        st.plotly_chart(bar, use_container_width=True)

    # -------------------------------
    # Charts Row 2
    # -------------------------------
    hist = px.histogram(df, x="Polarity", nbins=30, title="Polarity Distribution")
    st.plotly_chart(hist, use_container_width=True)

    # -------------------------------
    # Filter Table
    # -------------------------------
    st.subheader("🔍 Filter Reviews")

    selected = st.multiselect(
        "Select Sentiment",
        options=df["Sentiment"].unique(),
        default=df["Sentiment"].unique()
    )

    filtered_df = df[df["Sentiment"].isin(selected)]

    if not filtered_df.empty:
        st.dataframe(filtered_df[["ID", "Preview", "Sentiment", "Polarity"]])
    else:
        st.warning("No data for selected filter")

    # -------------------------------
    # Review Inspector
    # -------------------------------
    st.subheader("📄 Review Details")

    selected_id = st.selectbox("Select Review ID", df["ID"])

    review_row = df[df["ID"] == selected_id].iloc[0]

    st.info(f"""
    **Sentiment:** {review_row['Sentiment']}  
    **Polarity:** {review_row['Polarity']}

    **Full Review:**  
    {review_row['Review']}
    """)

    # -------------------------------
    # CSV Download
    # -------------------------------
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="sentiment_results.csv",
        mime="text/csv"
    )