import streamlit as st
import pandas as pd
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the pickled model
with open('sentiment_model.pkl', 'rb') as model_file:
    sid = pickle.load(model_file)

# Function to categorize the sentiment
def categorize_sentiment(text):
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit app
st.title("Sentiment Analysis of Reviews")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'review_description' in df.columns:
        # Apply the sentiment analysis
        df['Sentiment'] = df['review_description'].apply(categorize_sentiment)

        # Display the dataframe
        st.write(df)

        # Plot the results
        st.write("Sentiment Analysis Results")
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    else:
        st.write("Column 'review_description' not found in the CSV file.")
