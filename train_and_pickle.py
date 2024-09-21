import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Load the CSV file
file_path = 'instagram.csv'  
df = pd.read_csv(file_path)

# Ensure the column with reviews is correctly identified (assuming 'review_description')
if 'review_description' in df.columns:
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

    # Apply the sentiment analysis
    df['Sentiment'] = df['review_description'].apply(categorize_sentiment)

    # Separate the reviews into categories
    positive_reviews = df[df['Sentiment'] == 'Positive']
    negative_reviews = df[df['Sentiment'] == 'Negative']
    neutral_reviews = df[df['Sentiment'] == 'Neutral']

    # Display the results
    print(f"Positive reviews: {len(positive_reviews)}")
    print(f"Negative reviews: {len(negative_reviews)}")
    print(f"Neutral reviews: {len(neutral_reviews)}")

    # Pickle the SentimentIntensityAnalyzer
    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(sid, model_file)

else:
    print("Column 'review_description' not found in the CSV file.")
