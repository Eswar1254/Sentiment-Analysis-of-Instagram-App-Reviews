Instagram Review Sentiment Analysis

This project performs sentiment analysis on Instagram reviews using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. It categorizes reviews into Positive, Negative, and Neutral sentiments and provides a visual representation of the sentiment distribution.

Features:-
Loads reviews from a CSV file
Analyzes sentiment using VADER
Categorizes reviews as Positive, Negative, or Neutral
Displays the number of reviews in each category
Saves the results to a new CSV file
Visualizes the sentiment distribution using seaborn and matplotlib

Setup:-

Prerequisites
Python 3.x
pandas
nltk
matplotlib
seaborn

Installation:-

1)Clone the repository:
git clone https://github.com/yourusername/instagram-review-sentiment-analysis.git
cd instagram-review-sentiment-analysis

2)Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3)Install the required packages:

pip install pandas nltk matplotlib seaborn

4)Download the VADER lexicon:

import nltk
nltk.download('vader_lexicon')

5)Usage:-

Place your CSV file (with a column named review_description containing the reviews) in the project directory.

6)Update the file_path variable in the script to point to your CSV file:

file_path = 'path/to/your/instagram.csv'  # Replace with the actual path to your CSV file

6)Run the script:

python sentiment_analysis.py
Script Details

7)Here's a detailed breakdown of the script:

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Load the CSV file
file_path = 'path/to/your/instagram.csv'  # Replace with the actual path to your CSV file
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

    # Optionally, save the results to a new CSV file
    df.to_csv('sentiment_analysis_results.csv', index=False)

    # Plot the results
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Define a color palette
    palette = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}

    sns.countplot(x='Sentiment', data=df, palette=palette)
    plt.title('Sentiment Analysis of Reviews')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')

    plt.show()
else:
    print("Column 'review_description' not found in the CSV file.")
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements:-

VADER Sentiment Analysis
nltk
pandas
matplotlib
seaborn
