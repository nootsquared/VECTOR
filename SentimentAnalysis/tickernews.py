import pandas as pd
from finvizfinance.quote import finvizfinance
import csv
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

analyzer = SentimentIntensityAnalyzer()

# Expanded lists of positive and negative words with more impactful adjustments
positive_words = [
    'gain', 'profit', 'increase', 'growth', 'surge', 'rise', 'improvement', 
    'record', 'high', 'success', 'win', 'benefit', 'advantage', 'boost', 'positive'
]
negative_words = [
    'loss', 'decline', 'drop', 'fall', 'decrease', 'strike', 'plummet', 
    'low', 'negative', 'fail', 'failure', 'risk', 'threat', 'downturn', 'crash'
]

def fetch_content(link):
    try:
        response = requests.get(link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            return content
        else:
            return ""
    except Exception as e:
        print(f"Error fetching content from {link}: {e}")
        return ""

def calculate_weight(date_str):
    try:
        now = datetime.now()
        if ':' in date_str:
            date = datetime.strptime(date_str, '%I:%M%p').replace(year=now.year, month=now.month, day=now.day)
        else:
            date = datetime.strptime(date_str, '%b-%d').replace(year=now.year)
            if date > now:
                date = date.replace(year=now.year - 1)
        delta = now - date
        weight = 1 / (delta.total_seconds() + 1)
        return weight
    except Exception as e:
        print(f"Error calculating weight for date {date_str}: {e}")
        return 1

def adjust_sentiment(content, sentiment):
    words = content.split()
    for word in words:
        if word.lower() in positive_words:
            sentiment += .7  # Increased impact for positive words
        elif word.lower() in negative_words:
            sentiment -= .7  # Increased impact for negative words
    # Cap the sentiment score between -5 and 5
    sentiment = max(min(sentiment, 5), -5)
    return round(sentiment, 3)

ticker = input("Enter the stock ticker symbol: ")

stock = finvizfinance(ticker)
news_df = stock.ticker_news()

fieldNames = ['Category', 'Date', 'Title', 'Source', 'Link', 'Sentiment']

total_sentiment = 0
total_weighted_sentiment = 0
total_weight = 0
count = 0

with open(f'{ticker}_news.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
    writer.writeheader()
    for index, row in news_df.iterrows():
        link = row['Link']
        content = fetch_content(link)
        content_length = len(content)
        
        if content_length == 0:
            continue
        
        print(f"Fetching content from: {link}")
        print(f"Content length: {content_length} characters")
        
        sentiment = analyzer.polarity_scores(content)['compound']
        sentiment = adjust_sentiment(content, sentiment)
        print(f"Sentiment score: {sentiment}")
        
        weight = calculate_weight(row['Date'])
        weighted_sentiment = sentiment * weight
        
        total_sentiment += sentiment
        total_weighted_sentiment += weighted_sentiment
        total_weight += weight
        count += 1
        
        writer.writerow({
            'Category': 'news',
            'Date': row['Date'],
            'Title': row['Title'],
            'Source': row['Source'],
            'Link': row['Link'],
            'Sentiment': sentiment
        })
        print(f"Finished processing link: {link}\n")

if count > 0:
    average_sentiment = round(total_sentiment / count, 3) + 5
    weighted_average_sentiment = round(total_weighted_sentiment / total_weight, 3) + 5
    print(f"Regular Average Sentiment Score: {average_sentiment} / 10")
    print(f"Weighted Average Sentiment Score: {weighted_average_sentiment} / 10")
else:
    print("No content to analyze.")