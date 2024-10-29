import pandas as pd
from finvizfinance.news import News
import csv
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

analyzer = SentimentIntensityAnalyzer()

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

fnews = News()
all_news = fnews.get_news()
fieldNames = ['Category', 'Date', 'Title', 'Source', 'Link', 'Sentiment']

total_sentiment = 0
total_weighted_sentiment = 0
total_weight = 0
count = 0

with open('fnews.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
    writer.writeheader()
    for category, df in all_news.items():
        for index, row in df.iterrows():
            link = row['Link']
            print(f"Fetching content from: {link}")
            content = fetch_content(link)
            content_length = len(content)
            print(f"Content length: {content_length} characters")
            
            if content_length == 0:
                print(f"Skipping link due to 0 content length: {link}\n")
                continue
            
            sentiment = analyzer.polarity_scores(content)['compound']
            sentiment = round(sentiment, 3)
            print(f"Sentiment score: {sentiment}")
            
            weight = calculate_weight(row['Date'])
            weighted_sentiment = sentiment * weight
            
            total_sentiment += sentiment
            total_weighted_sentiment += weighted_sentiment
            total_weight += weight
            count += 1
            
            writer.writerow({
                'Category': category,
                'Date': row['Date'],
                'Title': row['Title'],
                'Source': row['Source'],
                'Link': row['Link'],
                'Sentiment': sentiment
            })
            print(f"Finished processing link: {link}\n")

if count > 0:
    average_sentiment = round(total_sentiment / count, 3)
    weighted_average_sentiment = round(total_weighted_sentiment / total_weight, 3)
    print(f"Regular Average Sentiment Score: {average_sentiment}")
    print(f"Weighted Average Sentiment Score: {weighted_average_sentiment}")
else:
    print("No content to analyze.")