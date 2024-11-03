import os
from finvizfinance.quote import finvizfinance
import csv
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import pipeline

pipe = pipeline("text-classification", model="yiyanghkust/finbert-tone")

todayWeight = 2

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

def open_for_analysis(article_text):
    max_length = 512
    chunks = [article_text[i:i + max_length] for i in range(0, len(article_text), max_length)]
    
    sentiment_scores = []
    for chunk in chunks:
        result = pipe(chunk)
        print(f"Model output: {result}")
        
        sentiment_score = 0
        if result[0]['label'].lower() == 'positive':
            sentiment_score = result[0]['score']
        elif result[0]['label'].lower() == 'negative':
            sentiment_score = -result[0]['score']
        
        if sentiment_score != 0:
            sentiment_scores.append(sentiment_score * 50 + 50)
    
    if sentiment_scores:
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
        return avg_sentiment_score
    else:
        return 0

def get_stock_sentiment(ticker):
    stock = finvizfinance(ticker)
    stock_fundament = stock.ticker_fundament()
    company_name = stock_fundament.get('Company', None)

    if not company_name:
        company_name = input("Enter the full name of the company: ")

    stock_description = stock.ticker_description()

    print(f"Company Name: {company_name}")
    print(f"Company Description: {stock_description}")

    news_df = stock.ticker_news()

    fieldNames = ['Category', 'Date', 'Title', 'Source', 'Link', 'Content', 'Sentiment']

    total_sentiment_score = 0
    num_articles = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, f'{ticker}_news.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
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
            
            content_with_context = content
            
            sentiment_score = open_for_analysis(content_with_context)
            
            if sentiment_score != 0:
                if datetime.strptime(str(row['Date']), '%Y-%m-%d %H:%M:%S').date() == datetime.today().date():
                    print("This is a today's article.")
                    total_sentiment_score += sentiment_score * todayWeight
                    num_articles += todayWeight
                else:
                    total_sentiment_score += sentiment_score
                    num_articles += 1
            
                writer.writerow({
                    'Category': 'news',
                    'Date': row['Date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Title': row['Title'],
                    'Source': row['Source'],
                    'Link': row['Link'],
                    'Content': content_with_context,
                    'Sentiment': sentiment_score
                })
            
                print(f"Sentiment score for article: {sentiment_score}")
                print(f"Finished processing link: {link}\n")
                
    if num_articles > 0:
        avg_sentiment_score = total_sentiment_score / num_articles
    else:
        avg_sentiment_score = 0

    print(f"Average Sentiment Score: {avg_sentiment_score}")
    return int(avg_sentiment_score)

# Example usage:
# sentiment_score = get_stock_sentiment("AAPL")
# print(f"Sentiment Score: {sentiment_score}")