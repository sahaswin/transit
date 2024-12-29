import tweepy
import snscrape.modules.twitter as sntwitter
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime
import re
import os

# ---- Twitter API Setup ---- #
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ---- MongoDB Setup ---- #
mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
db = mongo_client['transit_alerts']
tweet_collection = db['tweets']

# ---- Pre-trained NLP Model ---- #
classifier = pipeline("text-classification", model="distilbert-base-uncased")

# ---- Fetch Tweets ---- #
def fetch_tweets():
    query = '(from:ttcnotices OR from:ttchelps) -is:retweet'
    tweets = client.search_recent_tweets(query=query, max_results=100)
    return tweets.data if tweets else []

# ---- Alternative Tweet Scraping ---- #
def scrape_tweets():
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper('from:ttcnotices').get_items():
        tweets.append(tweet)
        if len(tweets) >= 100:
            break
    return tweets

# ---- Preprocess Tweets ---- #
def preprocess_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower().strip()

# ---- Analyze Tweets ---- #
def analyze_tweet(tweet):
    processed_text = preprocess_tweet(tweet.text)
    result = classifier(processed_text)
    category = result[0]['label']
    save_to_db(tweet, category, processed_text)

# ---- Save to MongoDB ---- #
def save_to_db(tweet, category, processed_text):
    if not tweet_collection.find_one({'id': tweet.id}):
        tweet_data = {
            'id': tweet.id,
            'text': tweet.text,
            'processed_text': processed_text,
            'category': category,
            'timestamp': datetime.now()
        }
        tweet_collection.insert_one(tweet_data)
        print(f"Saved tweet {tweet.id}")

# ---- Main Pipeline ---- #
def main_pipeline():
    tweets = fetch_tweets() or scrape_tweets()
    for tweet in tweets:
        analyze_tweet(tweet)

if __name__ == "__main__":
    main_pipeline()
