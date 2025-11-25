import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import tweepy
import pandas as pd
import time
from datetime import datetime

def setup_twitter_api():
    """Thiết lập kết nối Twitter API"""
    print("Setting up Twitter API...")
    
    # Thay thế bằng API keys của bạn
    API_KEYS = {
        'consumer_key': 'YOUR_CONSUMER_KEY',
        'consumer_secret': 'YOUR_CONSUMER_SECRET',
        'access_token': 'YOUR_ACCESS_TOKEN',
        'access_token_secret': 'YOUR_ACCESS_TOKEN_SECRET'
    }
    
    try:
        auth = tweepy.OAuthHandler(API_KEYS['consumer_key'], API_KEYS['consumer_secret'])
        auth.set_access_token(API_KEYS['access_token'], API_KEYS['access_token_secret'])
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Verify credentials
        api.verify_credentials()
        print("✅ Twitter API connected successfully!")
        return api
    except Exception as e:
        print(f"❌ Twitter API connection failed: {e}")
        return None

def collect_tweets_by_hashtag(api, hashtag, max_tweets=1000):
    """Thu thập tweets theo hashtag"""
    print(f"Collecting tweets with hashtag: #{hashtag}")
    
    tweets_data = []
    
    try:
        for tweet in tweepy.Cursor(api.search_tweets,
                                 q=f"#{hashtag} -filter:retweets",
                                 lang="en",
                                 tweet_mode='extended').items(max_tweets):
            
            tweet_info = {
                'tweet_id': tweet.id,
                'user_id': tweet.user.id,
                'user_screen_name': tweet.user.screen_name,
                'text': tweet.full_text,
                'created_at': tweet.created_at,
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count,
                'hashtags': [hashtag['text'] for hashtag in tweet.entities['hashtags']],
                'user_mentions': [mention['screen_name'] for mention in tweet.entities['user_mentions']]
            }
            tweets_data.append(tweet_info)
            
        print(f"✅ Collected {len(tweets_data)} tweets for #{hashtag}")
        return tweets_data
        
    except Exception as e:
        print(f"❌ Error collecting tweets: {e}")
        return []

def build_retweet_network(tweets_data):
    """Xây dựng mạng retweet từ dữ liệu tweets"""
    print("Building retweet network...")
    
    import networkx as nx
    
    G = nx.DiGraph()
    
    for tweet in tweets_data:
        user = tweet['user_screen_name']
        G.add_node(user)
        
        # Thêm edges cho user mentions (retweet/mention)
        for mentioned_user in tweet['user_mentions']:
            if mentioned_user != user:  # Tránh self-loop
                G.add_edge(user, mentioned_user)
    
    print(f"✅ Retweet network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def main():
    """Main function for Twitter data collection"""
    config = Config()
    
    print("\n" + "="*60)
    print("TWITTER DATA COLLECTION")
    print("="*60)
    
    # Thiết lập API
    api = setup_twitter_api()
    
    if not api:
        print("❌ Cannot proceed without Twitter API connection")
        return None
    
    # Thu thập dữ liệu
    hashtags = ['python', 'datascience', 'machinelearning']
    all_tweets = []
    
    for hashtag in hashtags:
        tweets = collect_tweets_by_hashtag(api, hashtag, max_tweets=100)
        all_tweets.extend(tweets)
        time.sleep(1)  # Tránh rate limit
    
    # Xây dựng mạng
    if all_tweets:
        G = build_retweet_network(all_tweets)
        
        # Lưu dữ liệu
        tweets_df = pd.DataFrame(all_tweets)
        tweets_df.to_csv(os.path.join(config.DATA_DIR, 'collected_tweets.csv'), index=False)
        
        # Lưu mạng
        nx.write_edgelist(G, os.path.join(config.DATA_DIR, 'twitter_network.edgelist'))
        
        print(f"✅ Data saved: {len(all_tweets)} tweets, {G.number_of_nodes()} users")
        return G
    else:
        print("❌ No data collected")
        return None

if __name__ == "__main__":
    main()