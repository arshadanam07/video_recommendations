import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
nltk.download('vader_lexicon')
load_dotenv()

# Set up your Google API key in the environment variables
# You need to enable the YouTube Data API v3 in your Google Cloud Console
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Function to scrape comments from a video
def get_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText"
    )
    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments

# Function to get the top 10 most recent videos of a channel
def get_channel_videos(channel_id, max_results=10):
    videos = []
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    for item in response["items"]:
        videos.append(item)
    return videos

# Function to perform topic modeling and suggest topics
def analyze_comments(df):
    df = df.dropna(subset=['Comment'])
    comments = df['Comment'].tolist()

    # Sentiment analysis
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['Comment'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Topic modeling using Latent Dirichlet Allocation (LDA)
    vectorizer = CountVectorizer(stop_words='english')
    comment_matrix = vectorizer.fit_transform(comments)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    topic_matrix = lda.fit_transform(comment_matrix)

    # Assign the dominant topic to each comment
    df['dominant_topic'] = topic_matrix.argmax(axis=1)

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Print the top terms for each topic
    st.write("Top terms for each topic:")
    for topic_idx, topic in enumerate(lda.components_):
        top_terms_idx = topic.argsort()[:-10 - 1:-1]
        top_terms = [feature_names[i] for i in top_terms_idx]
        st.write(f"Topic {topic_idx + 1}: {', '.join(top_terms)}")

    # Analyze sentiment scores and dominant topics to provide recommendations
    positive_comments = df[df['sentiment_score'] > 0.2]
    popular_topics = df['dominant_topic'].value_counts().index[:3]
    # Generate recommendations based on positive sentiment and popular topics
    recommended_topics = df[(df['dominant_topic'].isin(popular_topics)) & (df['sentiment_score'] > 0.2)]
    # Print recommendations for the content creator
    st.write("\nRecommendations for the content creator:")
    st.write("1. Consider creating content on popular topics:")
    for topic in popular_topics:
        top_terms_idx = lda.components_[topic].argsort()[:-10 - 1:-1]
        top_terms = [feature_names[i] for i in top_terms_idx]
        st.write(f"   - Topic {topic}: {', '.join(top_terms)}")

    st.write("\n2. Additionally, focus on creating content with positive sentiment.")

# Streamlit app
st.title("YouTube Video Comments Analysis")

# Input for YouTube Channel ID
channel_id = st.text_input("Enter YouTube Channel ID:")

if st.button("Get Comments of Top 10 Most Recent Videos"):
    if channel_id:
        videos = get_channel_videos(channel_id)
        if videos:
            for video in videos:
                video_id = video['id']['videoId']
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                comments = get_video_comments(video_id)
                df = pd.DataFrame({'Comment': comments})
                st.write(f"## Comments for Video: {video['snippet']['title']}")
                if comments:
                    analyze_comments(df)
                else:
                    st.write("No comments found for this video.")
                st.markdown("---")
        else:
            st.write("No videos found for this channel.")
    else:
        st.write("Please enter a valid YouTube channel ID.")
