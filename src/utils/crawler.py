import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Replace 'your_api_key' with your actual API keys
load_dotenv()
genius_api_key = os.getenv("GENIUS_API_KEY")
lastfm_api_key = os.getenv("LASTFM_API_KEY")

# Configure retries for the requests session
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.headers.update({'Authorization': f'Bearer {genius_api_key}'})

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def search_song(song_title, artist_name):
    base_url = 'https://api.genius.com/search'
    params = {'q': f'{song_title} {artist_name}'}
    try:
        response = session.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json()['response']['hits']
            for hit in results:
                hit_title = hit['result']['title'].lower()
                hit_artist = hit['result']['primary_artist']['name'].lower()
                if song_title.lower() in hit_title and artist_name.lower() in hit_artist:
                    return hit['result']['url']
    except requests.RequestException as e:
        logging.error(f"[DEBUG] Request failed: {e}")
    return None

def get_genius_about(song_url):
    try:
        response = session.get(song_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            about_section = soup.select_one("div[class*='RichText__Container']")
            if about_section:
                for a in about_section.find_all('a'):
                    a.decompose()
                paragraphs = about_section.find_all('p')
                about_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
                return about_text
    except requests.RequestException as e:
        logging.error(f"[DEBUG] Request failed: {e}")
    return None

def get_lastfm_about(artist, title):
    url = f'http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={lastfm_api_key}&artist={artist}&track={title}&format=json'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'track' in data and 'wiki' in data['track']:
                return data['track']['wiki']['content']
    except requests.RequestException as e:
        logging.error(f"[DEBUG] Request failed: {e}")
    return "About section not available"

def get_about_section(artist, title):
    song_url = search_song(title, artist)
    if song_url:
        about_section = get_genius_about(song_url)
        if about_section:
            return about_section
    about_section = get_lastfm_about(artist, title)
    return about_section

def preprocess_text(text):
    """Remove commas and newlines from text to keep it in one line"""
    if text:
        text = text.replace(',', ' ').replace('\n', ' ').replace('\r', ' ')
    return text

def classify_topic(text):
    """
    A simple keyword-based approach to classify the topic of a song
    """
    keywords = {
        'love': ['love', 'heart', 'romance', 'affection'],
        'friends': ['friend', 'friendship', 'buddy', 'pal'],
        'life': ['life', 'living', 'existence'],
        'news': ['news', 'headline', 'report', 'journalism'],
        'holiday': ['holiday', 'vacation', 'festival'],
        'friendship': ['friendship', 'companionship', 'bond'],
        'war': ['war', 'battle', 'conflict'],
        'peace': ['peace', 'harmony', 'calm'],
        'character': ['character', 'personality', 'traits']
        # Add more topics and keywords as needed
    }

    for topic, words in keywords.items():
        for word in words:
            if word in text.lower():
                return topic
    return 'other'

def process_song(row):
    try:
        song_title = row['title']
        artist_name = row['artist']
        if not isinstance(song_title, str) or not song_title.strip():
            raise ValueError("Invalid song title")
        about_section = get_about_section(artist_name, song_title)
        row['about_section'] = preprocess_text(about_section) if about_section else None
        row['topic'] = classify_topic(row['lyrics'] + ' ' + (row['about_section'] or ''))
        logging.debug(f"[DEBUG] Processed: {song_title} by {artist_name}")
        return row
    except Exception as e:
        logging.error(f"[DEBUG] Error processing row {row['id']}: {e}")
    return None

# Read the original dataset
original_df = pd.read_csv('src/data/song_lyrics.csv', usecols=['title', 'tag', 'artist', 'lyrics', 'id'], nrows=500000)

# Sample 150,000 rows for processing
sample_df = original_df.sample(n=150000, random_state=1)

# Preprocess lyrics to remove commas and newlines
sample_df['lyrics'] = sample_df['lyrics'].apply(preprocess_text)

# Ensure 'title' and 'about_section' columns are properly initialized
sample_df['title'] = sample_df['title'].fillna('').astype(str)
sample_df['about_section'] = None

# Find matches and save
matches = []

with ThreadPoolExecutor(max_workers=20) as executor:
    # Use tqdm to add a progress bar to the map function
    results = list(tqdm(executor.map(process_song, [row for _, row in sample_df.iterrows()]), total=len(sample_df), desc="[PROGRESS] Processing songs"))

# Filter out None results
matches = [result for result in results if result is not None and result['about_section'] != 'About section not available']

# Create a DataFrame from the matches
matches_df = pd.DataFrame(matches)

# Ensure 'about_section' is a column in the DataFrame
if 'about_section' not in matches_df.columns:
    matches_df['about_section'] = None

# Check for rows without 'about_section' and log them
missing_about = matches_df[matches_df['about_section'].isnull()]
if not missing_about.empty:
    logging.error(f"[DEBUG] Rows missing about section: {missing_about}")

# Save the matches to a new CSV file
matches_df.to_csv('matched_songs_with_about.csv', index=False)

logging.info("[DEBUG] Dataset populated successfully with 'about_section' information.")
