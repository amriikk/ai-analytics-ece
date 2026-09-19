import pandas as pd
import numpy as np

def load_and_clean_spotify_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Spotify dataset, strips commas from numeric columns, 
    and handles type coercion to ensure high data hygiene before LLM processing.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure it is in the data/ directory.")
    
    # Columns known to have comma-formatted strings in the raw dataset
    numeric_columns = [
        'Spotify Streams', 'Spotify Playlist Count', 'Spotify Popularity', 
        'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 
        'TikTok Views', 'Shazam Counts'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Only apply string methods if the column is actually of object/string type
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            # Coerce to numeric, turning unparseable values into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# Add this below your existing load_and_clean_spotify_data function

_df_cache = {}

def get_cached_dataframe(file_path: str) -> pd.DataFrame:
    """Fetches the dataframe from memory, or loads it if not present."""
    if file_path not in _df_cache:
        print(f"Cache miss: Loading and cleaning {file_path} into memory...")
        _df_cache[file_path] = load_and_clean_spotify_data(file_path)
    return _df_cache[file_path]