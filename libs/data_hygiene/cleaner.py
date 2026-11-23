import pandas as pd

# A cell counts as a number if it looks like "684", "390,470,936" or "1,234.5".
NUMBER_PATTERN = r"-?\d{1,3}(,\d{3})+(\.\d+)?|-?\d+(\.\d+)?"


def looks_numeric(column: pd.Series) -> bool:
    """True if every non-empty cell in a text column is a number (commas allowed)."""
    values = column.dropna().astype(str).str.strip()
    return len(values) > 0 and bool(values.str.fullmatch(NUMBER_PATTERN).all())


def load_and_clean_spotify_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Spotify dataset and turns comma-formatted text like "390,470,936"
    into real numbers, so the agents sort and aggregate actual values.

    Every text column is checked, so columns are never missed because they
    weren't on a hand-written list. Text columns such as Track and Artist
    stay as text.
    """
    try:
        df = pd.read_csv(file_path, encoding="latin1")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure it is in the data/ directory.")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # already numbers
        if looks_numeric(df[col]):
            no_commas = df[col].str.replace(",", "", regex=False).str.strip()
            df[col] = pd.to_numeric(no_commas, errors="coerce")

    return df


_df_cache = {}


def get_cached_dataframe(file_path: str) -> pd.DataFrame:
    """Fetches the dataframe from memory, or loads it if not present."""
    if file_path not in _df_cache:
        print(f"Cache miss: Loading and cleaning {file_path} into memory...")
        _df_cache[file_path] = load_and_clean_spotify_data(file_path)
    return _df_cache[file_path]