import pandas as pd

from libs.data_hygiene.cleaner import load_and_clean_spotify_data, looks_numeric
from tests.conftest import CSV


def test_comma_numbers_become_real_numbers():
    df = load_and_clean_spotify_data(CSV)
    assert pd.api.types.is_numeric_dtype(df["Spotify Streams"])
    # 113 rows are blank in the source file; every other row must have a value
    assert df["Spotify Streams"].notna().sum() == 4487


def test_cleaning_never_loses_a_value():
    raw = pd.read_csv(CSV, encoding="latin1")
    clean = load_and_clean_spotify_data(CSV)
    for col in raw.columns:
        assert clean[col].isna().sum() == raw[col].isna().sum(), col


def test_text_columns_stay_text():
    df = load_and_clean_spotify_data(CSV)
    for col in ["Track", "Artist", "ISRC", "Release Date"]:
        assert not pd.api.types.is_numeric_dtype(df[col]), col


def test_looks_numeric():
    assert looks_numeric(pd.Series(["684", "390,470,936", None]))
    assert not looks_numeric(pd.Series(["4/26/2024"]))
    assert not looks_numeric(pd.Series(["12", "QM24S2402528"]))