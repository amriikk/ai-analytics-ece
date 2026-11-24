import pandas as pd
import pytest

from libs.shared_operators.operators import (
    OPERATOR_DISPATCH, derive_columns, distinct_rows, filter_rows,
    group_and_aggregate, limit_rows, select_columns, sort_rows,
)


@pytest.fixture
def songs():
    return pd.DataFrame({
        "Track": ["B Song", "A Song", "C Song", "A Song"],
        "Artist": ["Ana", "Ben", "Ana", "Ben"],
        "Streams": [300.0, 100.0, 200.0, 100.0],
        "Views": [30.0, 50.0, 0.0, 50.0],
        "Release Date": ["1/5/2023", "3/1/2024", "7/9/2024", "3/1/2024"],
    })


def test_all_seven_operators_are_registered():
    assert set(OPERATOR_DISPATCH) == {
        "select_columns", "filter_rows", "derive_columns", "group_and_aggregate",
        "sort_rows", "limit_rows", "distinct_rows",
    }


def test_filter_rows_combines_conditions(songs):
    out = filter_rows(songs, [
        {"column": "Artist", "operator": "==", "value": "Ana"},
        {"column": "Streams", "operator": ">", "value": 250},
    ])
    assert out["Track"].tolist() == ["B Song"]


def test_filter_rows_contains_ignores_case(songs):
    assert len(filter_rows(songs, [{"column": "Track", "operator": "contains", "value": "a song"}])) == 2


def test_sort_by_text_keeps_the_text(songs):
    out = sort_rows(songs, [{"column": "Track", "direction": "asc"}])
    assert out["Track"].tolist() == ["A Song", "A Song", "B Song", "C Song"]


def test_sort_by_number_descending(songs):
    out = sort_rows(songs, [{"column": "Streams", "direction": "desc"}])
    assert out["Streams"].tolist() == [300.0, 200.0, 100.0, 100.0]


def test_group_and_aggregate(songs):
    out = group_and_aggregate(songs, ["Artist"], [
        {"column": "Streams", "function": "sum", "as": "total"},
        {"column": "Track", "function": "count", "as": "tracks"},
    ])
    assert out.set_index("Artist")["total"].to_dict() == {"Ana": 500.0, "Ben": 200.0}
    assert out.set_index("Artist")["tracks"].to_dict() == {"Ana": 2, "Ben": 2}


def test_operators_do_not_change_their_input(songs):
    before = songs.copy()
    group_and_aggregate(songs, ["Artist"], [{"column": "Track", "function": "count", "as": "n"}])
    sort_rows(songs, [{"column": "Track", "direction": "asc"}])
    pd.testing.assert_frame_equal(songs, before)


def test_derive_arithmetic_and_year(songs):
    out = derive_columns(songs, [
        {"new_column": "per_view", "type": "arithmetic", "operation": "divide",
         "left": {"type": "column", "value": "Streams"}, "right": {"type": "column", "value": "Views"}},
        {"new_column": "year", "type": "extract_date_part", "column": "Release Date", "part": "year"},
    ])
    assert out["per_view"].iloc[0] == 10.0
    assert out["year"].tolist() == [2023, 2024, 2024, 2024]


def test_limit_select_distinct(songs):
    assert len(limit_rows(songs, 2)) == 2
    assert select_columns(songs, ["Track", "Not a column"]).columns.tolist() == ["Track"]
    assert len(distinct_rows(songs, ["Track", "Artist"])) == 3