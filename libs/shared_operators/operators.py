import pandas as pd
from typing import List, Dict, Any

def derive_columns(df: pd.DataFrame, derive: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates new columns via arithmetic or date extraction."""
    df_out = df.copy()
    for op in derive:
        new_col = op["new_column"]
        if op["type"] == "arithmetic":
            left = df_out[op["left"]["value"]] if op["left"]["type"] == "column" else op["left"]["value"]
            right = df_out[op["right"]["value"]] if op["right"]["type"] == "column" else op["right"]["value"]
            
            # Ensure numeric types for arithmetic
            left = pd.to_numeric(left, errors='coerce') if isinstance(left, pd.Series) else left
            right = pd.to_numeric(right, errors='coerce') if isinstance(right, pd.Series) else right
            
            if op["operation"] == "divide":
                df_out[new_col] = left / right
            elif op["operation"] == "add":
                df_out[new_col] = left + right
            elif op["operation"] == "subtract":
                df_out[new_col] = left - right
            elif op["operation"] == "multiply":
                df_out[new_col] = left * right
                
        elif op["type"] == "extract_date_part":
            df_out[op["column"]] = pd.to_datetime(df_out[op["column"]], errors='coerce')
            if op["part"] == "year":
                df_out[new_col] = df_out[op["column"]].dt.year
    return df_out

def filter_rows(df: pd.DataFrame, conditions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Filters dataframe based on a list of AND conditions."""
    df_out = df.copy()
    for cond in conditions:
        col = cond["column"]
        val = cond["value"]
        op = cond["operator"]
        
        if op == "==": df_out = df_out[df_out[col] == val]
        elif op == "!=": df_out = df_out[df_out[col] != val]
        elif op == ">": df_out = df_out[pd.to_numeric(df_out[col], errors='coerce') > val]
        elif op == "<": df_out = df_out[pd.to_numeric(df_out[col], errors='coerce') < val]
        elif op == ">=": df_out = df_out[pd.to_numeric(df_out[col], errors='coerce') >= val]
        elif op == "<=": df_out = df_out[pd.to_numeric(df_out[col], errors='coerce') <= val]
        elif op == "contains": df_out = df_out[df_out[col].astype(str).str.contains(str(val), case=False, na=False)]
    return df_out

def group_and_aggregate(df: pd.DataFrame, group_by: List[str], metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Groups by columns and applies aggregation functions."""
    agg_dict = {}
    rename_dict = {}
    
    for m in metrics:
        col = m["column"]
        func = m["function"]
        out_name = m["as"]
        
        # HW3 Bug Fix: Do not coerce to numeric if simply counting strings
        if func != "count":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        if col not in agg_dict:
            agg_dict[col] = []
        agg_dict[col].append(func)
        rename_dict[f"{col}_{func}"] = out_name

    df_grouped = df.groupby(group_by).agg(agg_dict)
    df_grouped.columns = [f"{col}_{func}" for col, funcs in agg_dict.items() for func in funcs]
    df_grouped = df_grouped.rename(columns=rename_dict).reset_index()
    return df_grouped

def sort_rows(df: pd.DataFrame, sort_by: List[Dict[str, str]]) -> pd.DataFrame:
    """Sorts dataframe by specified columns."""
    cols = [s["column"] for s in sort_by]
    asc = [s["direction"] == "asc" for s in sort_by]
    
    # Coerce to numeric for accurate sorting
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df.sort_values(by=cols, ascending=asc, na_position='last')

def limit_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Returns the first k rows."""
    return df.head(k)

def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Projects a subset of columns."""
    existing_cols = [c for c in columns if c in df.columns]
    return df[existing_cols]

def distinct_rows(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Drops duplicate rows based on subset of columns."""
    return df.drop_duplicates(subset=columns)

# Dispatch dictionary for the Executor Node
OPERATOR_DISPATCH = {
    "derive_columns": derive_columns,
    "filter_rows": filter_rows,
    "group_and_aggregate": group_and_aggregate,
    "sort_rows": sort_rows,
    "limit_rows": limit_rows,
    "select_columns": select_columns,
    "distinct_rows": distinct_rows
}