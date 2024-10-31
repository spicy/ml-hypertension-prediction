import pandas as pd


def convert_numeric_to_int64(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to Int64 dtype where possible."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = df[col].astype("Int64")
            except (ValueError, TypeError):
                pass
    return df
