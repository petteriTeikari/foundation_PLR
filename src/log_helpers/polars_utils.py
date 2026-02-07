import polars as pl
from loguru import logger


def cast_numeric_polars_cols(df: pl.DataFrame, cast_to: str = "Float64"):
    """Cast all numeric columns in Polars DataFrame to specified type.

    Useful for avoiding schema errors when combining DataFrames with
    different numeric precision.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    cast_to : str, default "Float64"
        Target numeric type.

    Returns
    -------
    pl.DataFrame
        DataFrame with numeric columns cast to specified type.

    Raises
    ------
    NotImplementedError
        If cast_to is not "Float64".
    """
    # To avoid this:
    # polars.exceptions.SchemaError: type Float32 is incompatible with expected type Float64
    for col in df.columns:
        if df[col].dtype.is_numeric():
            if cast_to == "Float64":
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                except Exception as e:
                    logger.error(f"Error in casting the column {col} to Float64: {e}")
            else:
                logger.error(f"Unknown cast_to type: {cast_to}")
                raise NotImplementedError

    return df
