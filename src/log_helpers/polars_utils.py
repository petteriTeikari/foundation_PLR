from loguru import logger
import polars as pl


def cast_numeric_polars_cols(df: pl.DataFrame, cast_to: str = "Float64"):
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
