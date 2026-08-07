from io import BytesIO
import polars as pl

def _df_to_excel_stream(df: pl.DataFrame) -> BytesIO:
    byte_stream = BytesIO()

