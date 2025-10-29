"""
Raw file I/O only.
- Expect preprocessed CSVs to be present in data_dir.
- Encoding defaults to EUC-KR.
"""
from __future__ import annotations
import pandas as pd


def load_tables(data_dir: str, encoding: str = "euc-kr"):
    water = pd.read_csv(f"{data_dir}/new_water.csv", encoding=encoding)
    small_edge = pd.read_csv(f"{data_dir}/small_edge.csv", encoding=encoding)
    middle_edge = pd.read_csv(f"{data_dir}/middle_edge.csv", encoding=encoding)
    bio = pd.read_csv(f"{data_dir}/bio_final.csv", encoding=encoding)
    return water, small_edge, middle_edge, bio
