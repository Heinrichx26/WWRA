"""Shared utility functions."""

from __future__ import annotations

import zipfile
from pathlib import Path


def resolve_root(project_root: str | None = None) -> Path:
    if project_root:
        return Path(project_root).expanduser().resolve()
    return Path.cwd().resolve()


def month_range(start: str, end: str) -> list[str]:
    import pandas as pd

    return pd.period_range(start=start, end=end, freq="M").astype(str).to_list()


def parse_months(text: str) -> set[str]:
    text = text.strip()
    if ":" in text:
        start, end = [part.strip() for part in text.split(":", 1)]
        return set(month_range(start, end))
    return {item.strip() for item in text.split(",") if item.strip()}


def safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def first_csv_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                return name
    raise FileNotFoundError(f"No CSV member found in {zip_path.name}")
