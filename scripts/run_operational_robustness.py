#!/usr/bin/env python
"""Run operational-context robustness tables from an existing panel."""

from __future__ import annotations

import argparse

import pandas as pd

from wwra.config import ensure_paths
from wwra.robustness import (
    add_context_tiers,
    airport_fixed_gaps,
    context_stratified_gaps,
    sequential_operating_context_adjustment,
    top20_burden_by_year,
    year_portability,
)
from wwra.utils import resolve_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--label", default="full_2024_2025")
    parser.add_argument("--min-arrivals", type=int, default=500)
    parser.add_argument("--min-route-days", type=int, default=20)
    return parser.parse_args()


def write_table(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = resolve_root(args.project_root)
    paths = ensure_paths(root)
    panel_path = paths["processed"] / f"airport_route_reliability_panel_{args.label}.csv.gz"
    panel = pd.read_csv(panel_path, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = add_context_tiers(panel)

    airport_cells, airport_summary = airport_fixed_gaps(panel, args.min_arrivals, args.min_route_days)
    context_cells, context_summary = context_stratified_gaps(panel, args.min_arrivals, args.min_route_days)
    outputs = {
        "airport_fixed_weather_gap_cells": airport_cells,
        "airport_fixed_weather_gap_summary": airport_summary,
        "operational_context_stratified_gap_cells": context_cells,
        "operational_context_stratified_gap_summary": context_summary,
        "sequential_operating_context_adjustment": sequential_operating_context_adjustment(panel),
        "yearly_weather_gap_portability": year_portability(panel, args.min_arrivals, args.min_route_days),
        "yearly_top20_burden_share": top20_burden_by_year(panel),
    }
    for name, df in outputs.items():
        write_table(df, paths["tables"] / f"{name}_{args.label}.csv")
    print(outputs["operational_context_stratified_gap_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
