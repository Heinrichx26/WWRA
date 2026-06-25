#!/usr/bin/env python
"""Run core reliability-audit tables from an existing panel."""

from __future__ import annotations

import argparse

import pandas as pd

from wwra.config import ensure_paths
from wwra.metrics import (
    climate_failure_forms,
    excess_burden,
    pressure_state_ladder,
    sample_closure,
    threshold_robustness,
    variable_definitions,
    weather_bin_workload_gap,
    workload_weather_surface,
)
from wwra.utils import parse_months, resolve_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--label", default="full_2024_2025")
    parser.add_argument("--months", default="2024-01:2025-12")
    parser.add_argument("--high-workload-quantile", type=float, default=0.75)
    return parser.parse_args()


def write_table(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = resolve_root(args.project_root)
    paths = ensure_paths(root)
    months = parse_months(args.months)
    panel_path = paths["processed"] / f"airport_route_reliability_panel_{args.label}.csv.gz"
    panel = pd.read_csv(panel_path, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date

    topk, origin_burden, route_burden = excess_burden(panel)
    outputs = {
        "sample_closure": sample_closure(panel, months, args.label),
        "variable_definitions": variable_definitions(),
        "pressure_state_ladder": pressure_state_ladder(panel),
        "weather_bin_workload_gap": weather_bin_workload_gap(panel),
        "workload_weather_surface": workload_weather_surface(panel),
        "topk_burden_share": topk,
        "origin_burden_contribution": origin_burden,
        "route_burden_contribution": route_burden,
        "climate_failure_forms": climate_failure_forms(panel),
        "threshold_robustness": threshold_robustness(panel, [0.70, args.high_workload_quantile, 0.80]),
    }
    for name, df in outputs.items():
        write_table(df, paths["tables"] / f"{name}_{args.label}.csv")
    print(outputs["sample_closure"].to_string(index=False))


if __name__ == "__main__":
    main()
