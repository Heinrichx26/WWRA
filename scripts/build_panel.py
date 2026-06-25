#!/usr/bin/env python
"""Build the public-data airport-route-day panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from wwra.config import ensure_paths
from wwra.panel import build_panel, parse_airport_list
from wwra.utils import parse_months, resolve_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--months", required=True, help="Comma list or range such as 2024-01:2025-12.")
    parser.add_argument("--label", default="full_2024_2025")
    parser.add_argument("--origins", default=None)
    parser.add_argument("--destinations", default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--high-workload-quantile", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_root(args.project_root)
    paths = ensure_paths(root)
    months = parse_months(args.months)
    origins = parse_airport_list(args.origins)
    destinations = parse_airport_list(args.destinations)
    panel = build_panel(
        root=root,
        months=months,
        origins=origins,
        destinations=destinations,
        chunksize=args.chunksize,
        high_workload_quantile=args.high_workload_quantile,
    )
    out = paths["processed"] / f"airport_route_reliability_panel_{args.label}.csv.gz"
    panel.to_csv(out, index=False)
    print(f"Wrote {len(panel):,} route-days to {Path(out).as_posix()}")


if __name__ == "__main__":
    main()
