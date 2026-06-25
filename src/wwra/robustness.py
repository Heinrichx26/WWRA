"""Operational-context robustness tables for the reliability audit."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import WEATHER_BIN_ORDER
from .metrics import ci_for_rate_diff, weighted_rate


def tertile_labels(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    labels = ["low", "middle", "high"]
    try:
        return pd.qcut(valid.rank(method="first"), q=3, labels=labels).astype(str)
    except ValueError:
        return pd.Series(np.where(valid >= valid.median(), "high", "low"), index=series.index)


def add_context_tiers(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    origin_day = out[["origin", "date", "origin_daily_flights"]].drop_duplicates()
    origin_size = origin_day.groupby("origin", as_index=False)["origin_daily_flights"].mean()
    origin_size["origin_size_tier"] = tertile_labels(origin_size["origin_daily_flights"])
    out = out.merge(origin_size[["origin", "origin_size_tier"]], on="origin", how="left")

    route_volume = out.groupby("route", as_index=False)["flights"].mean().rename(columns={"flights": "route_mean_flights"})
    route_volume["route_volume_tier"] = tertile_labels(route_volume["route_mean_flights"])
    out = out.merge(route_volume[["route", "route_volume_tier"]], on="route", how="left")

    out["route_carrier_count_tier"] = pd.cut(
        pd.to_numeric(out["carriers"], errors="coerce").fillna(0),
        bins=[-np.inf, 1, 2, np.inf],
        labels=["one carrier", "two carriers", "three or more carriers"],
    ).astype(str)
    out["route_peak_share_tier"] = tertile_labels(out["route_peak_share"])
    out["origin_peak_share_tier"] = tertile_labels(out["origin_peak_share"])
    out["origin_carrier_breadth_tier"] = tertile_labels(out["origin_carriers"])
    out["origin_destination_breadth_tier"] = tertile_labels(out["origin_destinations"])
    return out


def gap_for_part(part: pd.DataFrame, min_arrivals: int, min_route_days: int) -> dict[str, float] | None:
    normal = part.loc[part["airport_workload_pressure"] == 0]
    high = part.loc[part["airport_workload_pressure"] == 1]
    if len(normal) < min_route_days or len(high) < min_route_days:
        return None
    if normal["arr15_obs"].sum() < min_arrivals or high["arr15_obs"].sum() < min_arrivals:
        return None
    normal_rate = weighted_rate(normal["arr15_delayed"], normal["arr15_obs"])
    high_rate = weighted_rate(high["arr15_delayed"], high["arr15_obs"])
    se, ci_low, ci_high = ci_for_rate_diff(high_rate, high["arr15_obs"].sum(), normal_rate, normal["arr15_obs"].sum())
    return {
        "normal_route_days": int(len(normal)),
        "high_workload_route_days": int(len(high)),
        "normal_arrivals": int(normal["arr15_obs"].sum()),
        "high_workload_arrivals": int(high["arr15_obs"].sum()),
        "normal_arr15_rate": normal_rate,
        "high_workload_arr15_rate": high_rate,
        "raw_gap_high_minus_normal": high_rate - normal_rate,
        "raw_gap_se": se,
        "raw_gap_ci_low": ci_low,
        "raw_gap_ci_high": ci_high,
    }


def airport_fixed_gaps(panel: pd.DataFrame, min_arrivals: int, min_route_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    weather_panel = panel.loc[panel["route_weather_pressure"] == 1].copy()
    for (origin, weather_bin), part in weather_panel.groupby(["origin", "weather_intensity_bin"], sort=False):
        gap = gap_for_part(part, min_arrivals, min_route_days)
        if gap is not None:
            rows.append({"origin": origin, "weather_intensity_bin": weather_bin, **gap})
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells, cells
    summary_rows = []
    for weather_bin, part in cells.groupby("weather_intensity_bin", sort=False):
        weights = part["normal_arrivals"] + part["high_workload_arrivals"]
        summary_rows.append(
            {
                "weather_intensity_bin": weather_bin,
                "eligible_origin_weather_cells": int(len(part)),
                "positive_gap_cells": int((part["raw_gap_high_minus_normal"] > 0).sum()),
                "positive_gap_share": float((part["raw_gap_high_minus_normal"] > 0).mean()),
                "arrival_weighted_gap": float(np.average(part["raw_gap_high_minus_normal"], weights=weights)),
            }
        )
    return cells, pd.DataFrame(summary_rows)


def context_stratified_gaps(panel: pd.DataFrame, min_arrivals: int, min_route_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("Origin size", "origin_size_tier"),
        ("Route volume", "route_volume_tier"),
        ("Route carrier count", "route_carrier_count_tier"),
        ("Route peak share", "route_peak_share_tier"),
        ("Origin peak share", "origin_peak_share_tier"),
        ("Origin carrier breadth", "origin_carrier_breadth_tier"),
        ("Origin destination breadth", "origin_destination_breadth_tier"),
    ]
    weather_panel = panel.loc[panel["route_weather_pressure"] == 1].copy()
    rows = []
    for dimension, column in specs:
        for (stratum, weather_bin), part in weather_panel.groupby([column, "weather_intensity_bin"], sort=False):
            gap = gap_for_part(part, min_arrivals, min_route_days)
            if gap is not None:
                rows.append(
                    {
                        "context_dimension": dimension,
                        "context_stratum": str(stratum),
                        "weather_intensity_bin": weather_bin,
                        **gap,
                    }
                )
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells, cells
    summary_rows = []
    for dimension, part in cells.groupby("context_dimension", sort=False):
        weights = part["normal_arrivals"] + part["high_workload_arrivals"]
        summary_rows.append(
            {
                "context_dimension": dimension,
                "eligible_stratum_weather_cells": int(len(part)),
                "positive_gap_cells": int((part["raw_gap_high_minus_normal"] > 0).sum()),
                "positive_gap_share": float((part["raw_gap_high_minus_normal"] > 0).mean()),
                "arrival_weighted_gap": float(np.average(part["raw_gap_high_minus_normal"], weights=weights)),
                "minimum_cell_gap": float(part["raw_gap_high_minus_normal"].min()),
                "maximum_cell_gap": float(part["raw_gap_high_minus_normal"].max()),
            }
        )
    return cells, pd.DataFrame(summary_rows)


def fit_weighted_ridge_coefficients(
    model_df: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    high_cols: list[str],
    alpha: float = 1.0,
) -> dict[str, float]:
    num = pd.DataFrame(index=model_df.index)
    for col in numeric:
        values = pd.to_numeric(model_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if col not in high_cols:
            std = float(values.std())
            values = (values - float(values.mean())) / std if std > 0 else values * 0.0
        num[col] = values.astype(float)
    cat = pd.get_dummies(model_df[categorical].astype(str), columns=categorical, dtype=float)
    feature_df = pd.concat([pd.DataFrame({"intercept": np.ones(len(model_df))}, index=model_df.index), num, cat], axis=1)
    x = feature_df.to_numpy(dtype=float, copy=False)
    y = model_df["arr15_rate"].clip(0, 1).to_numpy(dtype=float)
    weights = pd.to_numeric(model_df["arr15_obs"], errors="coerce").fillna(1).clip(lower=1).to_numpy(dtype=float)
    xtx = x.T @ (x * weights[:, None])
    xty = x.T @ (weights * y)
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xtx + penalty, xty)
    return {name: float(value) for name, value in zip(feature_df.columns, beta)}


def sequential_operating_context_adjustment(panel: pd.DataFrame) -> pd.DataFrame:
    model_df = panel.loc[(panel["route_weather_pressure"] == 1) & (panel["arr15_obs"].fillna(0) > 0)].copy()
    model_df["log_flights"] = np.log1p(model_df["flights"])
    for weather_bin in WEATHER_BIN_ORDER:
        model_df[f"high_x_{weather_bin}"] = (
            (model_df["airport_workload_pressure"] == 1) & (model_df["weather_intensity_bin"] == weather_bin)
        ).astype(int)
    weather_numeric = [
        "route_weather_score",
        "route_adverse_hours",
        "route_max_wind_kt",
        "route_min_visibility_mi",
        "route_min_ceiling_ft",
        "route_precip_hours",
    ]
    control_sets = [
        ("Airport-calendar-weather", weather_numeric),
        ("Traffic-carrier added", weather_numeric + ["log_flights", "carriers", "origin_destinations", "origin_carriers"]),
        (
            "Full operating context",
            weather_numeric
            + ["log_flights", "carriers", "origin_destinations", "origin_carriers", "route_peak_share", "origin_peak_share"],
        ),
    ]
    categorical = ["origin", "destination", "month_id", "day_of_week", "origin_climate_group", "weather_intensity_bin"]
    high_cols = [f"high_x_{weather_bin}" for weather_bin in WEATHER_BIN_ORDER]
    rows = []
    for label, numeric_base in control_sets:
        numeric = numeric_base + high_cols
        coefficients = fit_weighted_ridge_coefficients(model_df, categorical, numeric, high_cols)
        for weather_bin in WEATHER_BIN_ORDER:
            part = model_df.loc[model_df["weather_intensity_bin"] == weather_bin]
            rows.append(
                {
                    "control_set": label,
                    "weather_intensity_bin": weather_bin,
                    "adjusted_gap_high_minus_normal": coefficients.get(f"high_x_{weather_bin}", float("nan")),
                    "route_days": int(len(part)),
                    "arrivals": int(part["arr15_obs"].sum()),
                }
            )
    return pd.DataFrame(rows)


def year_portability(panel: pd.DataFrame, min_arrivals: int, min_route_days: int) -> pd.DataFrame:
    rows = []
    panel = panel.copy()
    panel["year"] = pd.to_datetime(panel["date"]).dt.year.astype(int)
    for year, part in panel.groupby("year", sort=True):
        weather_panel = part.loc[part["route_weather_pressure"] == 1]
        for weather_bin in WEATHER_BIN_ORDER:
            gap = gap_for_part(weather_panel.loc[weather_panel["weather_intensity_bin"] == weather_bin], min_arrivals, min_route_days)
            if gap is not None:
                rows.append({"year": int(year), "weather_intensity_bin": weather_bin, **gap})
    return pd.DataFrame(rows)


def top20_burden_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    from .metrics import excess_burden

    rows = []
    panel = panel.copy()
    panel["year"] = pd.to_datetime(panel["date"]).dt.year.astype(int)
    for year, part in panel.groupby("year", sort=True):
        topk, _, _ = excess_burden(part)
        top20 = topk.loc[np.isclose(topk["top_route_day_share"], 0.20)]
        rows.append(
            {
                "year": int(year),
                "top20_captured_burden_share": float(top20["captured_burden_share"].iloc[0]) if not top20.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)
