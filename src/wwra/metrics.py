"""Core reliability-audit metric tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PRESSURE_ORDER, WEATHER_BIN_ORDER, WORKLOAD_BIN_ORDER
from .panel import add_state_features


def weighted_rate(success: pd.Series, total: pd.Series) -> float:
    denominator = pd.to_numeric(total, errors="coerce").fillna(0).sum()
    if denominator <= 0:
        return float("nan")
    numerator = pd.to_numeric(success, errors="coerce").fillna(0).sum()
    return float(numerator / denominator)


def ci_for_rate_diff(p1: float, n1: float, p0: float, n0: float) -> tuple[float, float, float]:
    if min(n1, n0) <= 0 or any(pd.isna(x) for x in [p1, p0]):
        return float("nan"), float("nan"), float("nan")
    se = math.sqrt(max(p1 * (1 - p1), 0) / n1 + max(p0 * (1 - p0), 0) / n0)
    diff = p1 - p0
    return se, diff - 1.96 * se, diff + 1.96 * se


def sample_closure(panel: pd.DataFrame, months: set[str], label: str) -> pd.DataFrame:
    weather_rows = panel.loc[panel["route_weather_pressure"] == 1]
    return pd.DataFrame(
        [
            {"item": "label", "value": label},
            {"item": "months", "value": len(months)},
            {"item": "route_days", "value": len(panel)},
            {"item": "flights", "value": int(panel["flights"].sum())},
            {"item": "origin_airports", "value": panel["origin"].nunique()},
            {"item": "destination_airports", "value": panel["destination"].nunique()},
            {"item": "routes", "value": panel["route"].nunique()},
            {"item": "origin_weather_match_share", "value": float(panel["origin_weather_obs"].notna().mean())},
            {
                "item": "destination_weather_match_share",
                "value": float(panel["destination_weather_obs"].notna().mean()),
            },
            {"item": "weather_constrained_route_days", "value": len(weather_rows)},
            {
                "item": "joint_weather_workload_route_days",
                "value": int((panel["pressure_cell"] == "joint_weather_workload").sum()),
            },
        ]
    )


def variable_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "arrival_delay_15_rate",
                "definition": "Share of observed arrivals delayed by at least 15 minutes.",
            },
            {
                "variable": "route_weather_score",
                "definition": "Route endpoint weather score combining adverse-weather hours, wind exceedance, visibility shortfall, and ceiling shortfall.",
            },
            {
                "variable": "workload_percentile",
                "definition": "Origin-airport daily flight-volume percentile within the same airport's observed distribution.",
            },
            {
                "variable": "excess_delayed_arrivals",
                "definition": "Positive delayed arrivals above the normal-workload reference within the same weather-intensity bin.",
            },
        ]
    )


def pressure_state_ladder(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in PRESSURE_ORDER:
        part = panel.loc[panel["pressure_cell"] == cell]
        rows.append(
            {
                "pressure_cell": cell,
                "route_days": len(part),
                "flights": int(part["flights"].sum()),
                "arr15_obs": int(part["arr15_obs"].sum()),
                "flight_weighted_arr15_rate": weighted_rate(part["arr15_delayed"], part["arr15_obs"]),
                "flight_weighted_cancel_rate": weighted_rate(part["cancelled_flights"], part["flights"]),
                "mean_arr_delay": float(np.average(part["mean_arr_delay"], weights=part["arr_delay_obs"]))
                if len(part) and part["arr_delay_obs"].sum() > 0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def adjusted_weather_bin_gaps(weather_panel: pd.DataFrame) -> pd.DataFrame:
    model_df = weather_panel.loc[weather_panel["arr15_obs"].fillna(0) > 0].copy()
    if len(model_df) < 100:
        return pd.DataFrame()
    model_df["target"] = model_df["arr15_rate"].clip(0, 1)
    model_df["log_flights"] = np.log1p(model_df["flights"])
    for weather_bin in WEATHER_BIN_ORDER:
        model_df[f"high_x_{weather_bin}"] = (
            (model_df["airport_workload_pressure"] == 1) & (model_df["weather_intensity_bin"] == weather_bin)
        ).astype(int)

    categorical = ["origin", "destination", "month_id", "day_of_week", "origin_climate_group", "weather_intensity_bin"]
    numeric = [
        "route_weather_score",
        "route_adverse_hours",
        "route_max_wind_kt",
        "route_min_visibility_mi",
        "route_min_ceiling_ft",
        "route_precip_hours",
        "log_flights",
        "carriers",
        "route_peak_share",
        "origin_destinations",
        "origin_carriers",
        "origin_peak_share",
    ] + [f"high_x_{weather_bin}" for weather_bin in WEATHER_BIN_ORDER]
    feature_df = model_df[categorical + numeric].copy()
    for col in numeric:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(with_mean=False), numeric),
                    ]
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(feature_df, model_df["target"], model__sample_weight=model_df["arr15_obs"].clip(lower=1))

    rows = []
    for weather_bin in WEATHER_BIN_ORDER:
        part = model_df.loc[model_df["weather_intensity_bin"] == weather_bin].copy()
        if part.empty:
            continue
        normal_x = part[categorical + numeric].copy()
        high_x = part[categorical + numeric].copy()
        for bin_name in WEATHER_BIN_ORDER:
            normal_x[f"high_x_{bin_name}"] = 0
            high_x[f"high_x_{bin_name}"] = 0
        high_x[f"high_x_{weather_bin}"] = 1
        weights = part["arr15_obs"].clip(lower=1).to_numpy()
        normal_pred = np.clip(pipe.predict(normal_x), 0, 1)
        high_pred = np.clip(pipe.predict(high_x), 0, 1)
        rows.append(
            {
                "weather_intensity_bin": weather_bin,
                "adjusted_normal_arr15_rate": float(np.average(normal_pred, weights=weights)),
                "adjusted_high_workload_arr15_rate": float(np.average(high_pred, weights=weights)),
                "adjusted_gap_high_minus_normal": float(np.average(high_pred - normal_pred, weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def weather_bin_workload_gap(panel: pd.DataFrame) -> pd.DataFrame:
    weather_panel = panel.loc[panel["route_weather_pressure"] == 1].copy()
    rows = []
    for weather_bin in WEATHER_BIN_ORDER:
        part = weather_panel.loc[weather_panel["weather_intensity_bin"] == weather_bin]
        normal = part.loc[part["airport_workload_pressure"] == 0]
        high = part.loc[part["airport_workload_pressure"] == 1]
        normal_rate = weighted_rate(normal["arr15_delayed"], normal["arr15_obs"])
        high_rate = weighted_rate(high["arr15_delayed"], high["arr15_obs"])
        se, ci_low, ci_high = ci_for_rate_diff(
            high_rate,
            high["arr15_obs"].sum(),
            normal_rate,
            normal["arr15_obs"].sum(),
        )
        rows.append(
            {
                "weather_intensity_bin": weather_bin,
                "normal_route_days": len(normal),
                "high_workload_route_days": len(high),
                "normal_flights": int(normal["flights"].sum()),
                "high_workload_flights": int(high["flights"].sum()),
                "normal_arr15_rate": normal_rate,
                "high_workload_arr15_rate": high_rate,
                "raw_gap_high_minus_normal": high_rate - normal_rate,
                "raw_gap_se": se,
                "raw_gap_ci_low": ci_low,
                "raw_gap_ci_high": ci_high,
            }
        )
    out = pd.DataFrame(rows)
    adjusted = adjusted_weather_bin_gaps(weather_panel)
    return out.merge(adjusted, on="weather_intensity_bin", how="left") if not adjusted.empty else out


def workload_weather_surface(panel: pd.DataFrame) -> pd.DataFrame:
    part = panel.loc[panel["route_weather_pressure"] == 1].copy()
    out = (
        part.groupby(["weather_intensity_bin", "workload_bin"], as_index=False)
        .agg(
            route_days=("route", "count"),
            flights=("flights", "sum"),
            arr15_delayed=("arr15_delayed", "sum"),
            arr15_obs=("arr15_obs", "sum"),
            cancelled_flights=("cancelled_flights", "sum"),
        )
    )
    out["flight_weighted_arr15_rate"] = out["arr15_delayed"] / out["arr15_obs"]
    out["flight_weighted_cancel_rate"] = out["cancelled_flights"] / out["flights"]
    out["weather_bin_order"] = out["weather_intensity_bin"].map({name: i + 1 for i, name in enumerate(WEATHER_BIN_ORDER)})
    out["workload_bin_order"] = out["workload_bin"].map({name: i + 1 for i, name in enumerate(WORKLOAD_BIN_ORDER)})
    return out.sort_values(["weather_bin_order", "workload_bin_order"])


def excess_burden(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weather_panel = panel.loc[panel["route_weather_pressure"] == 1].copy()
    baseline = (
        weather_panel.loc[weather_panel["airport_workload_pressure"] == 0]
        .groupby("weather_intensity_bin")
        .apply(lambda group: weighted_rate(group["arr15_delayed"], group["arr15_obs"]), include_groups=False)
        .rename("normal_workload_baseline_arr15_rate")
        .reset_index()
    )
    joint = weather_panel.loc[weather_panel["airport_workload_pressure"] == 1].copy()
    joint = joint.merge(baseline, on="weather_intensity_bin", how="left")
    fallback = weighted_rate(
        weather_panel.loc[weather_panel["airport_workload_pressure"] == 0, "arr15_delayed"],
        weather_panel.loc[weather_panel["airport_workload_pressure"] == 0, "arr15_obs"],
    )
    joint["normal_workload_baseline_arr15_rate"] = joint["normal_workload_baseline_arr15_rate"].fillna(fallback)
    joint["positive_excess_arr15_delayed"] = (
        joint["arr15_delayed"] - joint["normal_workload_baseline_arr15_rate"] * joint["arr15_obs"]
    ).clip(lower=0)
    joint = joint.sort_values("positive_excess_arr15_delayed", ascending=False).reset_index(drop=True)
    total = float(joint["positive_excess_arr15_delayed"].sum())

    top_rows = []
    for share in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        k = max(1, int(math.ceil(len(joint) * share))) if len(joint) else 0
        captured = float(joint.head(k)["positive_excess_arr15_delayed"].sum()) if k else 0.0
        top_rows.append(
            {
                "top_route_day_share": share,
                "route_days": k,
                "captured_positive_excess_arr15": captured,
                "captured_burden_share": captured / total if total > 0 else float("nan"),
                "total_positive_excess_arr15": total,
            }
        )

    origin = (
        joint.groupby(["origin", "origin_climate_group"], as_index=False)
        .agg(
            route_days=("route", "count"),
            flights=("flights", "sum"),
            positive_excess_arr15=("positive_excess_arr15_delayed", "sum"),
        )
        .sort_values("positive_excess_arr15", ascending=False)
    )
    route = (
        joint.groupby(["origin", "destination", "route", "origin_climate_group"], as_index=False)
        .agg(
            route_days=("date", "count"),
            flights=("flights", "sum"),
            positive_excess_arr15=("positive_excess_arr15_delayed", "sum"),
        )
        .sort_values("positive_excess_arr15", ascending=False)
    )
    if total > 0:
        origin["burden_share"] = origin["positive_excess_arr15"] / total
        route["burden_share"] = route["positive_excess_arr15"] / total
    return pd.DataFrame(top_rows), origin, route


def climate_failure_forms(panel: pd.DataFrame) -> pd.DataFrame:
    out = (
        panel.groupby(["origin_climate_group", "pressure_cell"], as_index=False)
        .agg(
            route_days=("route", "count"),
            flights=("flights", "sum"),
            arr15_delayed=("arr15_delayed", "sum"),
            arr15_obs=("arr15_obs", "sum"),
            cancelled_flights=("cancelled_flights", "sum"),
            arr_delay_sum=("arr_delay_sum", "sum"),
            arr_delay_obs=("arr_delay_obs", "sum"),
        )
    )
    out["flight_weighted_arr15_rate"] = out["arr15_delayed"] / out["arr15_obs"]
    out["flight_weighted_cancel_rate"] = out["cancelled_flights"] / out["flights"]
    out["mean_arr_delay"] = out["arr_delay_sum"] / out["arr_delay_obs"]
    out["pressure_order"] = out["pressure_cell"].map({name: i + 1 for i, name in enumerate(PRESSURE_ORDER)})
    return out.sort_values(["origin_climate_group", "pressure_order"])


def threshold_robustness(panel: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    base = panel.drop(
        columns=[
            "high_workload_threshold",
            "workload_percentile",
            "airport_workload_pressure",
            "workload_bin",
            "pressure_cell",
        ],
        errors="ignore",
    )
    rows = []
    for quantile in quantiles:
        tmp = add_state_features(base.copy(), quantile)
        gaps = weather_bin_workload_gap(tmp)
        topk, _, _ = excess_burden(tmp)
        top20 = topk.loc[np.isclose(topk["top_route_day_share"], 0.20)]
        for _, row in gaps.iterrows():
            rows.append(
                {
                    "high_workload_quantile": quantile,
                    "weather_intensity_bin": row["weather_intensity_bin"],
                    "raw_gap_high_minus_normal": row["raw_gap_high_minus_normal"],
                    "adjusted_gap_high_minus_normal": row.get("adjusted_gap_high_minus_normal", np.nan),
                    "top20_captured_burden_share": float(top20["captured_burden_share"].iloc[0])
                    if not top20.empty
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
