"""Build an airport-route-day reliability panel from public flight and weather data."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AIRPORTS, CLIMATE_BY_AIRPORT, ON_TIME_USECOLS, WEATHER_BIN_ORDER, WORKLOAD_BIN_ORDER
from .utils import first_csv_member, safe_div


def parse_airport_list(text: str | None) -> list[str]:
    if not text:
        return AIRPORTS
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def parse_precip(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).strip()
    if text in {"", "M"}:
        return 0.0
    if text.upper() == "T":
        return 0.005
    try:
        return float(text)
    except ValueError:
        return 0.0


def month_from_on_time_zip(path: Path) -> str:
    parts = path.stem.split("_")
    return f"{parts[-2]}-{parts[-1]}"


def build_weather_daily(root: Path, airports: list[str], years: set[int]) -> pd.DataFrame:
    weather_root = root / "data" / "external" / "noaa_asos_metar"
    parts: list[pd.DataFrame] = []
    for year in sorted(years):
        for airport in airports:
            path = weather_root / str(year) / f"{airport}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, low_memory=False)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["valid"], errors="coerce").dt.date
            df = df.loc[df["date"].notna()].copy()
            df["sknt"] = pd.to_numeric(df.get("sknt"), errors="coerce")
            df["vsby"] = pd.to_numeric(df.get("vsby"), errors="coerce")
            df["skyl1"] = pd.to_numeric(df.get("skyl1"), errors="coerce")
            p01i = df.get("p01i", pd.Series(index=df.index, dtype="object"))
            df["p01i_num"] = p01i.map(parse_precip)
            df["low_visibility_hour"] = (df["vsby"] < 3).astype(int)
            df["high_wind_hour"] = (df["sknt"] >= 20).astype(int)
            df["low_ceiling_hour"] = (df["skyl1"] <= 1000).astype(int)
            df["precip_hour"] = (df["p01i_num"] > 0).astype(int)
            daily = (
                df.groupby(["station", "date"], as_index=False)
                .agg(
                    weather_obs=("valid", "count"),
                    max_wind_kt=("sknt", "max"),
                    min_visibility_mi=("vsby", "min"),
                    min_ceiling_ft=("skyl1", "min"),
                    low_visibility_hours=("low_visibility_hour", "sum"),
                    high_wind_hours=("high_wind_hour", "sum"),
                    low_ceiling_hours=("low_ceiling_hour", "sum"),
                    precip_hours=("precip_hour", "sum"),
                )
                .rename(columns={"station": "airport"})
            )
            daily["weather_pressure"] = (
                (daily["low_visibility_hours"] > 0)
                | (daily["high_wind_hours"] >= 2)
                | (daily["low_ceiling_hours"] >= 2)
                | (daily["precip_hours"] >= 2)
            ).astype(int)
            daily["airport_weather_score"] = (
                daily["low_visibility_hours"].fillna(0)
                + daily["low_ceiling_hours"].fillna(0)
                + daily["precip_hours"].fillna(0)
                + daily["high_wind_hours"].fillna(0)
                + np.maximum(daily["max_wind_kt"].fillna(0) - 20, 0) / 5
                + np.maximum(3 - daily["min_visibility_mi"].fillna(10), 0) * 2
                + np.maximum(1000 - daily["min_ceiling_ft"].fillna(10000), 0) / 250
            )
            parts.append(daily)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_on_time_route_days(
    root: Path,
    months: set[str],
    origins: list[str],
    destinations: list[str],
    chunksize: int,
) -> pd.DataFrame:
    data_dir = root / "data" / "external" / "bts_on_time"
    files = sorted(path for path in data_dir.glob("bts_on_time_*.zip") if month_from_on_time_zip(path) in months)
    route_parts: list[pd.DataFrame] = []
    workload_parts: list[pd.DataFrame] = []
    for zip_path in files:
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = first_csv_member(zip_path)
            with zf.open(csv_name) as fh:
                reader = pd.read_csv(
                    fh,
                    usecols=lambda col: col in ON_TIME_USECOLS,
                    chunksize=chunksize,
                    low_memory=False,
                )
                for chunk in reader:
                    chunk = chunk.loc[chunk["Origin"].isin(origins)].copy()
                    if chunk.empty:
                        continue
                    chunk["date"] = pd.to_datetime(chunk["FlightDate"], errors="coerce").dt.date
                    chunk = chunk.loc[chunk["date"].notna()].copy()
                    chunk["month_id"] = pd.to_datetime(chunk["date"]).dt.to_period("M").astype(str)
                    dep_time = pd.to_numeric(chunk["CRSDepTime"], errors="coerce").fillna(-1).astype(int)
                    chunk["crs_dep_hour"] = dep_time.floordiv(100)
                    chunk["scheduled_peak_departure"] = chunk["crs_dep_hour"].between(6, 9) | chunk["crs_dep_hour"].between(16, 19)
                    chunk["cancelled_num"] = pd.to_numeric(chunk["Cancelled"], errors="coerce").fillna(0)
                    chunk["arr_delay"] = pd.to_numeric(chunk["ArrDelay"], errors="coerce")
                    chunk["arr15"] = pd.to_numeric(chunk["ArrDel15"], errors="coerce")

                    workload = (
                        chunk.groupby(["date", "month_id", "Origin"], as_index=False)
                        .agg(
                            origin_daily_flights=("Dest", "size"),
                            origin_destinations=("Dest", "nunique"),
                            origin_carriers=("Reporting_Airline", "nunique"),
                            origin_peak_scheduled_flights=("scheduled_peak_departure", "sum"),
                        )
                        .rename(columns={"Origin": "origin"})
                    )
                    workload["origin_peak_share"] = workload.apply(
                        lambda row: safe_div(row["origin_peak_scheduled_flights"], row["origin_daily_flights"]),
                        axis=1,
                    )
                    workload_parts.append(workload)

                    chunk = chunk.loc[chunk["Dest"].isin(destinations) & (chunk["Dest"] != chunk["Origin"])].copy()
                    if chunk.empty:
                        continue
                    grouped = (
                        chunk.groupby(["date", "month_id", "Origin", "Dest"], as_index=False)
                        .agg(
                            flights=("Dest", "size"),
                            carriers=("Reporting_Airline", "nunique"),
                            cancelled_flights=("cancelled_num", "sum"),
                            arr_delay_sum=("arr_delay", "sum"),
                            arr_delay_obs=("arr_delay", "count"),
                            arr15_delayed=("arr15", "sum"),
                            arr15_obs=("arr15", "count"),
                            route_peak_scheduled_flights=("scheduled_peak_departure", "sum"),
                            mean_day_of_week=("DayOfWeek", "mean"),
                            mean_distance=("Distance", "mean"),
                        )
                        .rename(columns={"Origin": "origin", "Dest": "destination"})
                    )
                    grouped["route_peak_share"] = grouped.apply(
                        lambda row: safe_div(row["route_peak_scheduled_flights"], row["flights"]),
                        axis=1,
                    )
                    route_parts.append(grouped)
    if not route_parts:
        return pd.DataFrame()
    route_days = pd.concat(route_parts, ignore_index=True)
    route_days = (
        route_days.groupby(["date", "month_id", "origin", "destination"], as_index=False)
        .agg(
            flights=("flights", "sum"),
            carriers=("carriers", "max"),
            cancelled_flights=("cancelled_flights", "sum"),
            arr_delay_sum=("arr_delay_sum", "sum"),
            arr_delay_obs=("arr_delay_obs", "sum"),
            arr15_delayed=("arr15_delayed", "sum"),
            arr15_obs=("arr15_obs", "sum"),
            route_peak_scheduled_flights=("route_peak_scheduled_flights", "sum"),
            mean_day_of_week=("mean_day_of_week", "mean"),
            mean_distance=("mean_distance", "mean"),
        )
    )
    route_days["route_peak_share"] = route_days.apply(
        lambda row: safe_div(row["route_peak_scheduled_flights"], row["flights"]),
        axis=1,
    )
    if workload_parts:
        workload = pd.concat(workload_parts, ignore_index=True)
        workload = (
            workload.groupby(["date", "month_id", "origin"], as_index=False)
            .agg(
                origin_daily_flights=("origin_daily_flights", "sum"),
                origin_destinations=("origin_destinations", "max"),
                origin_carriers=("origin_carriers", "max"),
                origin_peak_scheduled_flights=("origin_peak_scheduled_flights", "sum"),
            )
        )
        workload["origin_peak_share"] = workload.apply(
            lambda row: safe_div(row["origin_peak_scheduled_flights"], row["origin_daily_flights"]),
            axis=1,
        )
        route_days = route_days.merge(workload, on=["date", "month_id", "origin"], how="left", validate="many_to_one")
    route_days["cancel_rate"] = route_days.apply(lambda row: safe_div(row["cancelled_flights"], row["flights"]), axis=1)
    route_days["arr15_rate"] = route_days.apply(lambda row: safe_div(row["arr15_delayed"], row["arr15_obs"]), axis=1)
    route_days["mean_arr_delay"] = route_days.apply(lambda row: safe_div(row["arr_delay_sum"], row["arr_delay_obs"]), axis=1)
    route_days["route"] = route_days["origin"] + "-" + route_days["destination"]
    return route_days


def rename_weather(weather: pd.DataFrame, key: str) -> pd.DataFrame:
    prefix = "origin" if key == "origin" else "destination"
    return weather.rename(
        columns={
            "airport": key,
            "weather_pressure": f"{prefix}_weather_pressure",
            "weather_obs": f"{prefix}_weather_obs",
            "airport_weather_score": f"{prefix}_weather_score",
            "max_wind_kt": f"{prefix}_max_wind_kt",
            "min_visibility_mi": f"{prefix}_min_visibility_mi",
            "min_ceiling_ft": f"{prefix}_min_ceiling_ft",
            "low_visibility_hours": f"{prefix}_low_visibility_hours",
            "high_wind_hours": f"{prefix}_high_wind_hours",
            "low_ceiling_hours": f"{prefix}_low_ceiling_hours",
            "precip_hours": f"{prefix}_precip_hours",
        }
    )


def add_state_features(panel: pd.DataFrame, high_workload_quantile: float) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel["day_of_week"] = pd.to_datetime(panel["date"]).dt.dayofweek + 1
    panel["origin_climate_group"] = panel["origin"].map(CLIMATE_BY_AIRPORT)
    panel["destination_climate_group"] = panel["destination"].map(CLIMATE_BY_AIRPORT)

    workload_daily = panel[
        ["date", "origin", "origin_daily_flights", "origin_destinations", "origin_carriers", "origin_peak_share"]
    ].drop_duplicates(["date", "origin"])
    thresholds = (
        workload_daily.groupby("origin")["origin_daily_flights"]
        .quantile(high_workload_quantile)
        .rename("high_workload_threshold")
        .reset_index()
    )
    workload_daily = workload_daily.merge(thresholds, on="origin", how="left")
    workload_daily["workload_percentile"] = workload_daily.groupby("origin")["origin_daily_flights"].rank(method="average", pct=True)
    workload_daily["airport_workload_pressure"] = (
        workload_daily["origin_daily_flights"] >= workload_daily["high_workload_threshold"]
    ).astype(int)
    workload_daily["workload_bin"] = pd.cut(
        workload_daily["workload_percentile"],
        bins=[0, 0.5, 0.75, 0.9, 1.000001],
        labels=WORKLOAD_BIN_ORDER,
        include_lowest=True,
    ).astype(str)
    panel = panel.drop(columns=["origin_daily_flights", "origin_destinations", "origin_carriers", "origin_peak_share"])
    panel = panel.merge(workload_daily, on=["date", "origin"], how="left", validate="many_to_one")

    panel["origin_weather_pressure"] = panel["origin_weather_pressure"].fillna(0).astype(int)
    panel["destination_weather_pressure"] = panel["destination_weather_pressure"].fillna(0).astype(int)
    panel["route_weather_pressure"] = (
        (panel["origin_weather_pressure"] == 1) | (panel["destination_weather_pressure"] == 1)
    ).astype(int)
    panel["pressure_cell"] = np.select(
        [
            (panel["route_weather_pressure"] == 1) & (panel["airport_workload_pressure"] == 1),
            panel["route_weather_pressure"] == 1,
            panel["airport_workload_pressure"] == 1,
        ],
        ["joint_weather_workload", "weather_only", "workload_only"],
        default="low_pressure",
    )

    adverse_cols = [
        "origin_low_visibility_hours",
        "destination_low_visibility_hours",
        "origin_low_ceiling_hours",
        "destination_low_ceiling_hours",
        "origin_precip_hours",
        "destination_precip_hours",
        "origin_high_wind_hours",
        "destination_high_wind_hours",
    ]
    panel["route_adverse_hours"] = panel[adverse_cols].fillna(0).sum(axis=1)
    panel["route_precip_hours"] = panel[["origin_precip_hours", "destination_precip_hours"]].fillna(0).sum(axis=1)
    panel["route_max_wind_kt"] = panel[["origin_max_wind_kt", "destination_max_wind_kt"]].fillna(0).max(axis=1)
    panel["route_min_visibility_mi"] = panel[["origin_min_visibility_mi", "destination_min_visibility_mi"]].fillna(10).min(axis=1)
    panel["route_min_ceiling_ft"] = panel[["origin_min_ceiling_ft", "destination_min_ceiling_ft"]].fillna(10000).min(axis=1)
    panel["route_weather_score"] = (
        panel["route_adverse_hours"]
        + np.maximum(panel["route_max_wind_kt"] - 20, 0) / 5
        + np.maximum(3 - panel["route_min_visibility_mi"], 0) * 2
        + np.maximum(1000 - panel["route_min_ceiling_ft"], 0) / 250
    )
    panel["weather_intensity_bin"] = "W0 clear"
    weather_mask = panel["route_weather_pressure"] == 1
    if weather_mask.any():
        ranks = panel.loc[weather_mask, "route_weather_score"].rank(method="first")
        panel.loc[weather_mask, "weather_intensity_bin"] = pd.qcut(ranks, 4, labels=WEATHER_BIN_ORDER).astype(str)
    return panel


def build_panel(
    root: Path,
    months: set[str],
    origins: list[str],
    destinations: list[str],
    chunksize: int = 250_000,
    high_workload_quantile: float = 0.75,
) -> pd.DataFrame:
    years = {int(month[:4]) for month in months}
    weather_airports = sorted(set(origins + destinations))
    weather = build_weather_daily(root, weather_airports, years)
    route_days = build_on_time_route_days(root, months, origins, destinations, chunksize)
    if weather.empty or route_days.empty:
        raise RuntimeError("Weather or route-day panel is empty. Check downloaded public data files.")
    panel = route_days.merge(rename_weather(weather, "origin"), on=["origin", "date"], how="left", validate="many_to_one")
    panel = panel.merge(rename_weather(weather, "destination"), on=["destination", "date"], how="left", validate="many_to_one")
    return add_state_features(panel, high_workload_quantile)
