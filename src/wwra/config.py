"""Configuration for the public airport-route reliability audit workflow."""

from __future__ import annotations

from pathlib import Path


CLIMATE_GROUPS = {
    "Cold/snowbelt": ["BOS", "BUF", "DTW", "MSP", "ORD"],
    "Pacific low-cloud": ["LAX", "SFO", "SAN", "SEA", "PDX", "OAK", "SNA", "SJC"],
    "Western dry / high-altitude": ["LAS", "PHX", "DEN", "SLC", "ABQ", "BOI", "SMF"],
    "Humid subtropical / thunderstorm": [
        "ATL",
        "CLT",
        "DFW",
        "IAH",
        "HOU",
        "BNA",
        "AUS",
        "RDU",
        "BHM",
        "CHS",
        "SAT",
        "MSY",
    ],
    "Tropical / Florida": ["MIA", "FLL", "MCO", "TPA"],
    "Northeast dense metro": ["DCA", "IAD", "BWI", "LGA", "JFK", "EWR", "PHL"],
}

AIRPORTS = sorted({airport for airports in CLIMATE_GROUPS.values() for airport in airports})
CLIMATE_BY_AIRPORT = {
    airport: group for group, airports in CLIMATE_GROUPS.items() for airport in airports
}

ON_TIME_USECOLS = {
    "Year",
    "Month",
    "FlightDate",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Cancelled",
    "DepDelay",
    "ArrDelay",
    "DepDel15",
    "ArrDel15",
    "Distance",
}

PRESSURE_ORDER = ["low_pressure", "workload_only", "weather_only", "joint_weather_workload"]
WEATHER_BIN_ORDER = ["W1 mild", "W2 moderate", "W3 severe", "W4 extreme"]
WORKLOAD_BIN_ORDER = ["P0-50", "P50-75", "P75-90", "P90-100"]


def project_paths(root: Path) -> dict[str, Path]:
    return {
        "external": root / "data" / "external",
        "processed": root / "data" / "processed" / "pressure_reliability",
        "tables": root / "results" / "tables" / "pressure_reliability",
        "logs": root / "results" / "logs" / "pressure_reliability",
    }


def ensure_paths(root: Path) -> dict[str, Path]:
    paths = project_paths(root)
    for key in ["processed", "tables", "logs"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths
