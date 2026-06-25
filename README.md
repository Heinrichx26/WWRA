# WWRA

WWRA stands for Weather-Workload Reliability Audit. This repository contains code for an airport-route operational reliability audit. The workflow builds a route-day panel from public flight and weather data, computes weather-workload reliability states, estimates same-weather workload gaps, and produces tabular audit outputs for burden concentration and robustness checks.

No raw data, manuscript files, submission documents, figures, or model-comparison code are included. Curated derived CSV result tables are included as reproduction material.

## Review Process Note

Reliability Engineering & System Safety uses a single anonymized review process according to Elsevier's guide for authors: https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety/publish/guide-for-authors. Reviewer identities are hidden from authors, while author information does not need to be removed from the manuscript.

## Public Data Sources

The workflow expects users to download the public source data directly:

- Bureau of Transportation Statistics Airline On-Time Performance data: https://transtats.bts.gov/ONTIME/
- National Oceanic and Atmospheric Administration ASOS/METAR weather observations via Iowa Environmental Mesonet: https://mesonet.agron.iastate.edu/request/download.phtml
- OurAirports open airport metadata: https://ourairports.com/data/

Place downloaded flight files under `data/external/bts_on_time/` and hourly weather files under `data/external/noaa_asos_metar/<year>/`. The repository does not redistribute these data.

## Repository Layout

```text
.
+-- data/
|   +-- external/          # user-downloaded public source data, ignored by git
|   +-- processed/         # generated panels, ignored by git
+-- results/
|   +-- tables/            # curated derived CSV result tables
+-- scripts/
|   +-- build_panel.py
|   +-- run_core_tables.py
|   +-- run_operational_robustness.py
+-- src/
    +-- wwra/
```

## Installation

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Expected Input Names

BTS monthly ZIP files should be named like:

```text
data/external/bts_on_time/bts_on_time_2024_01.zip
```

ASOS/METAR hourly CSV files should be named like:

```text
data/external/noaa_asos_metar/2024/ORD.csv
```

The code uses relative paths from the repository root and does not require local machine-specific paths.

## Quick Run

Build a smoke panel for the first three months of 2024:

```bash
python scripts/build_panel.py --months 2024-01,2024-02,2024-03 --label smoke_2024q1
python scripts/run_core_tables.py --label smoke_2024q1
python scripts/run_operational_robustness.py --label smoke_2024q1
```

Build the 2024-2025 panel:

```bash
python scripts/build_panel.py --months 2024-01:2025-12 --label full_2024_2025
python scripts/run_core_tables.py --label full_2024_2025
python scripts/run_operational_robustness.py --label full_2024_2025
```

## Outputs

Curated derived CSV tables for the full 2024-2025 analysis are included in `results/tables/pressure_reliability/`. Running the scripts with a new label writes additional CSV tables to the same folder. Key outputs include:

- `sample_closure_<label>.csv`
- `pressure_state_ladder_<label>.csv`
- `weather_bin_workload_gap_<label>.csv`
- `topk_burden_share_<label>.csv`
- `threshold_robustness_<label>.csv`
- `operational_context_stratified_gap_summary_<label>.csv`
- `airport_fixed_weather_gap_summary_<label>.csv`
- `sequential_operating_context_adjustment_<label>.csv`
- `yearly_weather_gap_portability_<label>.csv`

The included result tables are derived summaries computed from public sources. They do not contain the raw BTS flight files, raw ASOS/METAR observations, processed route-day panel, manuscript files, figure files, or comparison-model outputs.

## Scope

This release focuses on transparent tabular reproduction of the reliability audit. It excludes plotting code, manuscript files, prepared submission files, raw data, processed panels, and comparison-model implementations.
