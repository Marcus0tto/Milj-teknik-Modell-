import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

from main import (
    RepresentativeYearData,
    SolarStorageEstimate,
    WindOnlyStorageEstimate,
    calculate_solar_storage_requirement,
    calculate_wind_only_storage_requirement,
)


MARRAKESH_STORAGE_ROUND_TRIP_EFFICIENCY = 0.30
MONTH_NAMES = [
    "jan", "feb", "mar", "apr", "maj", "jun",
    "jul", "aug", "sep", "okt", "nov", "dec",
]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


@dataclass(frozen=True)
class MarrakeshScenario:
    name: str
    timestamps: list[datetime]
    consumption_mwh: list[float]
    solar_mwh: list[float]
    wind_mwh: list[float]
    balance_mwh: list[float]
    storage_mwh: list[float]
    storage_capacity_mwh: float
    solar_factor: float
    wind_factor: float
    charge_power_mw: float
    discharge_power_mw: float


def _read_monthly_solar(path: Path) -> dict[int, list[float]]:
    by_hour: dict[int, list[float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row_number, row in enumerate(csv.reader(file), start=1):
            if not row or not row[0].strip():
                continue
            label = row[0].strip()
            if label.lower().startswith("sum"):
                continue
            parts = label.split("-")
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                continue
            hour = int(parts[0])
            if not 0 <= hour < 24 or hour in by_hour or len(row) != 13:
                raise ValueError(f"Invalid solar row {row_number} in {path}")
            try:
                by_hour[hour] = [
                    float(value) if value.strip() else 0.0
                    for value in row[1:]
                ]
            except ValueError as error:
                raise ValueError(f"Invalid solar value in {path} row {row_number}") from error
    if not by_hour:
        raise ValueError(f"No hourly solar profile found in {path}")
    return by_hour


def _read_monthly_consumption(path: Path) -> list[list[float]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0].strip() == "Timme"),
        None,
    )
    if header_index is None or len(rows[header_index]) != 13:
        raise ValueError(f"Could not find 12-month consumption header in {path}")

    by_hour: dict[int, list[float]] = {}
    for row_number, row in enumerate(rows[header_index + 1:], start=header_index + 2):
        if not row or not row[0].strip():
            continue
        try:
            hour = int(row[0].split(":", 1)[0])
        except ValueError:
            continue
        if not 0 <= hour < 24 or hour in by_hour or len(row) != 13:
            raise ValueError(f"Invalid consumption row {row_number} in {path}")
        try:
            by_hour[hour] = [float(value) * 1000 for value in row[1:]]
        except ValueError as error:
            raise ValueError(
                f"Invalid consumption value in {path} row {row_number}"
            ) from error
    if len(by_hour) != 24:
        raise ValueError(f"Expected 24 hourly consumption rows in {path}")
    return [by_hour[hour] for hour in range(24)]


def _read_monthly_wind_speeds(path: Path) -> list[list[float]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    by_hour: dict[int, list[float]] = {}
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or not row[0].strip():
            continue
        try:
            hour = int(row[0])
        except ValueError:
            continue
        if not 0 <= hour < 24 or hour in by_hour or len(row) != 13:
            raise ValueError(f"Invalid wind-speed row {row_number} in {path}")
        try:
            values = [float(value) for value in row[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid wind speed in {path} row {row_number}") from error
        if any(speed < 0 for speed in values):
            raise ValueError(f"Negative wind speed in {path} row {row_number}")
        by_hour[hour] = values
    if len(by_hour) != 24:
        raise ValueError(f"Expected 24 hourly wind-speed rows in {path}")
    return [by_hour[hour] for hour in range(24)]


def load_marrakesh_representative_year(data_directory: Path) -> RepresentativeYearData:
    solar_by_hour = _read_monthly_solar(data_directory / "marrakesh_sol_data.csv")
    consumption_by_hour = _read_monthly_consumption(
        data_directory / "marrakesh_konsumption.csv"
    )
    wind_speeds_by_hour = _read_monthly_wind_speeds(
        data_directory / "heatmapData.csv"
    )

    timestamps: list[datetime] = []
    consumption: list[float] = []
    solar: list[float] = []
    wind_proxy: list[float] = []
    for month, days_in_month in enumerate(MONTH_DAYS):
        for day in range(1, days_in_month + 1):
            for hour in range(24):
                timestamps.append(datetime(2023, month + 1, day, hour))
                consumption.append(consumption_by_hour[hour][month])
                solar.append(solar_by_hour.get(hour, [0.0] * 12)[month] / 1000)
                wind_proxy.append(wind_speeds_by_hour[hour][month] ** 3)

    return RepresentativeYearData(
        timestamps=timestamps,
        consumption_mwh=consumption,
        solar_production_mwh=solar,
        wind_farm_production_mwh=wind_proxy,
        wind_farm_scale_factor=1.0,
        hydropower_production_mwh=[0.0] * len(timestamps),
        solar_source_year=2023,
        consumption_source_years=[2023],
        wind_profile_source_years=[2023],
        hydropower_source_year=None,
        hydropower_power_limit_mw=0.0,
    )


def _scenario_from_solar_estimate(
    name: str,
    estimate: SolarStorageEstimate,
) -> MarrakeshScenario:
    return MarrakeshScenario(
        name=name,
        timestamps=estimate.timestamps,
        consumption_mwh=estimate.consumption_mwh,
        solar_mwh=estimate.solar_production_mwh,
        wind_mwh=estimate.wind_farm_production_mwh,
        balance_mwh=estimate.surplus_deficit_mwh,
        storage_mwh=estimate.storage_state_mwh,
        storage_capacity_mwh=estimate.storage_capacity_mwh,
        solar_factor=estimate.solar_scale_factor,
        wind_factor=estimate.wind_farm_scale_factor,
        charge_power_mw=estimate.maximum_charge_power_mw,
        discharge_power_mw=estimate.maximum_discharge_power_mw,
    )


def _scenario_from_wind_estimate(
    estimate: WindOnlyStorageEstimate,
) -> MarrakeshScenario:
    return MarrakeshScenario(
        name="Vindproxy",
        timestamps=estimate.timestamps,
        consumption_mwh=estimate.consumption_mwh,
        solar_mwh=[0.0] * len(estimate.timestamps),
        wind_mwh=estimate.wind_production_mwh,
        balance_mwh=estimate.surplus_deficit_mwh,
        storage_mwh=estimate.storage_state_mwh,
        storage_capacity_mwh=estimate.storage_capacity_mwh,
        solar_factor=0.0,
        wind_factor=estimate.wind_scale_factor,
        charge_power_mw=estimate.maximum_charge_power_mw,
        discharge_power_mw=estimate.maximum_discharge_power_mw,
    )


def _estimate_hybrid(data: RepresentativeYearData) -> SolarStorageEstimate:
    wind_only = calculate_wind_only_storage_requirement(
        data,
        municipality_share=1.0,
        round_trip_efficiency=MARRAKESH_STORAGE_ROUND_TRIP_EFFICIENCY,
    )

    def estimate_at(scale: float) -> SolarStorageEstimate:
        scaled_data = replace(
            data,
            wind_farm_production_mwh=[
                value * scale for value in data.wind_farm_production_mwh
            ],
            wind_farm_scale_factor=scale,
        )
        return calculate_solar_storage_requirement(
            scaled_data,
            municipality_share=1.0,
            round_trip_efficiency=MARRAKESH_STORAGE_ROUND_TRIP_EFFICIENCY,
        )

    upper = wind_only.wind_scale_factor
    for _ in range(8):
        scales = [upper * index / 12 for index in range(13)]
        estimates = [estimate_at(scale) for scale in scales]
        best_index = min(
            range(len(estimates)),
            key=lambda index: estimates[index].storage_capacity_mwh,
        )
        if best_index < len(scales) - 1:
            break
        upper *= 2
    else:
        raise ValueError("Could not bracket the minimum hybrid storage requirement")

    lower_bound = scales[max(0, best_index - 1)]
    upper_bound = scales[min(len(scales) - 1, best_index + 1)]
    ratio = (sqrt(5) - 1) / 2
    left = upper_bound - ratio * (upper_bound - lower_bound)
    right = lower_bound + ratio * (upper_bound - lower_bound)
    left_estimate = estimate_at(left)
    right_estimate = estimate_at(right)
    for _ in range(20):
        if left_estimate.storage_capacity_mwh < right_estimate.storage_capacity_mwh:
            upper_bound, right, right_estimate = right, left, left_estimate
            left = upper_bound - ratio * (upper_bound - lower_bound)
            left_estimate = estimate_at(left)
        else:
            lower_bound, left, left_estimate = left, right, right_estimate
            right = lower_bound + ratio * (upper_bound - lower_bound)
            right_estimate = estimate_at(right)
    return estimate_at((lower_bound + upper_bound) / 2)


def _plot_scenario(scenario: MarrakeshScenario, output_path: Path) -> None:
    daily: dict[str, list[float]] = {
        "consumption": [],
        "solar": [],
        "wind": [],
        "balance": [],
        "storage": [],
    }
    for day_index in range(365):
        start = day_index * 24
        end = start + 24
        daily["consumption"].append(sum(scenario.consumption_mwh[start:end]))
        daily["solar"].append(sum(scenario.solar_mwh[start:end]))
        daily["wind"].append(sum(scenario.wind_mwh[start:end]))
        daily["balance"].append(sum(scenario.balance_mwh[start:end]))
        daily["storage"].append(scenario.storage_mwh[end] / 1_000_000)

    def mean_7(values: list[float]) -> list[float]:
        return [
            sum(values[max(0, index - 6):index + 1])
            / len(values[max(0, index - 6):index + 1])
            for index in range(len(values))
        ]

    days = [datetime(2023, 1, 1) + timedelta(days=index) for index in range(365)]
    solar_peak = max(scenario.solar_mwh, default=0.0)
    wind_peak = max(scenario.wind_mwh, default=0.0)
    total_peak = max(
        solar + wind
        for solar, wind in zip(scenario.solar_mwh, scenario.wind_mwh)
    )
    solar_capacity_kwp = scenario.solar_factor * 1000

    figure, axis = plt.subplots(figsize=(15, 7))
    for key, label, color in (
        ("solar", "Solproduktion", "#E07A28"),
        ("wind", "Vindproxy", "#7B2CBF"),
        ("consumption", "Förbrukning", "#2457A6"),
        ("balance", "Överskott / underskott", "#398C6D"),
    ):
        values = daily[key]
        if any(values):
            axis.plot(days, mean_7(values), label=f"{label} (7-dagars medel)", color=color, linewidth=1.8)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylabel("Energi (MWh per dygn)")
    axis.set_xlabel("Månad")
    axis.set_xticks([datetime(2023, month, 15) for month in range(1, 13)])
    axis.set_xticklabels(MONTH_NAMES)
    axis.grid(True, alpha=0.3)

    storage_axis = axis.twinx()
    storage_axis.plot(
        days,
        mean_7(daily["storage"]),
        label="Lagrad energi (7-dagars medel)",
        color="#D1495B",
        linewidth=1.5,
    )
    storage_axis.set_ylabel("Lagrad energi (TWh)")
    minimum, maximum = axis.get_ylim()
    if minimum >= 0:
        axis.set_ylim(bottom=-maximum * 0.05)
    minimum, maximum = axis.get_ylim()
    zero_position = -minimum / (maximum - minimum)
    storage_upper = max(max(daily["storage"], default=0.0) * 1.05, 0.001)
    storage_axis.set_ylim(
        -zero_position * storage_upper / (1 - zero_position),
        storage_upper,
    )

    lines, labels = axis.get_legend_handles_labels()
    storage_lines, storage_labels = storage_axis.get_legend_handles_labels()
    axis.legend(lines + storage_lines, labels + storage_labels, loc="upper left")
    figure.suptitle(scenario.name + " — Marrakesh, representativt år", y=0.97)
    figure.text(
        0.5,
        0.90,
        f"Toppeffekt: sol {solar_peak:,.0f} MW ({solar_capacity_kwp:,.0f} kWp)"
        f"  |  vindproxy {wind_peak:,.0f} MW"
        f"  |  total {total_peak:,.0f} MW\n"
        f"Lagringskrav {scenario.storage_capacity_mwh / 1_000_000:.3f} TWh"
        f"  |  max laddning/urladdning "
        f"{scenario.charge_power_mw:,.0f}/{scenario.discharge_power_mw:,.0f} MW"
        f"  |  verkningsgrad {MARRAKESH_STORAGE_ROUND_TRIP_EFFICIENCY:.0%}",
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_marrakesh_scenarios(data_directory: Path) -> list[MarrakeshScenario]:
    data = load_marrakesh_representative_year(data_directory)
    solar_only_data = replace(
        data,
        wind_farm_production_mwh=[0.0] * len(data.timestamps),
        wind_farm_scale_factor=0.0,
    )
    solar_only = calculate_solar_storage_requirement(
        solar_only_data,
        municipality_share=1.0,
        round_trip_efficiency=MARRAKESH_STORAGE_ROUND_TRIP_EFFICIENCY,
    )
    wind_only = calculate_wind_only_storage_requirement(
        data,
        municipality_share=1.0,
        round_trip_efficiency=MARRAKESH_STORAGE_ROUND_TRIP_EFFICIENCY,
    )
    hybrid = _estimate_hybrid(data)
    return [
        _scenario_from_solar_estimate("Solpark, skalad för årsbalans", solar_only),
        _scenario_from_wind_estimate(wind_only),
        _scenario_from_solar_estimate("Sol + vindproxy, optimerat lager", hybrid),
    ]


def main() -> None:
    project_directory = Path(__file__).resolve().parent
    marrakesh_directory = project_directory / "data" / "marrakesh"
    output_directory = marrakesh_directory / "grafer"
    scenarios = create_marrakesh_scenarios(marrakesh_directory)
    for index, scenario in enumerate(scenarios, start=1):
        output_path = output_directory / f"marrakesh_scenario_{index}.png"
        _plot_scenario(scenario, output_path)
        print(
            f"{scenario.name}: solfaktor={scenario.solar_factor:.3f} "
            f"({scenario.solar_factor * 1000:.0f} kWp), "
            f"vindfaktor={scenario.wind_factor:.3f}, "
            f"lager={scenario.storage_capacity_mwh / 1_000_000:.3f} TWh; "
            f"graf: {output_path}"
        )


if __name__ == "__main__":
    main()
