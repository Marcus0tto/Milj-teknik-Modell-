import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from io import StringIO
from math import cos, pi, sqrt
from pathlib import Path
import matplotlib.pyplot as plt


STORAGE_ROUND_TRIP_EFFICIENCY = 0.85


@dataclass(frozen=True)
class SolarStorageEstimate:
    timestamps: list[datetime]
    storage_timestamps: list[datetime]
    consumption_mwh: list[float]
    solar_production_mwh: list[float]
    wind_farm_production_mwh: list[float]
    wind_farm_scale_factor: float
    hydropower_production_mwh: list[float]
    surplus_deficit_mwh: list[float]
    storage_state_mwh: list[float]
    municipality_share: float
    solar_source_year: int
    consumption_source_years: list[int]
    wind_profile_source_years: list[int]
    hydropower_source_year: int | None
    hydropower_power_limit_mw: float
    round_trip_efficiency: float
    solar_scale_factor: float
    storage_capacity_mwh: float
    maximum_solar_power_mw: float
    maximum_generation_power_mw: float
    maximum_charge_power_mw: float
    maximum_discharge_power_mw: float


@dataclass(frozen=True)
class WindOnlyStorageEstimate:
    timestamps: list[datetime]
    consumption_mwh: list[float]
    wind_production_mwh: list[float]
    surplus_deficit_mwh: list[float]
    storage_state_mwh: list[float]
    wind_scale_factor: float
    round_trip_efficiency: float
    storage_capacity_mwh: float
    maximum_charge_power_mw: float
    maximum_discharge_power_mw: float


@dataclass(frozen=True)
class RepresentativeYearData:
    timestamps: list[datetime]
    consumption_mwh: list[float]
    solar_production_mwh: list[float]
    wind_farm_production_mwh: list[float]
    wind_farm_scale_factor: float
    hydropower_production_mwh: list[float]
    solar_source_year: int
    consumption_source_years: list[int]
    wind_profile_source_years: list[int]
    hydropower_source_year: int | None
    hydropower_power_limit_mw: float


@dataclass(frozen=True)
class HydropowerProfile:
    source_year: int
    timestamps: list[datetime]
    production_mwh: list[float]
    scale_factor: float
    target_annual_energy_gwh: float
    power_limit_mw: float


@dataclass(frozen=True)
class WindFarmProfile:
    timestamps: list[datetime]
    wind_speeds_m_s: list[float]
    estimated_production_mwh: list[float]
    annual_years: list[int]
    annual_mean_speeds_m_s: list[float]
    annual_production_gwh: list[float]
    profile_years: list[int]
    mean_speed_by_hour: dict[int, list[float]]
    mean_production_by_hour_mw: dict[int, list[float]]
    production_scale: float
    target_annual_energy_gwh: float



def load_data(
    csv_path: str | Path,
) -> dict[str, dict[str, list[tuple[int, int | None]]]]:
    data: dict[str, dict[str, list[tuple[int, int | None]]]] = {}

    with open(csv_path, encoding="cp1252", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for line_number, fields in enumerate(reader, start=2):
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(
                    f"Expected 4 columns on line {line_number}, got {len(fields)}"
                )

            production_type, year_text, table_content, energy_text = (
                field.strip() for field in fields
            )
            try:
                year = int(year_text)
                energy = None if energy_text in {"", ".."} else int(energy_text)
            except ValueError as error:
                raise ValueError(f"Invalid year or energy value on line {line_number}") from error

            data.setdefault(production_type, {}).setdefault(table_content, []).append(
                (year, energy)
            )

    return data


def plot_data(data: dict[str, dict[str, list[tuple[int, int | None]]]]) -> None:
    columns = 3
    rows = (len(data) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        sharex=True,
        squeeze=False,
    )
    series_colors = {
        "Brutto": "#2457A6",
        "Netto": "#E07A28",
        "Egenanvändning": "#398C6D",
    }

    for axis, (production_type, table_contents) in zip(axes.flat, data.items()):
        for table_content, color in series_colors.items():
            points = sorted(table_contents.get(table_content, []))
            years = [year for year, _ in points]
            energy_values = [value for _, value in points]
            axis.plot(years, energy_values, label=table_content, color=color)

        axis.set_title(production_type, fontsize=10)
        axis.grid(True, alpha=0.3)

    for axis in axes.flat[len(data):]:
        axis.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.suptitle("Elproduktion per kraftslag")
    figure.supxlabel("År")
    figure.supylabel("Energi (GWh)")
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    plt.show()


def load_municipal_consumption_share(
    csv_path: str | Path,
    municipality: str = "1480 Göteborg",
) -> tuple[list[int], list[float]]:
    csv_bytes = Path(csv_path).read_bytes()
    try:
        content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = csv_bytes.decode("cp1252")

    reader = csv.DictReader(StringIO(content, newline=""))
    required_columns = {
        "region",
        "förbrukarkategori",
        "bränsletyp",
        "år",
        "Slutanvändning (MWh)",
    }
    if reader.fieldnames is None or not required_columns.issubset(
        {name.strip() for name in reader.fieldnames}
    ):
        raise ValueError("CSV is missing required consumption columns")

    municipality_values: dict[int, int] = {}
    national_values: dict[int, int] = {}
    for row_number, row in enumerate(reader, start=2):
        if (
            row["förbrukarkategori"].strip() != "totalt"
            or row["bränsletyp"].strip() != "totalt"
        ):
            continue

        region = row["region"].strip()
        if region not in {municipality, "00 Riket"}:
            continue
        value_text = row["Slutanvändning (MWh)"].strip()
        if value_text in {"", ".."}:
            continue

        try:
            year = int(row["år"])
            value = int(value_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid year or consumption value on line {row_number}"
            ) from error

        target = municipality_values if region == municipality else national_values
        if year in target:
            raise ValueError(f"Duplicate total consumption row for {region}, {year}")
        target[year] = value

    years = sorted(municipality_values.keys() & national_values.keys())
    if not years:
        raise ValueError(
            f"No years with reported total consumption for {municipality} and Sweden"
        )
    if any(national_values[year] <= 0 for year in years):
        raise ValueError("National consumption totals must be positive")

    shares = [
        municipality_values[year] / national_values[year] * 100
        for year in years
    ]
    return years, shares


def plot_municipal_consumption_share(years: list[int], shares: list[float]) -> None:
    if not years or len(years) != len(shares):
        raise ValueError("Years and consumption shares must have the same non-empty data")

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(years, shares, marker="o", color="#2457A6")
    axis.annotate(
        f"{shares[-1]:.2f}%",
        (years[-1], shares[-1]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    axis.set_title("Göteborgs andel av Sveriges totala energianvändning")
    axis.set_xlabel("År")
    axis.set_ylabel("Göteborgs andel (%)")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()


def load_estimated_gothenburg_daily_electricity(
    data: RepresentativeYearData,
    *,
    share: float = 0.044,
) -> tuple[list[datetime], list[float]]:
    if not 0 < share < 1:
        raise ValueError("The estimated consumption share must be between 0 and 1")

    daily_totals: dict[datetime, float] = {}
    for timestamp, consumption in zip(data.timestamps, data.consumption_mwh):
        day = datetime.combine(timestamp.date(), datetime.min.time())
        daily_totals[day] = daily_totals.get(day, 0.0) + consumption * share

    if not daily_totals:
        raise ValueError("The CSV file contains no hourly consumption data")

    days = sorted(daily_totals)
    estimated_daily_consumption = [daily_totals[day] for day in days]
    return days, estimated_daily_consumption


def plot_estimated_gothenburg_daily_electricity(
    days: list[datetime],
    consumption: list[float],
    *,
    share: float = 0.044,
) -> None:
    if not days or len(days) != len(consumption):
        raise ValueError("Days and consumption must contain the same non-empty data")

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(days, consumption, color="#2457A6", linewidth=1.2)
    axis.set_title(
        f"Uppskattad daglig elförbrukning i Göteborg "
        f"({share * 100:.1f}% av Sveriges förbrukning)"
    )
    axis.set_xlabel("Datum")
    axis.set_ylabel("Uppskattad elförbrukning (MWh per dag)")
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    plt.show()


def load_estimated_gothenburg_daily_solar_and_consumption(
    data: RepresentativeYearData,
    *,
    share: float = 0.044,
    solar_scale_factor: float = 1.0,
) -> tuple[list[datetime], list[float], list[float], list[float]]:
    if solar_scale_factor <= 0:
        raise ValueError("The solar production scale factor must be positive")

    days, consumption = load_estimated_gothenburg_daily_electricity(
        data,
        share=share,
    )
    daily_solar: dict[datetime, float] = {}
    for timestamp, solar in zip(data.timestamps, data.solar_production_mwh):
        day = datetime.combine(timestamp.date(), datetime.min.time())
        daily_solar[day] = daily_solar.get(day, 0.0) + solar * share * solar_scale_factor

    if set(days) != set(daily_solar):
        raise ValueError("Solar and consumption data do not cover the same dates")

    solar_production = [daily_solar[day] for day in days]
    production_minus_consumption = [
        solar - used for solar, used in zip(solar_production, consumption)
    ]
    return days, consumption, solar_production, production_minus_consumption


def plot_estimated_gothenburg_daily_solar_balance(
    days: list[datetime],
    consumption: list[float],
    solar_production: list[float],
    production_minus_consumption: list[float],
    *,
    share: float = 0.044,
    solar_scale_factor: float = 1.0,
) -> None:
    if (
        not days
        or len(consumption) != len(days)
        or len(solar_production) != len(days)
        or len(production_minus_consumption) != len(days)
    ):
        raise ValueError("All daily series must have the same non-empty number of values")

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        days,
        consumption,
        label="Uppskattad elförbrukning i Göteborg",
        color="#2457A6",
        linewidth=1.2,
    )
    axis.plot(
        days,
        solar_production,
        label="Uppskattad solelproduktion i Göteborg",
        color="#E07A28",
        linewidth=1.2,
    )
    axis.plot(
        days,
        production_minus_consumption,
        label="Solelproduktion − elförbrukning",
        color="#398C6D",
        linewidth=1.2,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        "Kalibrerad solelproduktion, uppskattad elförbrukning och skillnad i Göteborg "
        f"(solfaktor {solar_scale_factor:.4f}×)"
    )
    axis.set_xlabel("Datum")
    axis.set_ylabel("Uppskattad energi (MWh per dag)")
    axis.text(
        0.01,
        0.02,
        f"Solel baserad på {share * 100:.1f}% av Sveriges produktion, "
        f"skalad {solar_scale_factor:.4f}× för att balansera periodens totalsumma.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    plt.show()


def load_representative_year_data(
    consumption_csv_path: str | Path,
    solar_csv_path: str | Path,
    wind_farm_profile: WindFarmProfile,
    hydropower_profile: HydropowerProfile | None = None,
) -> RepresentativeYearData:
    national_consumption = _load_national_energy_series(consumption_csv_path)
    national_solar = _load_national_energy_series(solar_csv_path)

    def complete_years(series: dict[datetime, float]) -> list[int]:
        years = sorted({timestamp.year for timestamp in series})
        complete: list[int] = []
        for year in years:
            year_timestamps = sorted(
                timestamp for timestamp in series if timestamp.year == year
            )
            expected_hours = (
                8784
                if datetime(year, 12, 31).timetuple().tm_yday == 366
                else 8760
            )
            if (
                len(year_timestamps) == expected_hours
                and year_timestamps[0] == datetime(year, 1, 1)
                and year_timestamps[-1] == datetime(year, 12, 31, 23)
                and all(
                    current - previous == timedelta(hours=1)
                    for previous, current in zip(
                        year_timestamps, year_timestamps[1:]
                    )
                )
            ):
                complete.append(year)
        return complete

    consumption_years = complete_years(national_consumption)
    solar_years = complete_years(national_solar)
    if not consumption_years:
        raise ValueError("The national consumption file has no complete years")
    if not solar_years:
        raise ValueError("The national solar file has no complete years")
    solar_year = solar_years[-1]

    def non_leap_key(timestamp: datetime) -> tuple[int, int, int] | None:
        if timestamp.month == 2 and timestamp.day == 29:
            return None
        return timestamp.month, timestamp.day, timestamp.hour

    consumption_samples: dict[tuple[int, int, int], list[float]] = {}
    for timestamp, value in national_consumption.items():
        if timestamp.year not in consumption_years:
            continue
        key = non_leap_key(timestamp)
        if key is not None:
            consumption_samples.setdefault(key, []).append(abs(value))

    solar_profile = {
        key: value
        for timestamp, value in national_solar.items()
        if timestamp.year == solar_year
        and (key := non_leap_key(timestamp)) is not None
    }

    wind_years = sorted(set(wind_farm_profile.annual_years))
    if not wind_years:
        raise ValueError("The SMHI wind-farm profile contains no complete years")
    wind_samples: dict[tuple[int, int, int], list[float]] = {}
    for timestamp, speed in zip(
        wind_farm_profile.timestamps,
        wind_farm_profile.wind_speeds_m_s,
    ):
        if timestamp.year not in wind_years:
            continue
        key = non_leap_key(timestamp)
        if key is not None:
            wind_samples.setdefault(key, []).append(
                wind_farm_profile.production_scale * speed**3
            )

    calendar_hours = [
        datetime(2023, 1, 1) + timedelta(hours=hour_index)
        for hour_index in range(365 * 24)
    ]
    expected_keys = [non_leap_key(timestamp) for timestamp in calendar_hours]
    if any(key is None for key in expected_keys):
        raise ValueError("Failed to construct a non-leap representative year")
    missing_consumption = sorted(
        {key for key in expected_keys if key not in consumption_samples}
    )
    missing_solar = sorted(
        {key for key in expected_keys if key not in solar_profile}
    )
    missing_wind = sorted(
        {key for key in expected_keys if key not in wind_samples}
    )
    if missing_consumption or missing_solar or missing_wind:
        raise ValueError(
            "Could not build a complete 365-day profile; missing hourly data "
            f"(consumption={len(missing_consumption)}, solar={len(missing_solar)}, "
            f"wind farm={len(missing_wind)})"
        )

    representative_hydropower = [0.0] * len(calendar_hours)
    if hydropower_profile is not None:
        if (
            len(hydropower_profile.timestamps) != len(calendar_hours)
            or len(hydropower_profile.production_mwh) != len(calendar_hours)
        ):
            raise ValueError(
                "The Lilla Edet hydropower profile must contain exactly 8760 hours"
            )
        hydro_by_month_day_hour = {
            (timestamp.month, timestamp.day, timestamp.hour): production
            for timestamp, production in zip(
                hydropower_profile.timestamps,
                hydropower_profile.production_mwh,
            )
        }
        representative_hydropower = [
            hydro_by_month_day_hour[
                (timestamp.month, timestamp.day, timestamp.hour)
            ]
            for timestamp in calendar_hours
        ]

    return RepresentativeYearData(
        timestamps=calendar_hours,
        consumption_mwh=[
            sum(consumption_samples[key]) / len(consumption_samples[key])
            for key in expected_keys
            if key is not None
        ],
        solar_production_mwh=[solar_profile[key] for key in expected_keys if key is not None],
        wind_farm_production_mwh=[
            sum(wind_samples[key]) / len(wind_samples[key])
            for key in expected_keys
            if key is not None
        ],
        wind_farm_scale_factor=1.0,
        hydropower_production_mwh=representative_hydropower,
        solar_source_year=solar_year,
        consumption_source_years=consumption_years,
        wind_profile_source_years=wind_years,
        hydropower_source_year=(
            hydropower_profile.source_year
            if hydropower_profile is not None
            else None
        ),
        hydropower_power_limit_mw=(
            hydropower_profile.power_limit_mw
            if hydropower_profile is not None
            else 36.0
        ),
    )


def calculate_solar_storage_requirement(
    data: RepresentativeYearData,
    *,
    municipality_share: float = 0.044,
    round_trip_efficiency: float = STORAGE_ROUND_TRIP_EFFICIENCY,
    include_hydropower: bool = False,
) -> SolarStorageEstimate:
    if not 0 < municipality_share <= 1:
        raise ValueError("The estimated municipality share must be greater than 0 and at most 1")
    if not 0 < round_trip_efficiency <= 1:
        raise ValueError("Round-trip storage efficiency must be greater than 0 and at most 1")
    if data.wind_farm_scale_factor < 0:
        raise ValueError("Wind-farm scale factor must be non-negative")
    series = (
        data.consumption_mwh,
        data.solar_production_mwh,
        data.wind_farm_production_mwh,
        data.hydropower_production_mwh,
    )
    if not data.timestamps or any(
        len(values) != len(data.timestamps) for values in series
    ):
        raise ValueError("All representative-year series must match the timestamp count")
    if len(data.timestamps) != 365 * 24:
        raise ValueError("The storage model requires exactly 365 days of hourly data")
    if any(
        current - previous != timedelta(hours=1)
        for previous, current in zip(data.timestamps, data.timestamps[1:])
    ):
        raise ValueError("Storage calculation requires continuous hourly data")
    if any(value < 0 for value in data.solar_production_mwh):
        raise ValueError("Solar production must be non-negative")
    if any(value < 0 for value in data.wind_farm_production_mwh):
        raise ValueError("Wind-farm production must be non-negative")
    if any(value < 0 for value in data.hydropower_production_mwh):
        raise ValueError("Hydropower production must be non-negative")

    timestamps = data.timestamps
    consumption = [value * municipality_share for value in data.consumption_mwh]
    baseline_solar = [
        value * municipality_share for value in data.solar_production_mwh
    ]
    if sum(baseline_solar) <= 0:
        raise ValueError("Solar data must contain positive production")

    wind_farm_production = data.wind_farm_production_mwh
    hydropower_production = (
        data.hydropower_production_mwh
        if include_hydropower
        else [0.0] * len(timestamps)
    )
    one_way_efficiency = sqrt(round_trip_efficiency)

    def storage_energy_change(scale_factor: float) -> list[float]:
        changes: list[float] = []
        for load, solar, wind_farm, hydropower in zip(
            consumption,
            baseline_solar,
            wind_farm_production,
            hydropower_production,
        ):
            surplus = scale_factor * solar + wind_farm + hydropower - load
            if surplus >= 0:
                changes.append(surplus * one_way_efficiency)
            else:
                changes.append(surplus / one_way_efficiency)
        return changes

    def net_stored_energy(scale_factor: float) -> float:
        return sum(storage_energy_change(scale_factor))

    lower_factor = 0.0
    upper_factor = 1.0
    while net_stored_energy(upper_factor) < 0:
        upper_factor *= 2
        if upper_factor > 1e12:
            raise ValueError("Could not find a solar scale factor that balances the period")

    for _ in range(80):
        midpoint = (lower_factor + upper_factor) / 2
        if net_stored_energy(midpoint) < 0:
            lower_factor = midpoint
        else:
            upper_factor = midpoint
    solar_scale_factor = (lower_factor + upper_factor) / 2

    solar_production = [
        solar * solar_scale_factor for solar in baseline_solar
    ]
    surplus_deficit = [
        solar + wind_farm + hydropower - load
        for solar, wind_farm, hydropower, load in zip(
            solar_production,
            wind_farm_production,
            hydropower_production,
            consumption,
        )
    ]
    storage_changes = storage_energy_change(solar_scale_factor)
    cumulative_storage = [0.0]
    for change in storage_changes:
        cumulative_storage.append(cumulative_storage[-1] + change)

    minimum_storage_level = min(cumulative_storage)
    storage_capacity = max(cumulative_storage) - minimum_storage_level
    initial_storage_level = -minimum_storage_level
    storage_state = [
        initial_storage_level + level for level in cumulative_storage
    ]
    storage_timestamps = [
        timestamps[0],
        *(timestamp + timedelta(hours=1) for timestamp in timestamps),
    ]
    total_generation = [
        solar + wind_farm + hydropower
        for solar, wind_farm, hydropower in zip(
            solar_production,
            wind_farm_production,
            hydropower_production,
        )
    ]

    return SolarStorageEstimate(
        timestamps=timestamps,
        storage_timestamps=storage_timestamps,
        consumption_mwh=consumption,
        solar_production_mwh=solar_production,
        wind_farm_production_mwh=wind_farm_production,
        wind_farm_scale_factor=data.wind_farm_scale_factor,
        hydropower_production_mwh=hydropower_production,
        surplus_deficit_mwh=surplus_deficit,
        storage_state_mwh=storage_state,
        municipality_share=municipality_share,
        solar_source_year=data.solar_source_year,
        consumption_source_years=data.consumption_source_years,
        wind_profile_source_years=data.wind_profile_source_years,
        hydropower_source_year=(
            data.hydropower_source_year if include_hydropower else None
        ),
        hydropower_power_limit_mw=data.hydropower_power_limit_mw,
        round_trip_efficiency=round_trip_efficiency,
        solar_scale_factor=solar_scale_factor,
        storage_capacity_mwh=storage_capacity,
        maximum_solar_power_mw=max(solar_production),
        maximum_generation_power_mw=max(total_generation),
        maximum_charge_power_mw=max(max(surplus, 0.0) for surplus in surplus_deficit),
        maximum_discharge_power_mw=max(max(-surplus, 0.0) for surplus in surplus_deficit),
    )


def calculate_wind_only_storage_requirement(
    data: RepresentativeYearData,
    *,
    municipality_share: float = 0.044,
    round_trip_efficiency: float = STORAGE_ROUND_TRIP_EFFICIENCY,
) -> WindOnlyStorageEstimate:
    if not 0 < municipality_share <= 1:
        raise ValueError("The estimated municipality share must be greater than 0 and at most 1")
    if not 0 < round_trip_efficiency <= 1:
        raise ValueError(
            "Round-trip storage efficiency must be greater than 0 and at most 1"
        )
    if (
        len(data.timestamps) != 365 * 24
        or len(data.consumption_mwh) != len(data.timestamps)
        or len(data.wind_farm_production_mwh) != len(data.timestamps)
    ):
        raise ValueError("Wind-only storage calculation requires 8760 hourly values")
    if any(
        current - previous != timedelta(hours=1)
        for previous, current in zip(data.timestamps, data.timestamps[1:])
    ):
        raise ValueError("Wind-only storage calculation requires continuous hourly data")
    if any(value < 0 for value in data.wind_farm_production_mwh):
        raise ValueError("Wind production must be non-negative")

    consumption = [
        value * municipality_share for value in data.consumption_mwh
    ]
    if sum(data.wind_farm_production_mwh) <= 0:
        raise ValueError("Wind data must contain positive production")

    one_way_efficiency = sqrt(round_trip_efficiency)

    def storage_changes(scale_factor: float) -> list[float]:
        changes: list[float] = []
        for wind, load in zip(data.wind_farm_production_mwh, consumption):
            surplus = wind * scale_factor - load
            changes.append(
                surplus * one_way_efficiency
                if surplus >= 0
                else surplus / one_way_efficiency
            )
        return changes

    lower_factor = 0.0
    upper_factor = 1.0
    while sum(storage_changes(upper_factor)) < 0:
        upper_factor *= 2
        if upper_factor > 1e12:
            raise ValueError("Could not find a wind scale factor that balances the year")

    for _ in range(80):
        midpoint = (lower_factor + upper_factor) / 2
        if sum(storage_changes(midpoint)) < 0:
            lower_factor = midpoint
        else:
            upper_factor = midpoint
    wind_scale_factor = (lower_factor + upper_factor) / 2

    wind_production = [
        value * wind_scale_factor
        for value in data.wind_farm_production_mwh
    ]
    surplus_deficit = [
        wind - load for wind, load in zip(wind_production, consumption)
    ]
    cumulative_storage = [0.0]
    for change in storage_changes(wind_scale_factor):
        cumulative_storage.append(cumulative_storage[-1] + change)

    minimum_storage = min(cumulative_storage)
    storage_capacity = max(cumulative_storage) - minimum_storage
    initial_storage = -minimum_storage
    storage_state = [
        initial_storage + level for level in cumulative_storage
    ]
    return WindOnlyStorageEstimate(
        timestamps=data.timestamps,
        consumption_mwh=consumption,
        wind_production_mwh=wind_production,
        surplus_deficit_mwh=surplus_deficit,
        storage_state_mwh=storage_state,
        wind_scale_factor=wind_scale_factor,
        round_trip_efficiency=round_trip_efficiency,
        storage_capacity_mwh=storage_capacity,
        maximum_charge_power_mw=max(max(value, 0.0) for value in surplus_deficit),
        maximum_discharge_power_mw=max(max(-value, 0.0) for value in surplus_deficit),
    )


def optimize_wind_farm_scale_for_storage(
    data: RepresentativeYearData,
    *,
    maximum_scale_factor: float = 12.0,
    grid_step: float = 0.25,
    refinement_iterations: int = 24,
    round_trip_efficiency: float = STORAGE_ROUND_TRIP_EFFICIENCY,
) -> tuple[float, SolarStorageEstimate]:
    if maximum_scale_factor <= 0 or grid_step <= 0:
        raise ValueError("Optimization bounds and grid step must be positive")
    if refinement_iterations < 1:
        raise ValueError("At least one refinement iteration is required")

    def estimate_at(scale_factor: float) -> SolarStorageEstimate:
        scaled_data = replace(
            data,
            wind_farm_production_mwh=[
                value * scale_factor
                for value in data.wind_farm_production_mwh
            ],
            wind_farm_scale_factor=scale_factor,
        )
        return calculate_solar_storage_requirement(
            scaled_data,
            round_trip_efficiency=round_trip_efficiency,
        )

    grid_factors = [
        min(index * grid_step, maximum_scale_factor)
        for index in range(int(maximum_scale_factor / grid_step) + 1)
    ]
    if grid_factors[-1] < maximum_scale_factor:
        grid_factors.append(maximum_scale_factor)
    grid_estimates = [
        estimate_at(scale_factor) for scale_factor in grid_factors
    ]
    best_index = min(
        range(len(grid_estimates)),
        key=lambda index: grid_estimates[index].storage_capacity_mwh,
    )
    if best_index == 0 or best_index == len(grid_estimates) - 1:
        raise ValueError(
            "Minimum storage lies outside the wind-farm optimization range; "
            "adjust maximum_scale_factor"
        )

    lower = grid_factors[best_index - 1]
    upper = grid_factors[best_index + 1]
    golden_ratio = (sqrt(5) - 1) / 2
    left_candidate = upper - golden_ratio * (upper - lower)
    right_candidate = lower + golden_ratio * (upper - lower)
    left_estimate = estimate_at(left_candidate)
    right_estimate = estimate_at(right_candidate)

    for _ in range(refinement_iterations):
        if left_estimate.storage_capacity_mwh < right_estimate.storage_capacity_mwh:
            upper = right_candidate
            right_candidate, right_estimate = left_candidate, left_estimate
            left_candidate = upper - golden_ratio * (upper - lower)
            left_estimate = estimate_at(left_candidate)
        else:
            lower = left_candidate
            left_candidate, left_estimate = right_candidate, right_estimate
            right_candidate = lower + golden_ratio * (upper - lower)
            right_estimate = estimate_at(right_candidate)

    optimal_scale_factor = (lower + upper) / 2
    optimal_estimate = estimate_at(optimal_scale_factor)
    if optimal_estimate.solar_scale_factor <= 1e-6:
        raise ValueError(
            "The optimized solution requires no solar production; "
            "the requested solar-and-wind balance is not feasible"
        )
    return optimal_scale_factor, optimal_estimate


def plot_solar_storage_requirement(estimate: SolarStorageEstimate) -> None:
    if (
        not estimate.timestamps
        or len(estimate.consumption_mwh) != len(estimate.timestamps)
        or len(estimate.solar_production_mwh) != len(estimate.timestamps)
        or len(estimate.wind_farm_production_mwh) != len(estimate.timestamps)
        or len(estimate.hydropower_production_mwh) != len(estimate.timestamps)
        or len(estimate.surplus_deficit_mwh) != len(estimate.timestamps)
        or len(estimate.storage_state_mwh) != len(estimate.storage_timestamps)
    ):
        raise ValueError("Production, consumption, and storage series have inconsistent lengths")

    daily_consumption: dict[datetime, float] = {}
    daily_solar: dict[datetime, float] = {}
    daily_wind_farm: dict[datetime, float] = {}
    daily_hydropower: dict[datetime, float] = {}
    daily_balance: dict[datetime, float] = {}
    daily_storage: dict[datetime, float] = {}
    for index, timestamp in enumerate(estimate.timestamps):
        day = datetime.combine(timestamp.date(), datetime.min.time())
        daily_consumption[day] = (
            daily_consumption.get(day, 0.0) + estimate.consumption_mwh[index]
        )
        daily_solar[day] = (
            daily_solar.get(day, 0.0) + estimate.solar_production_mwh[index]
        )
        daily_wind_farm[day] = (
            daily_wind_farm.get(day, 0.0)
            + estimate.wind_farm_production_mwh[index]
        )
        daily_hydropower[day] = (
            daily_hydropower.get(day, 0.0)
            + estimate.hydropower_production_mwh[index]
        )
        daily_balance[day] = (
            daily_balance.get(day, 0.0) + estimate.surplus_deficit_mwh[index]
        )
        daily_storage[day] = estimate.storage_state_mwh[index + 1]

    days = sorted(daily_consumption)
    consumption_by_day = [daily_consumption[day] for day in days]
    solar_by_day = [daily_solar[day] for day in days]
    wind_farm_by_day = [daily_wind_farm[day] for day in days]
    hydropower_by_day = [daily_hydropower[day] for day in days]
    has_hydropower = any(hydropower_by_day)
    has_wind_farm = any(wind_farm_by_day)
    balance_by_day = [daily_balance[day] for day in days]
    storage_by_day_twh = [
        daily_storage[day] / 1_000_000 for day in days
    ]

    def rolling_average(values: list[float], window: int = 7) -> list[float]:
        return [
            sum(values[max(0, index - window + 1):index + 1])
            / len(values[max(0, index - window + 1):index + 1])
            for index in range(len(values))
        ]

    def set_month_ticks(axis, year: int) -> None:
        axis.set_xticks([datetime(year, month, 15) for month in range(1, 13)])
        axis.set_xticklabels(
            ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]
        )

    figure, axis = plt.subplots(figsize=(14, 7))
    axis.plot(
        days,
        rolling_average(solar_by_day),
        label="Solelproduktion (7-dagars medel)",
        color="#E07A28",
        linewidth=1.8,
    )
    if has_wind_farm:
        axis.plot(
            days,
            rolling_average(wind_farm_by_day),
            label="Illustrativ vindpark (7-dagars medel)",
            color="#7B2CBF",
            linewidth=1.8,
        )
    if has_hydropower:
        axis.plot(
            days,
            rolling_average(hydropower_by_day),
            label="Lilla Edets vattenkraft (7-dagars medel)",
            color="#2A9D8F",
            linewidth=1.8,
        )
    axis.plot(
        days,
        rolling_average(consumption_by_day),
        label="Elförbrukning (7-dagars medel)",
        color="#2457A6",
        linewidth=1.8,
    )
    axis.plot(
        days,
        rolling_average(balance_by_day),
        label="Överskott / underskott (7-dagars medel)",
        color="#398C6D",
        linewidth=1.8,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("Månad")
    axis.set_ylabel("Energi (MWh per dygn)")
    axis.grid(True, alpha=0.3)
    set_month_ticks(axis, days[0].year)

    storage_axis = axis.twinx()
    storage_axis.plot(
        days,
        rolling_average(storage_by_day_twh),
        label="Lagrad energi (7-dagars medel)",
        color="#7B2CBF",
        linewidth=1.4,
    )
    storage_axis.set_ylabel("Lagrad energi (TWh)")
    primary_minimum, primary_maximum = axis.get_ylim()
    if primary_minimum >= 0:
        axis.set_ylim(bottom=-primary_maximum * 0.05)
    elif primary_maximum <= 0:
        axis.set_ylim(top=-primary_minimum * 0.05)

    primary_minimum, primary_maximum = axis.get_ylim()
    zero_position = -primary_minimum / (primary_maximum - primary_minimum)
    maximum_storage = max(storage_by_day_twh, default=0.0)
    storage_upper_limit = max(maximum_storage * 1.05, 1.0)
    storage_axis.set_ylim(
        -zero_position * storage_upper_limit / (1 - zero_position),
        storage_upper_limit,
    )

    handles, labels = axis.get_legend_handles_labels()
    storage_handles, storage_labels = storage_axis.get_legend_handles_labels()
    axis.legend(handles + storage_handles, labels + storage_labels, loc="upper left")
    title = (
        f"{'Optimerat ' if estimate.wind_farm_scale_factor > 1 else ''}"
        "representativt år (365 dagar): solel, förbrukning, "
        f"{'vindpark, ' if has_wind_farm else ''}"
        f"{'Lilla Edets vattenkraft och ' if has_hydropower else ''}lagring "
        f"(solfaktor {estimate.solar_scale_factor:.2f}×)"
    )
    figure.suptitle(title, y=0.97, fontsize=14, fontweight="bold")
    max_wind_power = max(estimate.wind_farm_production_mwh, default=0.0)
    max_hydropower_power = max(estimate.hydropower_production_mwh, default=0.0)
    metric_line = (
        f"Max timmedeleffekt: sol {estimate.maximum_solar_power_mw:,.0f} MW"
        f"  |  vind {max_wind_power:,.0f} MW"
        f"  |  vattenkraft {max_hydropower_power:,.0f} MW"
        f"  |  total {estimate.maximum_generation_power_mw:,.0f} MW\n"
        f"Lagringskrav: {estimate.storage_capacity_mwh / 1_000_000:.2f} TWh"
    )
    figure.text(
        0.5,
        0.87,
        metric_line,
        fontsize=9,
        ha="center",
        va="center",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.78))
    plt.show()


def plot_wind_only_storage_requirement(
    estimate: WindOnlyStorageEstimate,
) -> None:
    if (
        len(estimate.timestamps) != 365 * 24
        or len(estimate.consumption_mwh) != len(estimate.timestamps)
        or len(estimate.wind_production_mwh) != len(estimate.timestamps)
        or len(estimate.storage_state_mwh) != len(estimate.timestamps) + 1
    ):
        raise ValueError("Wind-only production, consumption, and storage data do not align")

    def rolling_average(values: list[float], window: int = 7 * 24) -> list[float]:
        return [
            sum(values[max(0, index - window + 1):index + 1])
            / len(values[max(0, index - window + 1):index + 1])
            for index in range(len(values))
        ]

    figure, axis = plt.subplots(figsize=(15, 7))
    axis.plot(
        estimate.timestamps,
        rolling_average(estimate.wind_production_mwh),
        color="#168AAD",
        linewidth=1.5,
        label="Vindproduktion (7-dagars medel)",
    )
    axis.plot(
        estimate.timestamps,
        rolling_average(estimate.consumption_mwh),
        color="#2457A6",
        linewidth=1.5,
        label="Uppskattad förbrukning (7-dagars medel)",
    )
    axis.plot(
        estimate.timestamps,
        rolling_average(estimate.surplus_deficit_mwh),
        color="#398C6D",
        linewidth=1.5,
        label="Överskott / underskott (7-dagars medel)",
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Månad")
    axis.set_ylabel("Effekt (MWh per timme ≈ MW)")
    axis.grid(True, alpha=0.3)
    axis.set_xticks(
        [datetime(estimate.timestamps[0].year, month, 15) for month in range(1, 13)]
    )
    axis.set_xticklabels(
        ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]
    )

    storage_axis = axis.twinx()
    storage_timestamps = [
        estimate.timestamps[0],
        *(timestamp + timedelta(hours=1) for timestamp in estimate.timestamps),
    ]
    storage_axis.plot(
        storage_timestamps,
        [
            value / 1_000_000
            for value in rolling_average(estimate.storage_state_mwh)
        ],
        color="#E07A28",
        linewidth=1.3,
        label="Lagrad energi (7-dagars medel)",
    )
    storage_axis.set_ylabel("Lagrad energi (TWh)")

    primary_minimum, primary_maximum = axis.get_ylim()
    if primary_minimum >= 0:
        axis.set_ylim(bottom=-primary_maximum * 0.05)
    elif primary_maximum <= 0:
        axis.set_ylim(top=-primary_minimum * 0.05)

    primary_minimum, primary_maximum = axis.get_ylim()
    zero_position = -primary_minimum / (primary_maximum - primary_minimum)
    maximum_storage = max(estimate.storage_state_mwh, default=0.0) / 1_000_000
    storage_upper_limit = max(maximum_storage * 1.05, 1.0)
    storage_axis.set_ylim(
        -zero_position * storage_upper_limit / (1 - zero_position),
        storage_upper_limit,
    )

    handles, labels = axis.get_legend_handles_labels()
    storage_handles, storage_labels = storage_axis.get_legend_handles_labels()
    axis.legend(
        handles + storage_handles,
        labels + storage_labels,
        loc="upper left",
    )
    annual_wind_gwh = sum(estimate.wind_production_mwh) / 1000
    figure.suptitle(
        "Vind som enda produktionskälla: representativt år",
        y=0.97,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.87,
        f"Max timmedeleffekt — vind {max(estimate.wind_production_mwh):,.0f} MW"
        f"       Lagringskrav: {estimate.storage_capacity_mwh / 1_000_000:.2f} TWh"
        f"       Max laddning/urladdning: "
        f"{estimate.maximum_charge_power_mw:,.0f}/{estimate.maximum_discharge_power_mw:,.0f} MW"
        f"       Vind {annual_wind_gwh:,.0f} GWh/år",
        fontsize=9,
        ha="center",
        va="center",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.78))
    plt.show()


def load_hourly_data(
    csv_path: str | Path,
) -> tuple[list[datetime], dict[str, list[float | None]]]:
    csv_bytes = Path(csv_path).read_bytes()
    try:
        content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = csv_bytes.decode("cp1252")

    reader = csv.reader(StringIO(content, newline=""))
    for line_number, fields in enumerate(reader, start=1):
        if fields and fields[0].strip() == "Tid":
            headers = [field.strip() for field in fields]
            while headers and not headers[-1]:
                headers.pop()
            break
    else:
        raise ValueError("Could not find a 'Tid' header in the CSV file")

    if len(headers) < 2:
        raise ValueError("Expected a timestamp column and at least one data column")

    timestamps: list[datetime] = []
    series = {header: [] for header in headers[1:]}

    for row_number, fields in enumerate(reader, start=line_number + 1):
        if not fields or not any(field.strip() for field in fields):
            continue
        values = fields[:len(headers)]
        if len(values) < len(headers) or any(
            field.strip() for field in fields[len(headers):]
        ):
            raise ValueError(
                f"Expected {len(headers)} columns on line {row_number}, got {len(fields)}"
            )

        try:
            timestamp = datetime.strptime(values[0].strip(), "%d.%m.%Y %H:%M")
        except ValueError as error:
            raise ValueError(f"Invalid timestamp on line {row_number}") from error

        timestamps.append(timestamp)
        for header, value in zip(headers[1:], values[1:]):
            value = value.strip()
            try:
                parsed_value = None if value in {"", ".."} else float(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid value for {header!r} on line {row_number}"
                ) from error
            series[header].append(parsed_value)

    if not timestamps:
        raise ValueError("The CSV file contains no hourly data rows")

    return timestamps, series


def _load_national_energy_series(csv_path: str | Path) -> dict[datetime, float]:
    csv_bytes = Path(csv_path).read_bytes()
    try:
        content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = csv_bytes.decode("cp1252")

    reader = csv.DictReader(StringIO(content, newline=""), delimiter=";")
    required_columns = {"Period", "Avräknad (kWh)"}
    if reader.fieldnames is None:
        raise ValueError(f"{csv_path} does not contain a CSV header")
    headers = {name.strip() for name in reader.fieldnames if name is not None}
    if not required_columns.issubset(headers):
        raise ValueError(
            f"{csv_path} must contain Period and Avräknad (kWh) columns"
        )

    series: dict[datetime, float] = {}
    for row_number, row in enumerate(reader, start=2):
        period = (row.get("Period") or "").strip()
        if not period or period == "Summa":
            continue
        try:
            timestamp = datetime.strptime(period, "%Y-%m-%d %H:%M")
        except ValueError as error:
            raise ValueError(
                f"Invalid hourly timestamp in {csv_path} on line {row_number}"
            ) from error

        value_text = (row.get("Avräknad (kWh)") or "").strip()
        try:
            value_mwh = float(value_text.replace(",", ".")) / 1000
        except ValueError as error:
            raise ValueError(
                f"Invalid energy value in {csv_path} on line {row_number}"
            ) from error
        if timestamp in series:
            raise ValueError(f"Duplicate hourly timestamp {timestamp} in {csv_path}")
        series[timestamp] = value_mwh

    if not series:
        raise ValueError(f"{csv_path} contains no hourly energy observations")
    return series


def load_latest_complete_daily_production(
    csv_path: str | Path,
    *,
    target_annual_energy_gwh: float = 210.0,
    power_limit_mw: float = 36.0,
) -> HydropowerProfile:
    if target_annual_energy_gwh <= 0:
        raise ValueError("Target annual hydropower energy must be positive")
    if power_limit_mw <= 0:
        raise ValueError("Hydropower power limit must be positive")

    hourly_production = _load_national_energy_series(csv_path)
    years = sorted({timestamp.year for timestamp in hourly_production})
    complete_years: list[int] = []
    for year in years:
        observations = sorted(
            timestamp for timestamp in hourly_production if timestamp.year == year
        )
        expected_hours = (
            8784
            if datetime(year, 12, 31).timetuple().tm_yday == 366
            else 8760
        )
        if (
            len(observations) == expected_hours
            and observations[0] == datetime(year, 1, 1)
            and observations[-1] == datetime(year, 12, 31, 23)
            and all(
                current - previous == timedelta(hours=1)
                for previous, current in zip(observations, observations[1:])
            )
        ):
            complete_years.append(year)
    if not complete_years:
        raise ValueError(f"{csv_path} contains no complete calendar year")

    source_year = complete_years[-1]
    source_timestamps = sorted(
        timestamp
        for timestamp in hourly_production
        if timestamp.year == source_year
        and not (timestamp.month == 2 and timestamp.day == 29)
    )
    source_profile = [hourly_production[timestamp] for timestamp in source_timestamps]
    if any(value < 0 for value in source_profile):
        raise ValueError(f"Hydropower production cannot be negative in {csv_path}")

    source_energy_mwh = sum(source_profile)
    if source_energy_mwh <= 0:
        raise ValueError("Hydropower source year must have positive production")
    scale_factor = target_annual_energy_gwh * 1000 / source_energy_mwh
    production_mwh = [
        min(value * scale_factor, power_limit_mw)
        for value in source_profile
    ]
    timestamps = [
        datetime(2023, 1, 1) + timedelta(hours=hour_index)
        for hour_index in range(365 * 24)
    ]
    if len(source_timestamps) != len(timestamps):
        raise ValueError(
            f"Expected 8760 non-leap hourly values in {source_year}, "
            f"found {len(source_timestamps)}"
        )
    return HydropowerProfile(
        source_year=source_year,
        timestamps=timestamps,
        production_mwh=production_mwh,
        scale_factor=scale_factor,
        target_annual_energy_gwh=target_annual_energy_gwh,
        power_limit_mw=power_limit_mw,
    )


def plot_national_hydropower_profile(
    profile: HydropowerProfile,
) -> None:
    if (
        len(profile.timestamps) != 365 * 24
        or len(profile.production_mwh) != 365 * 24
    ):
        raise ValueError("The hydropower profile must contain exactly 8760 hours")
    if any(value < 0 for value in profile.production_mwh):
        raise ValueError("Hydropower production must be non-negative")

    daily_production: list[float] = []
    for day_start in range(0, len(profile.production_mwh), 24):
        daily_production.append(
            sum(profile.production_mwh[day_start:day_start + 24])
        )
    rolling_average = [
        sum(daily_production[max(0, index - 6):index + 1])
        / len(daily_production[max(0, index - 6):index + 1])
        for index in range(len(daily_production))
    ]
    days = [
        datetime(2023, 1, 1) + timedelta(days=day_index)
        for day_index in range(365)
    ]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        days,
        [value / 1000 for value in daily_production],
        label="Uppskattad vattenkraft i Lilla Edet per dygn",
        color="#168AAD",
        linewidth=0.8,
        alpha=0.35,
    )
    axis.plot(
        days,
        [value / 1000 for value in rolling_average],
        label="Uppskattad vattenkraft i Lilla Edet (7-dagars medel)",
        color="#2457A6",
        linewidth=1.8,
    )
    axis.set_title(
        f"Uppskattad vattenkraft i Lilla Edet ({profile.source_year}, 365 dagar)"
    )
    axis.set_xlabel("Kalenderdag")
    axis.set_ylabel("Elproduktion (GWh per dygn)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.text(
        0.01,
        0.015,
        f"Svensk vattenkrafts timprofil från {profile.source_year}, skalad "
        f"{profile.scale_factor:.6f}× mot 210 GWh och begränsad till "
        f"{profile.power_limit_mw:.0f} MW per timme. Efter effektbegränsningen "
        f"blir årsenergin {sum(profile.production_mwh) / 1000:.1f} GWh.",
        fontsize=8,
        va="bottom",
    )
    figure.autofmt_xdate()
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()


def load_se3_renewable_hourly_data(
    csv_path: str | Path,
) -> tuple[list[datetime], dict[str, list[float | None]]]:
    csv_bytes = Path(csv_path).read_bytes()
    try:
        content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = csv_bytes.decode("cp1252")

    rows = list(csv.reader(StringIO(content, newline="")))
    if len(rows) < 5:
        raise ValueError("CSV does not contain the expected multi-row header")

    category_row, detail_row, zone_row = rows[:3]
    selected_columns: dict[str, int] = {}
    column_specs = (
        ("Vindkraft landbaserad (SE3)", "Vindkraft", "landbaserad"),
        ("Vindkraft havsbaserad (SE3)", "Vindkraft", "havsbaserad"),
        ("Solkraft (SE3)", "Solkraft", "produktion"),
    )
    for label, category, detail in column_specs:
        matches = [
            index
            for index in range(1, min(len(category_row), len(detail_row), len(zone_row)))
            if category_row[index].strip() == category
            and detail_row[index].strip() == detail
            and zone_row[index].strip() == "SE3"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one SE3 column for {label}, found {len(matches)}"
            )
        selected_columns[label] = matches[0]

    timestamps: list[datetime] = []
    series = {label: [] for label, _, _ in column_specs}
    for row_number, fields in enumerate(rows[5:], start=6):
        if not fields or not fields[0].strip():
            continue
        try:
            timestamp = datetime.strptime(fields[0].strip(), "%d.%m.%Y %H:%M")
        except ValueError as error:
            raise ValueError(f"Invalid timestamp on line {row_number}") from error

        timestamps.append(timestamp)
        for label, column_index in selected_columns.items():
            if column_index >= len(fields):
                raise ValueError(
                    f"Missing {label} value on line {row_number}"
                )
            value = fields[column_index].strip()
            try:
                parsed_value = (
                    None if value in {"", ".."} else float(value.replace(",", ""))
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid value for {label!r} on line {row_number}"
                ) from error
            series[label].append(parsed_value)

    if not timestamps:
        raise ValueError("The CSV file contains no hourly data rows")

    renewable_sources = (
        "Vindkraft landbaserad (SE3)",
        "Vindkraft havsbaserad (SE3)",
        "Solkraft (SE3)",
    )
    total_renewable_production = [
        None if any(value is None for value in values)
        else sum(value for value in values if value is not None)
        for values in zip(*(series[name] for name in renewable_sources))
    ]
    chart_series = {
        "Vindkraft landbaserad (SE3)": series["Vindkraft landbaserad (SE3)"],
        "Förnybar elproduktion exkl. vattenkraft (SE3)": total_renewable_production,
        "Solkraft (SE3)": series["Solkraft (SE3)"],
    }

    return timestamps, chart_series


def plot_hourly_data(
    timestamps: list[datetime],
    series: dict[str, list[float | None]],
    *,
    title: str = "Svensk elproduktion per timme",
    unit: str = "GWh",
) -> None:
    if not timestamps:
        raise ValueError("Cannot plot hourly data without timestamps")
    if not series:
        raise ValueError("Cannot plot hourly data without any series")

    daily_values: dict[datetime.date, dict[str, list[float | None]]] = {}
    for name, values in series.items():
        if len(values) != len(timestamps):
            raise ValueError(f"Series {name!r} does not match the timestamp count")

    for timestamp in timestamps:
        daily_values.setdefault(timestamp.date(), {})
    for name, values in series.items():
        for timestamp, value in zip(timestamps, values):
            day_values = daily_values[timestamp.date()].setdefault(
                name, [None] * 24
            )
            day_values[timestamp.hour] = value

    column_count = 2
    row_count = (len(series) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, 3.6 * row_count),
        sharex=True,
        squeeze=False,
    )
    colormap = plt.get_cmap("RdBu_r")
    days = sorted(daily_values)

    for axis, name in zip(axes.flat, series):
        for day in days:
            year_length = 366 if datetime(day.year, 12, 31).timetuple().tm_yday == 366 else 365
            season_position = (
                1 - cos(2 * pi * (day.timetuple().tm_yday - 15) / year_length)
            ) / 2
            values = daily_values[day].get(name, [None] * 24)
            axis.plot(
                range(24),
                values,
                color=colormap(season_position),
                linewidth=0.8,
                alpha=0.75,
            )

        axis.set_title(name)
        axis.set_xlim(0, 23)
        axis.set_xticks([0, 6, 12, 18, 23])
        axis.grid(True, alpha=0.3)

    for axis in axes.flat[len(series):]:
        axis.set_visible(False)

    figure.suptitle(f"{title} ({unit})")
    figure.supxlabel("Timme på dagen (0-23)")
    figure.supylabel(unit)
    colorbar_source = plt.cm.ScalarMappable(cmap=colormap)
    colorbar_source.set_array([])
    figure.tight_layout(rect=(0, 0.04, 0.88, 0.96), h_pad=3.0)
    figure.subplots_adjust(hspace=0.7)
    colorbar_axis = figure.add_axes([0.92, 0.15, 0.02, 0.7])
    colorbar = figure.colorbar(
        colorbar_source,
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Vinter", "Sommar"])
    plt.show()


def load_smhi_wind_data(
    csv_path: str | Path,
    *,
    include_suspect: bool = False,
) -> tuple[list[float], list[float]]:
    csv_bytes = Path(csv_path).read_bytes()
    try:
        content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = csv_bytes.decode("cp1252")

    reader = csv.reader(StringIO(content, newline=""), delimiter=";")
    for line_number, fields in enumerate(reader, start=1):
        if fields and fields[0].strip() == "Datum":
            headers = [field.strip() for field in fields]
            break
    else:
        raise ValueError("Could not find the SMHI 'Datum' header")

    try:
        date_index = headers.index("Datum")
        direction_index = headers.index("Vindriktning")
        speed_index = headers.index("Vindhastighet")
    except ValueError as error:
        raise ValueError("CSV is missing a required wind data column") from error

    quality_indexes = [index for index, name in enumerate(headers) if name == "Kvalitet"]
    if len(quality_indexes) < 2:
        raise ValueError("CSV must contain quality columns for direction and speed")

    valid_qualities = {"G", "Y"} if include_suspect else {"G"}
    directions: list[float] = []
    speeds: list[float] = []
    required_index = max(
        date_index,
        direction_index,
        speed_index,
        quality_indexes[0],
        quality_indexes[1],
    )

    for row_number, fields in enumerate(reader, start=line_number + 1):
        if len(fields) <= required_index:
            continue
        try:
            datetime.strptime(fields[date_index].strip(), "%Y-%m-%d")
        except ValueError:
            continue

        if any(fields[index].strip() not in valid_qualities for index in quality_indexes[:2]):
            continue

        try:
            direction = float(fields[direction_index].strip())
            speed = float(fields[speed_index].strip())
        except ValueError:
            continue
        if 0 <= direction <= 360 and speed >= 0:
            directions.append(direction)
            speeds.append(speed)

    if not directions:
        raise ValueError("No valid wind direction and speed observations were found")

    return directions, speeds


def load_smhi_wind_speed_data(
    csv_path: str | Path,
    *,
    include_suspect: bool = False,
) -> tuple[list[datetime], list[float]]:
    csv_bytes = Path(csv_path).read_bytes()
    try:
        content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = csv_bytes.decode("cp1252")

    reader = csv.reader(StringIO(content, newline=""), delimiter=";")
    for fields in reader:
        if fields and fields[0].strip() == "Datum":
            headers = [field.strip() for field in fields]
            break
    else:
        raise ValueError("Could not find the SMHI 'Datum' header")

    try:
        date_index = headers.index("Datum")
        time_index = headers.index("Tid (UTC)")
        speed_index = headers.index("Vindhastighet")
        quality_indexes = [
            index for index, name in enumerate(headers) if name == "Kvalitet"
        ]
        speed_quality_index = quality_indexes[1]
    except (ValueError, IndexError) as error:
        raise ValueError("CSV is missing a required wind speed data column") from error

    valid_qualities = {"G", "Y"} if include_suspect else {"G"}
    required_index = max(date_index, time_index, speed_index, speed_quality_index)
    timestamps: list[datetime] = []
    speeds: list[float] = []

    for fields in reader:
        if len(fields) <= required_index:
            continue
        if fields[speed_quality_index].strip() not in valid_qualities:
            continue

        try:
            timestamp = datetime.fromisoformat(
                f"{fields[date_index].strip()}T{fields[time_index].strip()}"
            )
            speed = float(fields[speed_index].strip())
        except ValueError:
            continue
        if speed >= 0:
            timestamps.append(timestamp)
            speeds.append(speed)

    if not timestamps:
        raise ValueError("No valid wind speed observations were found")

    return timestamps, speeds


def plot_average_wind_speed_by_hour(
    timestamps: list[datetime],
    speeds: list[float],
) -> None:
    if not timestamps or len(timestamps) != len(speeds):
        raise ValueError("Timestamps and speeds must contain the same non-empty data")

    hourly_speeds: list[list[float]] = [[] for _ in range(24)]
    for timestamp, speed in zip(timestamps, speeds):
        if speed < 0:
            raise ValueError("Wind speeds must be non-negative")
        hourly_speeds[timestamp.hour].append(speed)

    hours = [hour for hour, values in enumerate(hourly_speeds) if values]
    average_speeds = [
        sum(hourly_speeds[hour]) / len(hourly_speeds[hour])
        for hour in hours
    ]
    standard_deviations = [
        sqrt(
            sum((speed - average) ** 2 for speed in hourly_speeds[hour])
            / len(hourly_speeds[hour])
        )
        for hour, average in zip(hours, average_speeds)
    ]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.errorbar(
        hours,
        average_speeds,
        yerr=standard_deviations,
        fmt="o-",
        capsize=3,
        color="#168AAD",
        label="Medelvärde ± 1 standardavvikelse",
    )
    axis.set_title("Genomsnittlig vindhastighet efter tid på dygnet")
    axis.set_xlabel("Tid på dygnet (UTC)")
    axis.set_ylabel("Genomsnittlig vindhastighet (m/s)")
    axis.set_xlim(0, 23)
    axis.set_xticks(range(0, 24, 3))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()


def estimate_wind_farm_profile(
    timestamps: list[datetime],
    wind_speeds: list[float],
    *,
    target_annual_energy_gwh: float = 330.0,
    calibration_years: tuple[int, int] = (2015, 2022),
) -> WindFarmProfile:
    if not timestamps or len(timestamps) != len(wind_speeds):
        raise ValueError("Timestamps and wind speeds must have the same non-empty length")
    if target_annual_energy_gwh <= 0:
        raise ValueError("Target annual energy must be positive")
    if calibration_years[0] > calibration_years[1]:
        raise ValueError("Calibration years must be in ascending order")

    by_year: dict[int, list[tuple[datetime, float]]] = {}
    for timestamp, speed in zip(timestamps, wind_speeds):
        if speed < 0:
            raise ValueError("Wind speeds must be non-negative")
        by_year.setdefault(timestamp.year, []).append((timestamp, speed))

    complete_years: dict[int, list[tuple[datetime, float]]] = {}
    annual_proxy_totals: list[float] = []
    for year, observations in by_year.items():
        year_hours = 8784 if datetime(year, 12, 31).timetuple().tm_yday == 366 else 8760
        if len(observations) < year_hours * 0.95:
            continue
        complete_years[year] = observations
        if calibration_years[0] <= year <= calibration_years[1]:
            mean_cubed_speed = sum(speed**3 for _, speed in observations) / len(observations)
            annual_proxy_totals.append(mean_cubed_speed * year_hours)

    expected_calibration_years = set(range(calibration_years[0], calibration_years[1] + 1))
    if not expected_calibration_years.issubset(complete_years):
        missing_years = sorted(expected_calibration_years - complete_years.keys())
        raise ValueError(
            f"Missing sufficiently complete SMHI data for calibration years {missing_years}"
        )
    if not annual_proxy_totals or sum(annual_proxy_totals) <= 0:
        raise ValueError("Calibration period must contain positive wind energy")

    mean_annual_proxy = sum(annual_proxy_totals) / len(annual_proxy_totals)
    production_scale = target_annual_energy_gwh * 1000 / mean_annual_proxy
    estimated_production = [
        production_scale * speed**3 for speed in wind_speeds
    ]

    annual_years = sorted(complete_years)
    annual_mean_speeds = [
        sum(speed for _, speed in complete_years[year]) / len(complete_years[year])
        for year in annual_years
    ]
    annual_production_gwh = [
        sum(production_scale * speed**3 for _, speed in complete_years[year]) / 1000
        * (
            8784
            if datetime(year, 12, 31).timetuple().tm_yday == 366
            else 8760
        )
        / len(complete_years[year])
        for year in annual_years
    ]

    profile_years = [
        year for year in annual_years
        if year >= calibration_years[0]
    ]
    mean_speed_by_hour: dict[int, list[float]] = {}
    mean_production_by_hour_mw: dict[int, list[float]] = {}
    for year in profile_years:
        hour_speeds: list[list[float]] = [[] for _ in range(24)]
        hour_production: list[list[float]] = [[] for _ in range(24)]
        for timestamp, speed in complete_years[year]:
            hour_speeds[timestamp.hour].append(speed)
            hour_production[timestamp.hour].append(production_scale * speed**3)
        if any(not speeds_for_hour for speeds_for_hour in hour_speeds):
            continue
        mean_speed_by_hour[year] = [
            sum(speeds_for_hour) / len(speeds_for_hour)
            for speeds_for_hour in hour_speeds
        ]
        mean_production_by_hour_mw[year] = [
            sum(outputs_for_hour) / len(outputs_for_hour)
            for outputs_for_hour in hour_production
        ]

    profile_years = sorted(mean_speed_by_hour)
    if not annual_years or not profile_years:
        raise ValueError("No sufficiently complete annual wind profiles were found")

    return WindFarmProfile(
        timestamps=timestamps,
        wind_speeds_m_s=wind_speeds,
        estimated_production_mwh=estimated_production,
        annual_years=annual_years,
        annual_mean_speeds_m_s=annual_mean_speeds,
        annual_production_gwh=annual_production_gwh,
        profile_years=profile_years,
        mean_speed_by_hour=mean_speed_by_hour,
        mean_production_by_hour_mw=mean_production_by_hour_mw,
        production_scale=production_scale,
        target_annual_energy_gwh=target_annual_energy_gwh,
    )


def plot_wind_farm_profile(profile: WindFarmProfile) -> None:
    figure, (hour_axis, annual_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
    )
    colormap = plt.get_cmap("RdYlBu_r")
    year_min = min(profile.profile_years)
    year_max = max(profile.profile_years)

    production_axis = hour_axis.twinx()
    for year in profile.profile_years:
        color_position = (
            (year - year_min) / (year_max - year_min)
            if year_max > year_min
            else 0.5
        )
        color = colormap(color_position)
        hour_axis.plot(
            range(24),
            profile.mean_speed_by_hour[year],
            color=color,
            alpha=0.75,
            linewidth=1.1,
            label=str(year),
        )
        production_axis.plot(
            range(24),
            profile.mean_production_by_hour_mw[year],
            color=color,
            alpha=0.5,
            linewidth=0.9,
            linestyle="--",
        )

    hour_axis.set_title("Dygnsprofil per år")
    hour_axis.set_xlabel("Timme (UTC)")
    hour_axis.set_ylabel("Medelvindhastighet (m/s)")
    production_axis.set_ylabel("Illustrativ parkeffekt (MWh per timme)")
    hour_axis.set_xlim(0, 23)
    hour_axis.set_xticks(range(0, 24, 3))
    hour_axis.grid(True, alpha=0.3)
    handles, labels = hour_axis.get_legend_handles_labels()
    hour_axis.legend(handles, labels, title="År", fontsize=8, ncol=2)

    annual_axis.plot(
        profile.annual_years,
        profile.annual_mean_speeds_m_s,
        marker="o",
        color="#2457A6",
        label="Årsmedelvind",
    )
    annual_axis.set_xlabel("År")
    annual_axis.set_ylabel("Årsmedelvind (m/s)", color="#2457A6")
    annual_axis.tick_params(axis="y", labelcolor="#2457A6")
    annual_axis.grid(True, alpha=0.3)
    annual_axis.set_title("Årsvariation och illustrativ årsproduktion")

    annual_production_axis = annual_axis.twinx()
    annual_production_axis.plot(
        profile.annual_years,
        profile.annual_production_gwh,
        marker="s",
        color="#E07A28",
        label="Uppskattad årsproduktion",
    )
    annual_production_axis.axhline(
        profile.target_annual_energy_gwh,
        color="#E07A28",
        linestyle=":",
        alpha=0.8,
        label=f"Målnivå {profile.target_annual_energy_gwh:.0f} GWh/år",
    )
    annual_production_axis.set_ylabel("Parkens årsproduktion (GWh)", color="#E07A28")
    annual_production_axis.tick_params(axis="y", labelcolor="#E07A28")
    annual_handles, annual_labels = annual_axis.get_legend_handles_labels()
    output_handles, output_labels = annual_production_axis.get_legend_handles_labels()
    annual_axis.legend(
        annual_handles + output_handles,
        annual_labels + output_labels,
        fontsize=8,
        loc="upper left",
    )

    figure.suptitle("Vind vid Vinga och illustrativ vindpark lik Lillgrund")
    figure.text(
        0.5,
        0.01,
        "Vindhastighet enligt SMHI (kvalitet G). Parkens produktion är en kubisk "
        "vindhastighetsproxy kalibrerad till 330 GWh/år i medeltal 2015–2022; "
        "ingen turbin-effektkurva används. Vinga är en proxy, inte Lillgrunds vindläge.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    plt.show()


def plot_mean_wind_speed_wind_roses(
    station_data: list[tuple[str, list[float], list[float]]],
) -> None:
    if len(station_data) != 2:
        raise ValueError("Exactly two wind datasets are required")

    sector_count = 16
    sector_width_degrees = 360 / sector_count
    sector_width_radians = 2 * pi / sector_count
    compass_labels = [
        "N", "NNO", "NO", "ONO", "O", "OSO", "SO", "SSO",
        "S", "SSV", "SV", "VSV", "V", "VNV", "NV", "NNV",
    ]
    rose_data: list[tuple[str, list[float], int, float]] = []

    for station_name, directions, speeds in station_data:
        if not directions or len(directions) != len(speeds):
            raise ValueError(
                f"{station_name}: directions and speeds must have the same "
                "non-empty length"
            )

        speed_totals = [0.0] * sector_count
        observation_counts = [0] * sector_count
        for direction, speed in zip(directions, speeds):
            if not 0 <= direction <= 360 or speed < 0:
                raise ValueError(
                    f"{station_name}: wind directions must be 0-360 and "
                    "speeds non-negative"
                )
            sector_index = int(
                ((direction % 360 + sector_width_degrees / 2) % 360)
                / sector_width_degrees
            )
            speed_totals[sector_index] += speed
            observation_counts[sector_index] += 1

        average_speeds = [
            total / count if count else 0.0
            for total, count in zip(speed_totals, observation_counts)
        ]
        rose_data.append(
            (
                station_name,
                average_speeds,
                len(directions),
                sum(speeds) / len(speeds),
            )
        )

    maximum_average_speed = max(
        max(average_speeds) for _, average_speeds, _, _ in rose_data
    )
    sector_centers = [
        index * sector_width_radians for index in range(sector_count)
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6.5),
        subplot_kw={"projection": "polar"},
    )

    for axis, (station_name, average_speeds, observation_count, overall_average) in zip(
        axes,
        rose_data,
    ):
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.bar(
            sector_centers,
            average_speeds,
            width=sector_width_radians,
            color="#168AAD",
            edgecolor="white",
            linewidth=1.2,
            align="center",
        )
        axis.set_xticks(sector_centers)
        axis.set_xticklabels(compass_labels)
        axis.set_ylim(0, maximum_average_speed * 1.15)
        axis.set_title(
            f"{station_name}\nTotalt medel: {overall_average:.2f} m/s",
            pad=24,
        )
        axis.set_ylabel("Medelvindhastighet (m/s)")
        axis.set_rlabel_position(22.5)
        axis.grid(alpha=0.35)
        axis.text(
            0.5,
            -0.13,
            f"{observation_count:,} godkända observationer",
            transform=axis.transAxes,
            ha="center",
        )

    figure.suptitle("Genomsnittlig vindhastighet per vindriktning", y=1.02)
    figure.text(
        0.5,
        0.015,
        "Stapelhöjden visar medelhastigheten för vindar från respektive riktning; "
        "samma hastighetsskala används för båda stationerna.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    plt.show()


if __name__ == "__main__":
    data_directory = Path.cwd() / "data"
    smhi_csv_path = (
        data_directory
        / "vinddata"
        / "smhi-opendata_3_4_71380_20261005_154420.csv"
    )
    wind_timestamps, wind_speeds = load_smhi_wind_speed_data(smhi_csv_path)
    wind_farm_profile = estimate_wind_farm_profile(
        wind_timestamps,
        wind_speeds,
        target_annual_energy_gwh=330.0,
        calibration_years=(2015, 2022),
    )
    print(
        f"Vindprofil: {len(wind_timestamps):,} godkända SMHI-observationer; "
        f"hela år {wind_farm_profile.annual_years[0]}–"
        f"{wind_farm_profile.annual_years[-1]}; "
        f"kalibrerad faktor={wind_farm_profile.production_scale:.8f} "
        "MWh/(m/s)^3-timme."
    )
    print(
        f"Illustrativ vindparksproduktion uppskattad för "
        f"{wind_farm_profile.annual_years[-1]}: "
        f"{wind_farm_profile.annual_production_gwh[-1]:.1f} GWh."
    )
    plot_wind_farm_profile(wind_farm_profile)

    vinga_directions, vinga_speeds = load_smhi_wind_data(smhi_csv_path)
    landvetter_csv_path = (
        data_directory
        / "vinddata"
        / "smhi-opendata_3_4_72420_20261011_181505.csv"
    )
    landvetter_directions, landvetter_speeds = load_smhi_wind_data(
        landvetter_csv_path
    )
    plot_mean_wind_speed_wind_roses(
        [
            ("Vinga", vinga_directions, vinga_speeds),
            ("Landvetter", landvetter_directions, landvetter_speeds),
        ]
    )


    hydropower_profile = load_latest_complete_daily_production(
        data_directory / "vattendata" / "ProduktionsStatistik_vatten.csv",
        target_annual_energy_gwh=210.0,
        power_limit_mw=36.0,
    )
    scaled_hydropower_gwh = sum(hydropower_profile.production_mwh) / 1000
    print(
        f"Lilla Edet-profil: nationell vattenkraftsprofil från "
        f"{hydropower_profile.source_year}, skalad mot "
        f"{hydropower_profile.target_annual_energy_gwh:.0f} GWh och begränsad "
        f"till {hydropower_profile.power_limit_mw:.0f} MW. Efter begränsning: "
        f"{scaled_hydropower_gwh:.1f} GWh; "
        f"maxeffekt {max(hydropower_profile.production_mwh):.1f} MW."
    )
    plot_national_hydropower_profile(hydropower_profile)

    representative_data = load_representative_year_data(
        data_directory / "konsumptiondata" / "Förbrukningsstatistik.csv",
        data_directory / "soldata" / "ProduktionsStatistik_sol.csv",
        wind_farm_profile,
        hydropower_profile,
    )
    print(
        f"Representativ profil: {len(representative_data.timestamps) // 24} dagar "
        f"(8760 timmar); sol från {representative_data.solar_source_year}, "
        f"förbrukningsmedel från {representative_data.consumption_source_years[0]}–"
        f"{representative_data.consumption_source_years[-1]}, illustrativ "
        f"vindpark från SMHI-år {representative_data.wind_profile_source_years[0]}–"
        f"{representative_data.wind_profile_source_years[-1]}. Faktisk svensk "
        "vindkraft ingår inte i årsmodellen."
    )

    consumption_csv_path = (
        data_directory
        / "konsumptiondata"
        / "TAB3654_sv(sveriges_konsumption_per_komun).csv"
    )
    years, gothenburg_shares = load_municipal_consumption_share(
        consumption_csv_path
    )
    print(
        f"Göteborgs andel av Sveriges totala energianvändning "
        f"{years[-1]}: {gothenburg_shares[-1]:.2f}%."
    )
    plot_municipal_consumption_share(years, gothenburg_shares)

    estimated_days, estimated_consumption = (
        load_estimated_gothenburg_daily_electricity(representative_data)
    )
    print(
        f"Uppskattad elförbrukning i Göteborg för {len(estimated_days):,} dagar "
        "med 4,4% av Sveriges timförbrukning."
    )
    plot_estimated_gothenburg_daily_electricity(
        estimated_days,
        estimated_consumption,
    )

    storage_estimate = calculate_solar_storage_requirement(
        representative_data,
        round_trip_efficiency=STORAGE_ROUND_TRIP_EFFICIENCY,
    )
    no_wind_farm_data = replace(
        representative_data,
        wind_farm_production_mwh=[0.0] * len(representative_data.timestamps),
        wind_farm_scale_factor=0.0,
    )
    no_wind_farm_estimate = calculate_solar_storage_requirement(
        no_wind_farm_data,
        round_trip_efficiency=STORAGE_ROUND_TRIP_EFFICIENCY,
    )
    lilla_edet_estimate = calculate_solar_storage_requirement(
        representative_data,
        round_trip_efficiency=STORAGE_ROUND_TRIP_EFFICIENCY,
        include_hydropower=True,
    )
    effective_solar_share = (
        storage_estimate.solar_scale_factor
        * storage_estimate.municipality_share
        * 100
    )
    print(
        "Solfaktor som balanserar det representativa året efter tillägg av "
        "den illustrativa vindparken (faktisk svensk vindkraft ingår inte): "
        f"{storage_estimate.solar_scale_factor:.4f}× "
        f"(motsvarar {effective_solar_share:.1f}% av Sveriges solel "
        "allokerad till Göteborg)."
    )
    print(
        f"Maximal uppskattad soleffekt: "
        f"{storage_estimate.maximum_solar_power_mw:,.0f} MW; maximal samlad "
        f"produktion: {storage_estimate.maximum_generation_power_mw:,.0f} MW; "
        f"lagringskapacitet: "
        f"{storage_estimate.storage_capacity_mwh / 1_000_000:.2f} TWh; "
        f"maximal laddnings-/urladdningseffekt: "
        f"{storage_estimate.maximum_charge_power_mw:,.0f}/"
        f"{storage_estimate.maximum_discharge_power_mw:,.0f} MW."
    )
    plot_solar_storage_requirement(storage_estimate)
    optimal_wind_scale, optimized_wind_estimate = (
        optimize_wind_farm_scale_for_storage(
            representative_data,
            round_trip_efficiency=STORAGE_ROUND_TRIP_EFFICIENCY,
        )
    )
    current_farm_annual_gwh = (
        sum(representative_data.wind_farm_production_mwh) / 1000
    )
    optimized_farm_annual_gwh = (
        sum(optimized_wind_estimate.wind_farm_production_mwh) / 1000
    )
    baseline_solar_annual_gwh = sum(storage_estimate.solar_production_mwh) / 1000
    optimized_solar_annual_gwh = (
        sum(optimized_wind_estimate.solar_production_mwh) / 1000
    )
    additional_wind_annual_gwh = (
        optimized_farm_annual_gwh - current_farm_annual_gwh
    )
    solar_reduction_gwh = baseline_solar_annual_gwh - optimized_solar_annual_gwh
    solar_reduction_percent = (
        solar_reduction_gwh / baseline_solar_annual_gwh * 100
    )
    print(
        f"Optimerad sol- och vindmodell (utan vattenkraft): vindparken skalas "
        f"till {optimal_wind_scale:.3f}×, alltså {optimized_farm_annual_gwh:.1f} "
        f"GWh/år totalt och {additional_wind_annual_gwh:.1f} GWh/år extra "
        f"({optimal_wind_scale - 1:.3f} ytterligare gårdsstorlekar). "
        f"Solfaktorn blir {optimized_wind_estimate.solar_scale_factor:.4f}× "
        f"och solelen {optimized_solar_annual_gwh:.1f} GWh/år, "
        f"{solar_reduction_gwh:.1f} GWh/år ({solar_reduction_percent:.1f}%) "
        f"mindre än i dagens vindparksscenario. Minsta beräknade "
        f"lagringskapacitet: "
        f"{optimized_wind_estimate.storage_capacity_mwh / 1_000_000:.3f} TWh."
    )
    plot_solar_storage_requirement(optimized_wind_estimate)
    storage_reduction_mwh = (
        no_wind_farm_estimate.storage_capacity_mwh
        - storage_estimate.storage_capacity_mwh
    )
    storage_reduction_percent = (
        storage_reduction_mwh / no_wind_farm_estimate.storage_capacity_mwh * 100
    )
    print(
        f"Jämförelse utan vindpark: solfaktor "
        f"{no_wind_farm_estimate.solar_scale_factor:.4f}× "
        f"({no_wind_farm_estimate.solar_scale_factor * no_wind_farm_estimate.municipality_share * 100:.1f}% "
        f"av Sveriges solel), lagringskapacitet "
        f"{no_wind_farm_estimate.storage_capacity_mwh / 1_000_000:.2f} TWh. "
        f"Med vindparken minskar behovet med "
        f"{storage_reduction_mwh / 1_000_000:.2f} TWh "
        f"({storage_reduction_percent:.1f}%)."
    )
    plot_solar_storage_requirement(no_wind_farm_estimate)
    lilla_edet_storage_reduction_mwh = (
        storage_estimate.storage_capacity_mwh
        - lilla_edet_estimate.storage_capacity_mwh
    )
    lilla_edet_solar_factor_change = (
        storage_estimate.solar_scale_factor
        - lilla_edet_estimate.solar_scale_factor
    )
    print(
        f"Med Lilla Edets vattenkraft tillagd ovanpå vindparksmodellen: "
        f"solfaktor {lilla_edet_estimate.solar_scale_factor:.4f}× "
        f"({lilla_edet_estimate.solar_scale_factor * lilla_edet_estimate.municipality_share * 100:.1f}% "
        f"av Sveriges solel), lagringskapacitet "
        f"{lilla_edet_estimate.storage_capacity_mwh / 1_000_000:.2f} TWh. "
        f"Jämfört med modellen med vindpark minskar solfaktorn med "
        f"{lilla_edet_solar_factor_change:.4f}× och lagringsbehovet med "
        f"{lilla_edet_storage_reduction_mwh / 1_000_000:.2f} TWh."
    )
    plot_solar_storage_requirement(lilla_edet_estimate)

    wind_only_estimate = calculate_wind_only_storage_requirement(
        representative_data,
        round_trip_efficiency=STORAGE_ROUND_TRIP_EFFICIENCY,
    )
    print(
        "Vind-only-scenario: vindprofilen skalas "
        f"{wind_only_estimate.wind_scale_factor:.3f}× till "
        f"{sum(wind_only_estimate.wind_production_mwh) / 1000:.1f} GWh/år; "
        f"lagringskapacitet "
        f"{wind_only_estimate.storage_capacity_mwh / 1_000_000:.2f} TWh; "
        f"maximal laddnings-/urladdningseffekt "
        f"{wind_only_estimate.maximum_charge_power_mw:,.0f}/"
        f"{wind_only_estimate.maximum_discharge_power_mw:,.0f} MW."
    )
    plot_wind_only_storage_requirement(wind_only_estimate)

    
