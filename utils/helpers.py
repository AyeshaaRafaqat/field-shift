"""Small shared helpers: input validation and simple data-quality reporting.

Nothing in this module talks to the network, so it is safe to import from
anywhere (services, analysis or the Streamlit app).
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Tuple, Union

import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]

# NASA POWER daily data starts on this date (documented by NASA POWER).
POWER_DAILY_EARLIEST_DATE = date(1981, 1, 1)


def validate_coordinates(latitude, longitude) -> Tuple[float, float]:
    """Check that latitude/longitude are real numbers inside valid ranges.

    Returns the values as floats. Raises ValueError with a readable message
    otherwise.
    """
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be numbers.") from None

    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude must be between -90 and 90 degrees (got {lat}).")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude must be between -180 and 180 degrees (got {lon}).")
    return lat, lon


def to_date(value: DateLike, name: str = "date") -> date:
    """Convert a date, datetime or 'YYYY-MM-DD' / 'YYYYMMDD' string to a date."""
    # datetime is a subclass of date, so it must be checked first.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"{name} must be a date or a 'YYYY-MM-DD' string (got {value!r}).")


def validate_date_range(
    start: DateLike,
    end: DateLike,
    earliest: date = POWER_DAILY_EARLIEST_DATE,
    latest: date | None = None,
) -> Tuple[date, date]:
    """Validate a start/end pair and return them as `date` objects."""
    start_date = to_date(start, "start_date")
    end_date = to_date(end, "end_date")
    latest = latest or date.today()

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date.")
    if start_date < earliest:
        raise ValueError(f"start_date cannot be before {earliest.isoformat()} (start of NASA POWER daily data).")
    if start_date > latest:
        raise ValueError("start_date is in the future.")
    if end_date > latest:
        raise ValueError("end_date is in the future.")
    return start_date, end_date


def parse_month_day(text: str) -> Tuple[int, int]:
    """Parse an 'MM-DD' string (e.g. '11-01') into (month, day).

    '02-29' is accepted because seasons are defined for any year.
    """
    try:
        month_text, day_text = text.strip().split("-")
        month, day = int(month_text), int(day_text)
        date(2000, month, day)  # 2000 is a leap year, so 02-29 is valid here
    except (ValueError, AttributeError):
        raise ValueError(f"'{text}' is not a valid MM-DD date (example: 11-01).") from None
    return month, day


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values and value ranges for each column of `df`."""
    rows = []
    total = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "days": total,
                "missing_days": missing,
                "missing_pct": round(100 * missing / total, 2) if total else float("nan"),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)
