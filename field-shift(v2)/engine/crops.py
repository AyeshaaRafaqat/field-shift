"""Crop trait table: loading and validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["crop", "season", "family", "duration_days", "water_mm", "n_fixer", "root", "residue",
                    "heat_tol", "frost_tol", "drought_tol", "income", "notes"]
RATING_COLUMNS = ["root", "residue", "heat_tol", "frost_tol", "drought_tol", "income"]   # 1-5 scales
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "crops.csv"


def validate_crops(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = table is fine)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {', '.join(missing)}"]
    if df["crop"].duplicated().any():
        problems.append(f"duplicate crop names: {sorted(df.loc[df['crop'].duplicated(), 'crop'])}")
    bad_season = df.loc[~df["season"].isin(["Kharif", "Rabi"]), "crop"].tolist()
    if bad_season:
        problems.append(f"season must be Kharif or Rabi for: {bad_season}")
    for season in ("Kharif", "Rabi"):
        if not (df["season"] == season).any():
            problems.append(f"no {season} crops in the table")
    for col in RATING_COLUMNS:
        bad = df.loc[~pd.to_numeric(df[col], errors="coerce").between(1, 5), "crop"].tolist()
        if bad:
            problems.append(f"{col} must be 1-5 for: {bad}")
    for col, lo, hi in (("duration_days", 30, 400), ("water_mm", 50, 3000)):
        bad = df.loc[~pd.to_numeric(df[col], errors="coerce").between(lo, hi), "crop"].tolist()
        if bad:
            problems.append(f"{col} must be {lo}-{hi} for: {bad}")
    bad = df.loc[~df["n_fixer"].isin([0, 1]), "crop"].tolist()
    if bad:
        problems.append(f"n_fixer must be 0 or 1 for: {bad}")
    return problems


def load_crops(path: str | Path | None = None) -> pd.DataFrame:
    """Load crops.csv (indexed by crop name). Defaults to the file next to the app."""
    df = pd.read_csv(path or DEFAULT_PATH)
    problems = validate_crops(df)
    if problems:
        raise ValueError("crops.csv problems: " + "; ".join(problems))
    return df.set_index("crop")
