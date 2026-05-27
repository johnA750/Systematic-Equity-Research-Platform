from pathlib import Path

import pandas as pd

from data.sp500 import normalize_yahoo_ticker


CURRENT_SP500 = "CURRENT_SP500"
HISTORICAL_SP500 = "HISTORICAL_SP500"
CUSTOM_UNIVERSE = "CUSTOM_UNIVERSE"
VALID_UNIVERSE_MODES = {CURRENT_SP500, HISTORICAL_SP500, CUSTOM_UNIVERSE}

REQUIRED_HISTORY_COLUMNS = {"ticker", "entry_date"}
OPTIONAL_HISTORY_COLUMNS = {
    "exit_date",
    "security",
    "sector",
    "sub_industry",
    "event_reason",
    "source",
}


def load_constituent_history(path: str | Path) -> pd.DataFrame:
    """
    Load point-in-time membership periods from a user- or vendor-supplied CSV.

    `entry_date` and `exit_date` are treated as inclusive membership dates.
    A missing `exit_date` identifies a security still active at the end of
    the supplied membership history.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Historical constituent file not found: {source_path}. "
            "Provide a point-in-time membership CSV before running HISTORICAL_SP500."
        )

    history = pd.read_csv(source_path)
    missing = REQUIRED_HISTORY_COLUMNS.difference(history.columns)
    if missing:
        required = ", ".join(sorted(REQUIRED_HISTORY_COLUMNS))
        raise ValueError(
            f"Historical constituent file is missing columns {sorted(missing)}. "
            f"Required columns: {required}."
        )
    history = history.copy()
    history["ticker"] = history["ticker"].map(normalize_yahoo_ticker)
    history["entry_date"] = pd.to_datetime(history["entry_date"], errors="coerce")
    if "exit_date" not in history:
        history["exit_date"] = pd.NaT
    else:
        history["exit_date"] = pd.to_datetime(history["exit_date"], errors="coerce")
    if history["entry_date"].isna().any():
        raise ValueError("Historical constituent file contains invalid entry_date values.")
    invalid_periods = history["exit_date"].notna() & (
        history["exit_date"] < history["entry_date"]
    )
    if invalid_periods.any():
        raise ValueError("Historical constituent file contains exit dates before entry dates.")

    duplicate_periods = history.duplicated(["ticker", "entry_date", "exit_date"])
    history = history.loc[~duplicate_periods].copy()
    for column in OPTIONAL_HISTORY_COLUMNS:
        if column not in history:
            history[column] = ""
    columns = [
        "ticker",
        "entry_date",
        "exit_date",
        "security",
        "sector",
        "sub_industry",
        "event_reason",
        "source",
    ]
    return history[columns].sort_values(["entry_date", "ticker"]).reset_index(drop=True)


def build_membership_mask(
    dates: pd.Index,
    constituent_history: pd.DataFrame,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Return a daily point-in-time membership mask with no future membership use."""
    index = pd.DatetimeIndex(dates)
    columns = tickers or sorted(constituent_history["ticker"].unique())
    mask = pd.DataFrame(False, index=index, columns=columns)
    for record in constituent_history.itertuples():
        if record.ticker not in mask.columns:
            continue
        active = index >= record.entry_date
        if pd.notna(record.exit_date):
            active &= index <= record.exit_date
        mask.loc[active, record.ticker] = True
    return mask


def combine_eligibility_masks(
    membership_mask: pd.DataFrame | None,
    liquidity_mask: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Combine point-in-time membership and liquidity screens."""
    if membership_mask is None:
        return liquidity_mask
    if liquidity_mask is None:
        return membership_mask
    liquidity = liquidity_mask.reindex(
        index=membership_mask.index,
        columns=membership_mask.columns,
        fill_value=False,
    )
    return membership_mask & liquidity


def compute_universe_turnover(membership_mask: pd.DataFrame) -> pd.DataFrame:
    """Measure daily additions, removals, and constituent count."""
    membership = membership_mask.astype(int)
    changes = membership.diff().fillna(0)
    additions = changes.clip(lower=0).sum(axis=1)
    removals = (-changes.clip(upper=0)).sum(axis=1)
    output = pd.DataFrame(
        {
            "constituent_count": membership.sum(axis=1),
            "additions": additions.astype(int),
            "removals": removals.astype(int),
        },
        index=membership.index,
    )
    denominator = output["constituent_count"].replace(0, pd.NA)
    output["one_way_turnover"] = (
        output[["additions", "removals"]].max(axis=1) / denominator
    )
    output["two_way_turnover"] = (
        (output["additions"] + output["removals"]) / denominator
    )
    output.index.name = "date"
    return output.reset_index()


def compute_annual_membership_changes(universe_turnover: pd.DataFrame) -> pd.DataFrame:
    """Summarize annual point-in-time additions and removals."""
    changes = universe_turnover.copy()
    changes["year"] = pd.to_datetime(changes["date"]).dt.year
    return (
        changes.groupby("year", as_index=False)
        .agg(
            additions=("additions", "sum"),
            removals=("removals", "sum"),
            average_constituent_count=("constituent_count", "mean"),
            ending_constituent_count=("constituent_count", "last"),
        )
    )


def report_missing_historical_prices(
    constituent_history: pd.DataFrame,
    returned_prices: pd.DataFrame,
) -> pd.DataFrame:
    """List membership symbols for which the price vendor returned no history."""
    requested = sorted(constituent_history["ticker"].unique())
    returned = set(returned_prices.columns)
    missing = [ticker for ticker in requested if ticker not in returned]
    return pd.DataFrame(
        {
            "ticker": missing,
            "price_history_available": False,
            "note": "No price series returned by configured market-data vendor.",
        }
    )
