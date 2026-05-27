import pandas as pd


def compute_gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Return gross portfolio exposure as total absolute position weight."""
    exposure = weights.abs().sum(axis=1)
    exposure.name = "gross_exposure"
    return exposure


def summarize_gross_exposure(weights: pd.DataFrame) -> dict[str, float]:
    """Return concise gross exposure statistics."""
    exposure = compute_gross_exposure(weights)
    return {
        "Average Gross Exposure": exposure.mean(),
        "Maximum Gross Exposure": exposure.max(),
        "Minimum Gross Exposure": exposure.min(),
    }


def compute_sector_exposures(
    weights: pd.DataFrame,
    ticker_to_sector: dict[str, str],
) -> pd.DataFrame:
    """Aggregate long-only position weights into daily sector exposures."""
    mapped_columns = {
        ticker: ticker_to_sector[ticker]
        for ticker in weights.columns
        if ticker in ticker_to_sector
    }
    if not mapped_columns:
        return pd.DataFrame(index=weights.index)

    sector_weights = weights[list(mapped_columns)].rename(columns=mapped_columns)
    return sector_weights.T.groupby(level=0).sum().T


def summarize_sector_exposures(
    weights: pd.DataFrame,
    ticker_to_sector: dict[str, str],
) -> pd.DataFrame:
    """Return average and maximum daily exposure by mapped sector."""
    exposure = compute_sector_exposures(weights, ticker_to_sector)
    if exposure.empty:
        return exposure

    summary = pd.DataFrame(
        {
            "Average Exposure": exposure.mean(),
            "Maximum Exposure": exposure.max(),
        }
    )
    return summary.sort_values("Maximum Exposure", ascending=False)
