import pandas as pd


def compute_regime_weights(
    regime_series: pd.Series,
    bull_momentum_weight: float = 0.7,
    bear_momentum_weight: float = 0.3,
) -> pd.DataFrame:
    weights = pd.DataFrame(index=regime_series.index)

    weights["momentum"] = 0.0
    weights["mean_reversion"] = 0.0

    bull_mask = regime_series == "bull"
    bear_mask = regime_series == "bear"

    weights.loc[bull_mask, "momentum"] = bull_momentum_weight
    weights.loc[bull_mask, "mean_reversion"] = 1 - bull_momentum_weight

    weights.loc[bear_mask, "momentum"] = bear_momentum_weight
    weights.loc[bear_mask, "mean_reversion"] = 1 - bear_momentum_weight

    weights = weights.ffill().fillna(0.5)

    return weights