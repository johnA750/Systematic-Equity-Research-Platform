import pandas as pd


def normalize_with_position_cap(
    raw_weights: pd.Series,
    max_position_weight: float | None = None,
) -> pd.Series:
    """
    Normalize selected-name weights and optionally cap individual positions.

    When the selected names cannot absorb a full 100% allocation without
    exceeding the cap, the residual allocation remains in cash.
    """
    total_weight = raw_weights.sum()
    if total_weight <= 0:
        return raw_weights

    normalized = raw_weights / total_weight
    if max_position_weight is None:
        return normalized

    if not 0 < max_position_weight <= 1:
        raise ValueError("max_position_weight must be in (0, 1] or None.")

    capped = normalized.clip(upper=max_position_weight)
    residual = 1.0 - capped.sum()
    tolerance = 1e-12

    while residual > tolerance:
        eligible = (normalized > 0) & (capped < max_position_weight - tolerance)
        if not eligible.any():
            break

        redistribution = normalized.loc[eligible]
        redistribution = residual * redistribution / redistribution.sum()
        capacity = max_position_weight - capped.loc[eligible]
        addition = redistribution.where(redistribution <= capacity, capacity)
        capped.loc[eligible] += addition
        residual = 1.0 - capped.sum()

    return capped
