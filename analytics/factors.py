from io import BytesIO, StringIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
import statsmodels.api as sm


KEN_FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FACTOR_DATASETS = {
    ("ff3", "daily"): "F-F_Research_Data_Factors_daily_CSV.zip",
    ("ff3", "monthly"): "F-F_Research_Data_Factors_CSV.zip",
    ("ff5", "daily"): "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3_CSV.zip",
}
MODEL_FACTORS = {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


def _parse_french_csv(text: str, frequency: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_index = next(
        index for index, line in enumerate(lines)
        if line.replace(" ", "").startswith(",Mkt-RF")
    )
    date_length = 8 if frequency == "daily" else 6
    rows = []
    for line in lines[header_index + 1:]:
        first_field = line.split(",", 1)[0].strip()
        if first_field.isdigit() and len(first_field) == date_length:
            rows.append(line)
        elif rows:
            break

    frame = pd.read_csv(StringIO("\n".join([lines[header_index]] + rows)), index_col=0)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce") / 100.0
    if frequency == "daily":
        frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    else:
        frame.index = pd.PeriodIndex(
            frame.index.astype(str), freq="M"
        ).to_timestamp(how="end").normalize()
    frame.index.name = "date"
    return frame


def load_fama_french_factors(
    dataset: str = "ff3",
    frequency: str = "daily",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve official Ken French factor returns in decimal units.

    Parameters use `ff3` or `ff5` datasets and `daily` or `monthly`
    frequency. Monthly indices are normalized to calendar month end.
    """
    dataset = dataset.lower()
    frequency = frequency.lower()
    try:
        filename = FACTOR_DATASETS[(dataset, frequency)]
    except KeyError as exc:
        raise ValueError("Supported factor datasets are ff3/ff5 at daily/monthly frequency.") from exc

    request = Request(
        f"{KEN_FRENCH_BASE_URL}/{filename}",
        headers={"User-Agent": "Trading-Backtester/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        content = response.read()
    with ZipFile(BytesIO(content)) as archive:
        member = archive.namelist()[0]
        text = archive.read(member).decode("utf-8", errors="replace")

    factors = _parse_french_csv(text, frequency)
    if start_date is not None:
        factors = factors.loc[pd.Timestamp(start_date):]
    if end_date is not None:
        factors = factors.loc[:pd.Timestamp(end_date)]
    return factors


def align_portfolio_and_factors(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Align realized portfolio returns with contemporaneous factor returns."""
    frequency = frequency.lower()
    returns = portfolio_returns.copy()
    if frequency == "monthly":
        returns = (1 + returns).resample("ME").prod() - 1
    elif frequency != "daily":
        raise ValueError("frequency must be 'daily' or 'monthly'.")

    return pd.concat([returns.rename("portfolio"), factors], axis=1).dropna()


def run_factor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    frequency: str = "daily",
    use_hac: bool = True,
    hac_lags: int = 5,
) -> dict:
    """Estimate CAPM, FF3, or FF5 exposure using statsmodels OLS."""
    model = model.lower()
    if model not in MODEL_FACTORS:
        raise ValueError("model must be capm, ff3, or ff5.")
    required = MODEL_FACTORS[model] + ["RF"]
    missing = [column for column in required if column not in factors.columns]
    if missing:
        raise ValueError(f"Factor data does not contain required columns: {missing}.")

    aligned = align_portfolio_and_factors(portfolio_returns, factors, frequency)
    if aligned.empty:
        raise ValueError("No overlapping observations between portfolio and factor data.")

    y = aligned["portfolio"] - aligned["RF"]
    X = sm.add_constant(aligned[MODEL_FACTORS[model]])
    regression = sm.OLS(y, X)
    if use_hac:
        fitted = regression.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        covariance = f"HAC(Newey-West, maxlags={hac_lags})"
    else:
        fitted = regression.fit()
        covariance = "nonrobust"

    periods_per_year = 252 if frequency == "daily" else 12
    result = {
        "model": model.upper(),
        "frequency": frequency,
        "observations": len(aligned),
        "alpha": fitted.params["const"],
        "alpha_annualized": fitted.params["const"] * periods_per_year,
        "alpha_t_stat": fitted.tvalues["const"],
        "alpha_p_value": fitted.pvalues["const"],
        "r_squared": fitted.rsquared,
        "covariance": covariance,
    }
    for factor in MODEL_FACTORS[model]:
        label = "market_beta" if factor == "Mkt-RF" else factor.lower()
        result[label] = fitted.params[factor]
        result[f"{label}_t_stat"] = fitted.tvalues[factor]
        result[f"{label}_p_value"] = fitted.pvalues[factor]
    return result


def run_factor_model_suite(
    portfolios: dict[str, pd.Series],
    frequency: str = "daily",
    include_ff5: bool = False,
    use_hac: bool = True,
    hac_lags: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run Ken French CAPM/FF3 and optionally FF5 models for portfolios."""
    factor_data = {
        "ff3": load_fama_french_factors("ff3", frequency, start_date, end_date)
    }
    models = [("capm", "ff3"), ("ff3", "ff3")]
    if include_ff5:
        factor_data["ff5"] = load_fama_french_factors("ff5", frequency, start_date, end_date)
        models.append(("ff5", "ff5"))

    rows = []
    for portfolio_name, returns in portfolios.items():
        for model, dataset in models:
            result = run_factor_regression(
                returns,
                factor_data[dataset],
                model=model,
                frequency=frequency,
                use_hac=use_hac,
                hac_lags=hac_lags,
            )
            rows.append({"portfolio": portfolio_name, **result})
    return pd.DataFrame(rows), factor_data


def compute_rolling_factor_exposures(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    frequency: str = "daily",
    window: int = 126,
    use_hac: bool = True,
    hac_lags: int = 5,
) -> pd.DataFrame:
    """Estimate trailing factor loadings and alpha over fixed observation windows."""
    if window < len(MODEL_FACTORS[model.lower()]) + 2:
        raise ValueError("window is too short for the requested factor model.")
    aligned = align_portfolio_and_factors(portfolio_returns, factors, frequency)
    results = []
    for end in range(window - 1, len(aligned)):
        sample = aligned.iloc[end - window + 1:end + 1]
        result = run_factor_regression(
            sample["portfolio"],
            sample.drop(columns="portfolio"),
            model=model,
            frequency=frequency,
            use_hac=use_hac,
            hac_lags=hac_lags,
        )
        results.append({"date": sample.index[-1], **result})

    if not results:
        return pd.DataFrame()
    rolling = pd.DataFrame(results).set_index("date")
    columns = ["alpha_annualized", "market_beta", "smb", "hml", "rmw", "cma"]
    return rolling[[column for column in columns if column in rolling.columns]]
