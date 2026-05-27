from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    description: str
    universe_mode: str
    enable_liquidity_filter: bool
    max_position_weight: float | None
    slippage_model: str
    enable_constrained_optimizer: bool = False

    @property
    def use_sp500_universe(self) -> bool:
        return self.universe_mode in {"CURRENT_SP500", "HISTORICAL_SP500"}


PRODUCTION_BASELINE = ExperimentMode(
    name="production_baseline",
    description="Fixed 50-stock baseline using production inverse-vol construction.",
    universe_mode="CUSTOM_UNIVERSE",
    enable_liquidity_filter=False,
    max_position_weight=None,
    slippage_model="fixed",
)

SP500_RESEARCH = ExperimentMode(
    name="sp500_research",
    description="Current S&P 500 research universe with liquidity and position controls.",
    universe_mode="CURRENT_SP500",
    enable_liquidity_filter=True,
    max_position_weight=0.10,
    slippage_model="turnover_adjusted",
)

HISTORICAL_SP500_RESEARCH = ExperimentMode(
    name="historical_sp500_research",
    description="Point-in-time historical S&P 500 research universe from dated membership input.",
    universe_mode="HISTORICAL_SP500",
    enable_liquidity_filter=True,
    max_position_weight=0.10,
    slippage_model="turnover_adjusted",
)

WALK_FORWARD_VALIDATION = ExperimentMode(
    name="walk_forward_validation",
    description="Walk-forward validation of production-style strategy configurations.",
    universe_mode="CUSTOM_UNIVERSE",
    enable_liquidity_filter=False,
    max_position_weight=None,
    slippage_model="fixed",
)


MODES = {
    mode.name: mode
    for mode in (
        PRODUCTION_BASELINE,
        SP500_RESEARCH,
        HISTORICAL_SP500_RESEARCH,
        WALK_FORWARD_VALIDATION,
    )
}


def get_experiment_mode(name: str) -> ExperimentMode:
    try:
        return MODES[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODES))
        raise ValueError(f"Unknown experiment mode '{name}'. Available modes: {available}.") from exc
