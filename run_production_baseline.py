import os
import runpy


if __name__ == "__main__":
    os.environ["BACKTEST_MODE"] = "production_baseline"
    runpy.run_path(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "run_backtest.py")),
        run_name="__main__",
    )
