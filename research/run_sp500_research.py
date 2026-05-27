import os
import runpy
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    os.environ["BACKTEST_MODE"] = "sp500_research"
    runpy.run_path(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run_backtest.py")),
        run_name="__main__",
    )
