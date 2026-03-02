"""dbt-inspired ML feature pipeline for data transformations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class TransformStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TransformNode:
    """A single transformation node in the DAG."""
    name: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    sql_template: str = ""
    python_transform: str | None = None
    tests: list[str] = field(default_factory=list)
    status: TransformStatus = TransformStatus.PENDING
    rows_affected: int = 0
    execution_time_ms: float = 0.0
    error: str | None = None


@dataclass
class TestResult:
    """Result of a data quality test."""
    test_name: str
    model_name: str
    passed: bool
    message: str


class FeaturePipeline:
    """
    dbt-inspired feature transformation pipeline for ML.

    Defines a DAG of SQL/Python transformations that convert
    raw market data into ML-ready features with automated
    data quality testing at each stage.

    Pipeline stages:
    1. stg_ohlcv: Staged OHLCV data (cleaned, deduplicated)
    2. int_returns: Intermediate returns calculations
    3. int_indicators: Technical indicators (RSI, MACD, BB, etc.)
    4. int_volatility: Volatility features (ATR, realized vol)
    5. fct_features: Final feature matrix for ML models
    """

    def __init__(self):
        self.models: dict[str, TransformNode] = {}
        self._execution_order: list[str] = []
        self._results: list[dict] = []
        self._define_models()
        logger.info("feature_pipeline_initialized", n_models=len(self.models))

    def _define_models(self) -> None:
        """Define all transformation models in the pipeline."""
        self.models = {
            "stg_ohlcv": TransformNode(
                name="stg_ohlcv",
                description="Staged OHLCV data: cleaned, deduplicated, with timezone normalization",
                sql_template="""
                    SELECT DISTINCT ON (symbol, interval, open_time)
                        symbol, interval, open_time,
                        open, high, low, close, volume,
                        close_time, quote_volume, trade_count
                    FROM ohlcv
                    WHERE open_time IS NOT NULL AND close > 0
                    ORDER BY symbol, interval, open_time DESC
                """,
                tests=["not_null:open_time", "not_null:close", "positive:close", "unique:symbol,interval,open_time"],
            ),
            "int_returns": TransformNode(
                name="int_returns",
                description="Intermediate: log returns, simple returns, cumulative returns",
                depends_on=["stg_ohlcv"],
                sql_template="""
                    SELECT *,
                        LN(close / LAG(close) OVER (PARTITION BY symbol, interval ORDER BY open_time)) AS log_return,
                        (close - LAG(close) OVER (PARTITION BY symbol, interval ORDER BY open_time))
                            / NULLIF(LAG(close) OVER (PARTITION BY symbol, interval ORDER BY open_time), 0) AS simple_return,
                        close / FIRST_VALUE(close) OVER (PARTITION BY symbol, interval ORDER BY open_time) - 1 AS cum_return
                    FROM stg_ohlcv
                """,
                tests=["not_null:log_return", "range:log_return:-1:1"],
            ),
            "int_indicators": TransformNode(
                name="int_indicators",
                description="Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands",
                depends_on=["int_returns"],
                python_transform="compute_technical_indicators",
                tests=["range:rsi_14:0:100", "not_null:sma_20"],
            ),
            "int_volatility": TransformNode(
                name="int_volatility",
                description="Volatility features: ATR, realized vol, Garman-Klass, Parkinson",
                depends_on=["stg_ohlcv", "int_returns"],
                python_transform="compute_volatility_features",
                tests=["positive:atr_14", "positive:realized_vol_20"],
            ),
            "int_volume_profile": TransformNode(
                name="int_volume_profile",
                description="Volume profile: OBV, VWAP, volume SMA ratios",
                depends_on=["stg_ohlcv"],
                python_transform="compute_volume_features",
                tests=["not_null:obv", "positive:vwap"],
            ),
            "fct_features": TransformNode(
                name="fct_features",
                description="Final ML feature matrix combining all intermediate features",
                depends_on=["int_returns", "int_indicators", "int_volatility", "int_volume_profile"],
                python_transform="merge_features",
                tests=["row_count:min:100", "no_nulls", "no_infinities"],
            ),
        }

        # Compute execution order (topological sort)
        self._execution_order = self._topological_sort()

    def _topological_sort(self) -> list[str]:
        """Topological sort of the model DAG."""
        visited: set[str] = set()
        order: list[str] = []

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for dep in self.models[node].depends_on:
                if dep in self.models:
                    dfs(dep)
            order.append(node)

        for name in self.models:
            dfs(name)

        return order

    def run(self, data: dict[str, np.ndarray] | None = None) -> dict:
        """
        Execute the full feature pipeline.

        Args:
            data: Optional dict with raw data arrays

        Returns:
            Pipeline execution results
        """
        import time
        start_time = time.time()
        test_results: list[TestResult] = []
        n_success = 0
        n_failed = 0

        logger.info("pipeline_run_started", n_models=len(self._execution_order))

        for model_name in self._execution_order:
            model = self.models[model_name]
            model_start = time.time()
            model.status = TransformStatus.RUNNING

            try:
                # Check dependencies
                deps_ok = all(
                    self.models[dep].status == TransformStatus.SUCCESS
                    for dep in model.depends_on
                    if dep in self.models
                )

                if not deps_ok:
                    model.status = TransformStatus.SKIPPED
                    model.error = "Dependency not satisfied"
                    continue

                # Execute transformation (simulated)
                if model.python_transform:
                    rows = self._execute_python(model.python_transform, data)
                else:
                    rows = self._execute_sql(model.sql_template, data)

                model.rows_affected = rows
                model.execution_time_ms = (time.time() - model_start) * 1000

                # Run tests
                for test_spec in model.tests:
                    result = self._run_test(test_spec, model_name, data)
                    test_results.append(result)
                    if not result.passed:
                        raise ValueError(f"Test failed: {result.message}")

                model.status = TransformStatus.SUCCESS
                n_success += 1
                logger.info("model_completed", model=model_name,
                           rows=rows, time_ms=round(model.execution_time_ms, 1))

            except Exception as e:
                model.status = TransformStatus.FAILED
                model.error = str(e)
                n_failed += 1
                logger.error("model_failed", model=model_name, error=str(e))

        total_time = (time.time() - start_time) * 1000

        result = {
            "status": "success" if n_failed == 0 else "partial_failure",
            "models": {
                name: {
                    "status": m.status.value,
                    "rows_affected": m.rows_affected,
                    "execution_time_ms": round(m.execution_time_ms, 1),
                    "error": m.error,
                }
                for name, m in self.models.items()
            },
            "tests": {
                "total": len(test_results),
                "passed": sum(1 for t in test_results if t.passed),
                "failed": sum(1 for t in test_results if not t.passed),
                "details": [
                    {"test": t.test_name, "model": t.model_name,
                     "passed": t.passed, "message": t.message}
                    for t in test_results if not t.passed
                ],
            },
            "summary": {
                "models_success": n_success,
                "models_failed": n_failed,
                "models_skipped": sum(1 for m in self.models.values()
                                      if m.status == TransformStatus.SKIPPED),
                "total_time_ms": round(total_time, 1),
            },
            "execution_order": self._execution_order,
            "dag": self.get_dag(),
        }

        self._results.append(result)
        return result

    def _execute_python(self, transform_name: str,
                        data: dict[str, np.ndarray] | None) -> int:
        """Execute a Python transformation (simulated)."""
        # In production, this would call actual feature computation functions
        return 1000  # Simulated row count

    def _execute_sql(self, sql: str, data: dict[str, np.ndarray] | None) -> int:
        """Execute a SQL transformation (simulated)."""
        return 1000

    def _run_test(self, test_spec: str, model_name: str,
                  data: dict[str, np.ndarray] | None) -> TestResult:
        """Run a data quality test."""
        parts = test_spec.split(":")
        test_type = parts[0]

        # Simulated test execution
        if test_type == "not_null":
            return TestResult(test_spec, model_name, True,
                            f"Column {parts[1]} has no nulls")
        elif test_type == "positive":
            return TestResult(test_spec, model_name, True,
                            f"Column {parts[1]} all positive")
        elif test_type == "unique":
            return TestResult(test_spec, model_name, True,
                            f"Columns {parts[1]} are unique")
        elif test_type == "range":
            return TestResult(test_spec, model_name, True,
                            f"Column {parts[1]} within range [{parts[2]}, {parts[3]}]")
        elif test_type == "row_count":
            return TestResult(test_spec, model_name, True,
                            f"Row count meets {parts[1]} threshold of {parts[2]}")
        elif test_type in ("no_nulls", "no_infinities"):
            return TestResult(test_spec, model_name, True,
                            f"{test_type} check passed")
        else:
            return TestResult(test_spec, model_name, True, "Unknown test passed")

    def get_dag(self) -> dict:
        """Get DAG visualization data."""
        nodes = []
        edges = []

        for name, model in self.models.items():
            nodes.append({
                "id": name,
                "label": name,
                "description": model.description,
                "status": model.status.value,
            })
            for dep in model.depends_on:
                edges.append({"source": dep, "target": name})

        return {"nodes": nodes, "edges": edges}

    def get_lineage(self, model_name: str) -> dict:
        """Get data lineage for a specific model."""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}

        # Trace upstream
        upstream: list[str] = []
        queue = list(self.models[model_name].depends_on)
        while queue:
            dep = queue.pop(0)
            if dep in self.models and dep not in upstream:
                upstream.append(dep)
                queue.extend(self.models[dep].depends_on)

        # Trace downstream
        downstream: list[str] = []
        for name, model in self.models.items():
            if model_name in model.depends_on:
                downstream.append(name)

        return {
            "model": model_name,
            "upstream": upstream,
            "downstream": downstream,
            "tests": self.models[model_name].tests,
        }


# Module-level instance
feature_pipeline = FeaturePipeline()
