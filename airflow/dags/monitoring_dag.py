'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Airflow DAG for monitoring: export Prometheus metrics and execution traces (calls src/ directly)."
'''

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.monitoring.metrics import export_metrics_snapshot
from src.monitoring.tracing import export_traces_snapshot
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("airflow.monitoring_dag")

## ============================================================
## DAG DEFAULTS
## ============================================================
DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "georges_nassopoulos",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

## ============================================================
## TASKS
## ============================================================
def run_metrics_export() -> None:
    """
        Export Prometheus metrics snapshot

        Workflow:
            1) Collect metrics from src.monitoring.metrics
            2) Export snapshot to artifacts/reports
    """

    logger.info("Starting metrics export")

    export_metrics_snapshot()

    logger.info("Metrics export completed")

def run_tracing_export() -> None:
    """
        Export execution traces snapshot

        Workflow:
            1) Collect traces from src.monitoring.tracing
            2) Persist JSON trace report
    """

    logger.info("Starting tracing export")

    export_traces_snapshot()

    logger.info("Tracing export completed")

## ============================================================
## DAG
## ============================================================
with DAG(
    dag_id="autonomous_ai_platform_monitoring",
    description="Monitoring pipeline: export metrics + traces",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/10 * * * *",  # every 10 minutes
    catchup=False,
    max_active_runs=1,
    tags=["autonomous-ai-platform", "monitoring"],
) as dag:

    export_metrics = PythonOperator(
        task_id="export_metrics",
        python_callable=run_metrics_export,
    )

    export_traces = PythonOperator(
        task_id="export_traces",
        python_callable=run_tracing_export,
    )

    export_metrics >> export_traces