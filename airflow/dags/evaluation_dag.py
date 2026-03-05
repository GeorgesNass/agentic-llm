'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Airflow DAG for evaluation: run offline metrics + optional LLM-as-a-judge scoring (calls src/ directly)."
'''

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.monitoring.evaluation import run_evaluation_pipeline
from src.utils.env_utils import _get_env_bool, _get_env_str
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("airflow.evaluation_dag")

## ============================================================
## DAG DEFAULTS
## ============================================================
DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "georges_nassopoulos",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

## ============================================================
## TASKS
## ============================================================
def run_evaluation_task() -> None:
    """
        Run evaluation pipeline by calling src/ modules directly

        High-level workflow:
            1) Resolve evaluation input file path
            2) Resolve judge settings (optional)
            3) Run src.monitoring.evaluation.run_evaluation_pipeline
            4) Persist JSON report under artifacts/reports or artifacts/evaluations

        Env variables:
            EVAL_INPUT_PATH: Path to evaluation dataset JSON
            EVAL_OUTPUT_DIR: Output directory for reports
            EVAL_ENABLE_LLM_JUDGE: Enable LLM-as-a-judge
            EVAL_JUDGE_PREFER_LOCAL: Prefer local judge model if enabled
            USE_GPU: Whether GPU is allowed for local judge
            EVAL_MAX_SAMPLES: Optional cap (0 means no limit)

        Returns:
            None
    """

    ## Resolve input / output
    eval_input = _get_env_str("EVAL_INPUT_PATH", "./artifacts/evaluations/eval_input.json")
    eval_output_dir = _get_env_str("EVAL_OUTPUT_DIR", "./artifacts/evaluations")

    input_path = Path(eval_input).expanduser().resolve()
    output_dir = Path(eval_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Resolve evaluation flags
    enable_judge = _get_env_bool("EVAL_ENABLE_LLM_JUDGE", False)
    judge_prefer_local = _get_env_bool("EVAL_JUDGE_PREFER_LOCAL", False)
    use_gpu = _get_env_bool("USE_GPU", False)

    max_samples_raw = _get_env_str("EVAL_MAX_SAMPLES", "0").strip()
    try:
        max_samples = int(max_samples_raw)
    except ValueError:
        max_samples = 0

    logger.info(
        "Evaluation start | input=%s | output_dir=%s | llm_judge=%s | judge_local=%s | use_gpu=%s | max_samples=%s",
        str(input_path),
        str(output_dir),
        enable_judge,
        judge_prefer_local,
        use_gpu,
        max_samples,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"EVAL_INPUT_PATH not found: {input_path}")

    ## Call src evaluation pipeline
    run_evaluation_pipeline(
        eval_input_path=str(input_path),
        output_dir=str(output_dir),
        enable_llm_judge=enable_judge,
        judge_prefer_local=judge_prefer_local,
        use_gpu=use_gpu,
        max_samples=max_samples if max_samples > 0 else None,
    )

    logger.info("Evaluation success | input=%s", str(input_path))

## ============================================================
## DAG
## ============================================================
with DAG(
    dag_id="autonomous_ai_platform_evaluation",
    description="Evaluation pipeline: offline metrics + optional LLM judge",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["autonomous-ai-platform", "evaluation"],
) as dag:
    run_evaluation = PythonOperator(
        task_id="run_evaluation",
        python_callable=run_evaluation_task,
    )

    run_evaluation