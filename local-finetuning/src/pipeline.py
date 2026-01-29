"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pipeline orchestrator: prepare dataset, train (LoRA SFT), evaluate, and full-run workflow."
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config.settings import load_settings, settings_to_dict
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_dir, set_global_seed, snapshot_config
from src.core.train import run_training
from src.core.prepare_dataset import run_prepare_dataset
from src.core.errors import log_and_raise_missing_raw_data
from src.core.evaluate import run_evaluation

def _create_run_dir(base_dir: Path) -> Path:
    """
		Create a unique run directory

		Args:
			base_dir: Base directory for runs

		Returns:
			Created run directory path
    """
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    return ensure_dir(run_dir)

def prepare(env_path: Optional[Path] = None) -> Path:
    """
		Run dataset preparation step

		Args:
			env_path: Optional .env path

		Returns:
			Path to the processed data directory
    """
    
    settings = load_settings(env_path=env_path)

    run_dir = _create_run_dir(settings.training.output_dir)
    logs_dir = ensure_dir(run_dir / "logs")

    logger = get_logger(__name__, logs_dir=logs_dir)
    logger.info("Starting dataset preparation")

    snapshot_config(settings_to_dict(settings), run_dir=run_dir)

    ensure_dir(settings.data.raw_dir)
    ensure_dir(settings.data.interim_dir)
    ensure_dir(settings.data.processed_dir)

    try:
        run_prepare_dataset(
            raw_dir=settings.data.raw_dir,
            interim_dir=settings.data.interim_dir,
            processed_dir=settings.data.processed_dir,
            train_file=settings.data.train_file,
            val_file=settings.data.val_file,
            test_file=settings.data.test_file,
            label_list_file=settings.data.label_list_file,
            split_seed=settings.data.split_seed,
            train_ratio=settings.data.train_ratio,
            val_ratio=settings.data.val_ratio,
            test_ratio=settings.data.test_ratio,
            logger=logger,
        )
    except FileNotFoundError:
        log_and_raise_missing_raw_data(settings.data.raw_dir)
        
    logger.info("Dataset preparation completed")
    return settings.data.processed_dir

def train(env_path: Optional[Path] = None) -> Path:
    """
		Run training step (LoRA SFT)

		Args:
			env_path: Optional .env path

		Returns:
			Path to the training run directory
    """
    
    settings = load_settings(env_path=env_path)

    run_dir = _create_run_dir(settings.training.output_dir)
    logs_dir = ensure_dir(run_dir / "logs")

    logger = get_logger(__name__, logs_dir=logs_dir)
    logger.info("Starting training (LoRA SFT)")

    snapshot_config(settings_to_dict(settings), run_dir=run_dir)

    set_global_seed(settings.training.seed)

    run_training(
        run_dir=run_dir,
        processed_dir=settings.data.processed_dir,
        train_file=settings.data.train_file,
        val_file=settings.data.val_file,
        base_model_name=settings.training.base_model_name,
        use_gpu=settings.training.use_gpu,
        seed=settings.training.seed,
        num_train_epochs=settings.training.num_train_epochs,
        learning_rate=settings.training.learning_rate,
        batch_size=settings.training.batch_size,
        grad_accum_steps=settings.training.grad_accum_steps,
        max_seq_len=settings.training.max_seq_len,
        lora_r=settings.training.lora_r,
        lora_alpha=settings.training.lora_alpha,
        lora_dropout=settings.training.lora_dropout,
        bf16=settings.training.bf16,
        fp16=settings.training.fp16,
        quantization_4bit=settings.training.quantization_4bit,
        logging_steps=settings.training.logging_steps,
        save_steps=settings.training.save_steps,
        logger=logger,
    )

    logger.info("Training completed")
    return run_dir

def evaluate(env_path: Optional[Path] = None, run_dir: Optional[Path] = None) -> Path:
    """
		Run evaluation step

		Args:
			env_path: Optional .env path
			run_dir: Optional run directory to evaluate. If None, a new run dir is created

		Returns:
			Path to the evaluation report directory
    """
    
    settings = load_settings(env_path=env_path)

    if run_dir is None:
        run_dir = _create_run_dir(settings.training.output_dir)

    logs_dir = ensure_dir(run_dir / "logs")
    logger = get_logger(__name__, logs_dir=logs_dir)
    logger.info("Starting evaluation")

    snapshot_config(settings_to_dict(settings), run_dir=run_dir)

    report_dir = ensure_dir(run_dir / "reports")

    run_evaluation(
        run_dir=run_dir,
        report_dir=report_dir,
        processed_dir=settings.data.processed_dir,
        test_file=settings.data.test_file,
        top_k=settings.evaluation.top_k,
        enable_reject=settings.evaluation.enable_reject,
        reject_token=settings.evaluation.reject_token,
        logger=logger,
        base_model_name=None,
        use_gpu=settings.training.use_gpu,
        label_list_file=settings.data.label_list_file,
        max_new_tokens=16,
    )

    logger.info("Evaluation completed")
    return report_dir

def full_run(env_path: Optional[Path] = None) -> Path:
    """
		Run full pipeline: prepare -> train -> evaluate

		Args:
			env_path: Optional .env path

		Returns:
			Path to the final evaluation report directory
    """
    
    prepare(env_path=env_path)
    run_dir = train(env_path=env_path)
    report_dir = evaluate(env_path=env_path, run_dir=run_dir)

    return report_dir