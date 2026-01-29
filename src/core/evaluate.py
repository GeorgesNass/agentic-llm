"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Evaluation workflow: deterministic inference, strict CISP metrics, and JSON report generation."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.metrics import (
    confusion_matrix,
    exact_match,
    hallucination_rate,
    label_coverage,
    missing_labels,
    most_confused_pairs,
)
from src.utils.io_utils import read_jsonl, read_label_list, save_json
from src.utils.utils import detect_device, ensure_dir    
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

## --------------------------------------------------------------------------------------
## Prompting helpers (kept local: core ML logic)
## --------------------------------------------------------------------------------------
def _format_inference_prompt(record: Dict[str, Any]) -> str:
    """
		Format a record into an inference prompt (no answer appended)

		Args:
			record: Instruction record

		Returns:
			Prompt string
    """
    
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{user_input}\n\n"
        f"### Output:\n"
    )

def _postprocess_prediction(text: str) -> str:
    """
		Postprocess generated text into a clean label

		Args:
			text: Raw generated text

		Returns:
			Clean predicted label
    """
    
    if not text:
        return ""

    ## Keep only first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line

    return ""

## --------------------------------------------------------------------------------------
## Model loading
## --------------------------------------------------------------------------------------
def _load_model_and_tokenizer(
    run_dir: Path,
    base_model_name: Optional[str],
    use_gpu: bool,
    logger: Any,
):
    """
		Load base model + LoRA adapter for inference

		Args:
			run_dir: Training run directory
			base_model_name: Optional base model override
			use_gpu: Whether GPU is allowed
			logger: Logger instance

		Returns:
			(model, tokenizer, device_str)

		Raises:
			FileNotFoundError: If adapter directory is missing
			ValueError: If base model name cannot be resolved
    """

    adapter_dir = run_dir / "exports" / "lora_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")

    ## Infer base model from metadata if needed
    if base_model_name is None:
        metadata_path = run_dir / "training_metadata.json"
        if not metadata_path.exists():
            raise ValueError("base_model_name not provided and training_metadata.json missing")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        base_model_name = metadata.get("base_model_name")

    if not base_model_name:
        raise ValueError("Unable to resolve base_model_name for evaluation")

    device = detect_device(use_gpu=use_gpu)
    logger.info(f"Evaluation device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer, device

## --------------------------------------------------------------------------------------
## Main evaluation entry
## --------------------------------------------------------------------------------------
def run_evaluation(
    run_dir: Path,
    report_dir: Path,
    processed_dir: Path,
    test_file: str,
    top_k: int,
    enable_reject: bool,
    reject_token: str,
    logger: Any,
    base_model_name: Optional[str] = None,
    use_gpu: bool = True,
    label_list_file: Optional[Path] = None,
    max_new_tokens: int = 16,
) -> Path:
    """
		Run deterministic evaluation and export metrics/report

		Args:
			run_dir: Training run directory
			report_dir: Output report directory
			processed_dir: Processed dataset directory
			test_file: Test JSONL filename
			top_k: Top-k value (reserved for future ranking)
			enable_reject: Whether reject token is allowed
			reject_token: Canonical reject token
			logger: Logger instance
			base_model_name: Optional base model override
			use_gpu: Whether GPU is allowed
			label_list_file: Optional allowed label list
			max_new_tokens: Max tokens generated per sample

		Returns:
			Path to evaluation report JSON file
    """
    
    ensure_dir(report_dir)

    test_path = processed_dir / test_file
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    ## Load data and labels
    records = read_jsonl(test_path)
    allowed_labels = read_label_list(label_list_file)

    if allowed_labels:
        logger.info(f"Loaded {len(allowed_labels)} allowed labels")
    else:
        logger.info("No allowed label list provided")

    ## Load model
    model, tokenizer, device = _load_model_and_tokenizer(
        run_dir=run_dir,
        base_model_name=base_model_name,
        use_gpu=use_gpu,
        logger=logger,
    )

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ValueError("torch is required for evaluation") from exc

    ## Deterministic generation config (anti-hallucination)
    gen_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    y_true: List[str] = []
    y_pred: List[str] = []

    ## Inference loop (intentionally sequential for clarity/debugging)
    for idx, rec in enumerate(records):
        true_label = str(rec.get("output", "")).strip()
        y_true.append(true_label)

        prompt = _format_inference_prompt(rec)
        inputs = tokenizer(prompt, return_tensors="pt")

        if device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = decoded.split("### Output:\n", 1)[-1]
        pred = _postprocess_prediction(generated)

        if enable_reject and not pred:
            pred = reject_token

        y_pred.append(pred)

        if (idx + 1) % 50 == 0:
            logger.info(f"Evaluated {idx + 1}/{len(records)} samples")

    ## Metrics
    em = exact_match(y_true, y_pred)
    hall = hallucination_rate(y_pred, allowed_labels)
    conf = confusion_matrix(y_true, y_pred)

    coverage = label_coverage(y_pred)
    never_pred = missing_labels(allowed_labels, y_pred) if allowed_labels else None

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_samples": len(records),
        "metrics": {
            "exact_match": em,
            "hallucination_rate": hall,
            "top_k": top_k,
        },
        "diagnostics": {
            "num_unique_predictions": len(coverage),
            "most_confused_pairs_top20": most_confused_pairs(conf)[:20],
            "never_predicted_labels": never_pred,
        },
        "generation": {
            "deterministic": True,
            "max_new_tokens": max_new_tokens,
            "device": device,
        },
    }

    report_path = report_dir / "evaluation_report.json"
    save_json(report, report_path)

    ## Save per-sample predictions
    predictions_path = report_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for rec, yt, yp in zip(records, y_true, y_pred):
            f.write(
                json.dumps(
                    {"input": rec.get("input", ""), "y_true": yt, "y_pred": yp},
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(f"Evaluation completed. Report written to: {report_path}")
    return report_path