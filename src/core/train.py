"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Training workflow: LoRA SFT with HuggingFace Transformers + PEFT (GPU/CPU/QLoRA supported)."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io_utils import read_jsonl
from src.utils.utils import detect_device, ensure_dir
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

## --------------------------------------------------------------------------------------
## Prompting helpers (core ML logic)
## --------------------------------------------------------------------------------------
def _format_train_text(record: Dict[str, Any]) -> str:
    """
		Format a single record into a causal LM training text

		Args:
			record: Instruction record

		Returns:
			Formatted training text
    """
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{user_input}\n\n"
        f"### Output:\n{output}"
    )

def _build_text_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
		Build minimal HF-compatible text dataset

		Args:
			records: Instruction records

		Returns:
			List of dicts with a single 'text' field
    """
    
    return [{"text": _format_train_text(r)} for r in records]

## --------------------------------------------------------------------------------------
## Training entry
## --------------------------------------------------------------------------------------
def run_training(
    run_dir: Path,
    processed_dir: Path,
    train_file: str,
    val_file: str,
    base_model_name: str,
    use_gpu: bool,
    seed: int,
    num_train_epochs: int,
    learning_rate: float,
    batch_size: int,
    grad_accum_steps: int,
    max_seq_len: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
    fp16: bool,
    quantization_4bit: bool,
    logging_steps: int,
    save_steps: int,
    logger: Any,
) -> Path:
    """
		Run LoRA SFT training

		Args:
			run_dir: Run directory for artifacts
			processed_dir: Directory with processed JSONL files
			train_file: Train JSONL filename
			val_file: Validation JSONL filename
			base_model_name: HF model id or local path
			use_gpu: Whether GPU usage is allowed
			seed: Random seed
			num_train_epochs: Number of epochs
			learning_rate: Learning rate
			batch_size: Per-device batch size
			grad_accum_steps: Gradient accumulation steps
			max_seq_len: Max sequence length
			lora_r: LoRA rank
			lora_alpha: LoRA alpha
			lora_dropout: LoRA dropout
			bf16: Use bfloat16 (GPU only)
			fp16: Use float16 (GPU only)
			quantization_4bit: Enable QLoRA (4-bit)
			logging_steps: Logging frequency
			save_steps: Checkpoint save frequency
			logger: Logger instance

		Returns:
			Path to exported LoRA adapter directory
    """

    ## bitsandbytes is required for QLoRA
    if quantization_4bit:
        try:
            import bitsandbytes  # noqa: F401  # type: ignore
        except ImportError as exc:
            raise ValueError("quantization_4bit=True requires bitsandbytes") from exc

    ensure_dir(run_dir)

    train_path = processed_dir / train_file
    val_path = processed_dir / val_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    ## Device resolution
    device = detect_device(use_gpu=use_gpu)
    logger.info(f"Training device: {device}")

    ## Disable mixed precision on CPU
    if device != "cuda":
        bf16 = False
        fp16 = False

    ## Seed everything
    set_seed(seed)

    ## Load datasets
    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)

    train_ds = Dataset.from_list(_build_text_dataset(train_records))
    val_ds = Dataset.from_list(_build_text_dataset(val_records))

    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ## Tokenization
    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        """
			Tokenize a batch of texts

			Args:
				batch: Batch with 'text' field

			Returns:
				Tokenized batch
        """
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    ## Model loading
    model_kwargs: Dict[str, Any] = {}
    if quantization_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "device_map": "auto",
            }
        )
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    if quantization_4bit:
        model = prepare_model_for_kbit_training(model)

    ## LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    ## Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        evaluation_strategy="steps",
        eval_steps=logging_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

    ## Export adapter
    adapter_dir = ensure_dir(run_dir / "exports" / "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    ## Save metadata
    metadata = {
        "base_model_name": base_model_name,
        "device": device,
        "seed": seed,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "max_seq_len": max_seq_len,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bf16": bf16,
        "fp16": fp16,
        "quantization_4bit": quantization_4bit,
        "adapter_dir": str(adapter_dir),
    }

    with (run_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"LoRA adapter exported to: {adapter_dir}")
    return adapter_dir