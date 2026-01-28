'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "OCR orchestration for rag-drive-gcp: run OCR Universal locally (Docker) or via a remote microservice."
'''

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from src.model.settings import get_settings
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_directories

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("ocr")

## ============================================================
## DATA CLASSES
## ============================================================
@dataclass
class OcrResult:
    """
        OCR result container

        Attributes:
            input_path (Path): Original input file path
            text_path (Path): Output text file path
            duration_s (float): OCR runtime in seconds
            mode (str): OCR mode used: local_docker|remote_service
    """

    input_path: Path
    text_path: Path
    duration_s: float
    mode: str

## ============================================================
## HELPERS
## ============================================================
def _is_text_file(path: Path) -> bool:
    """
        Check if the file is already a text file

        Args:
            path (Path): Input file path

        Returns:
            bool: True if path looks like a text file
    """
    
    return path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".jsonl"}

def _default_text_output_path(input_path: Path, output_dir: Path) -> Path:
    """
        Build a default output text path for an OCR job

        Args:
            input_path (Path): Input file path
            output_dir (Path): Output directory

        Returns:
            Path: Output text file path
    """
    
    safe_stem = input_path.name.replace("/", "_").replace("\\", "_")
    return output_dir / f"{safe_stem}.txt"

## ============================================================
## LOCAL DOCKER OCR (OCR UNIVERSAL)
## ============================================================
def run_ocr_local_docker(
    input_path: Path,
    output_dir: Path,
) -> Path:
    """
        Run OCR using the OCR Universal project in a local Docker container

        Notes:
            - This function assumes the OCR Universal Docker image is available locally
            - We mount the input file and output directory into the container
            - The container is expected to write a text file to the mounted output dir

        Required env vars:
            - OCR_DOCKER_IMAGE

        Args:
            input_path (Path): Path to the file to OCR
            output_dir (Path): Directory where OCR output will be stored

        Returns:
            Path: Path to the generated text file

        Raises:
            FileNotFoundError: If input_path does not exist
            ValueError: If OCR_DOCKER_IMAGE is missing
            RuntimeError: If Docker execution fails or output is missing
    """
    
    settings = get_settings()

    if not input_path.exists():
        raise FileNotFoundError(f"OCR input file not found: {input_path}")

    if not settings.ocr_docker_image:
        raise ValueError("OCR_DOCKER_IMAGE is not set in .env for local_docker mode.")

    ensure_directories(output_dir)

    ## We isolate OCR inputs/outputs for predictable docker mounts
    input_abs = input_path.resolve()
    output_abs = output_dir.resolve()

    ## IMPORTANT:
    ## The OCR Universal project may expose different CLI conventions.
    ## Here we use a conservative pattern:
    ## - Mount input as /data/input/<filename>
    ## - Mount output as /data/output/
    ## - Call OCR project in "convert" mode with --path and standard output folder.
    container_input_dir = "/data/input"
    container_output_dir = "/data/output"
    container_input_path = f"{container_input_dir}/{input_abs.name}"

    ## Build docker command
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{str(input_abs.parent)}:{container_input_dir}:ro",
        "-v", f"{str(output_abs)}:{container_output_dir}",
        settings.ocr_docker_image,
        "--mode", "convert",
        "--path", container_input_path,
    ]

    logger.info("Running OCR Universal via local Docker.")
    logger.debug(f"Docker command: {' '.join(cmd)}")

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        logger.error(f"Docker OCR failed with code {completed.returncode}.")
        logger.error(f"stdout: {completed.stdout}")
        logger.error(f"stderr: {completed.stderr}")
        raise RuntimeError("Local Docker OCR failed. See logs for details.")

    ## Heuristic: look for an output file matching "<input_name>.<ext>.txt"
    ## Example from OCR Universal: "file.pdf.txt" or "file.docx.txt"
    candidates = list(output_abs.glob(f"{input_abs.name}*.txt"))
    if not candidates:
        ## Fallback: any txt produced recently
        candidates = sorted(output_abs.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise RuntimeError("OCR output text file not found after Docker OCR execution.")

    logger.info(f"OCR output detected: {candidates[0]}")
    return candidates[0]

## ============================================================
## REMOTE OCR MICROSERVICE
## ============================================================
def run_ocr_remote_service(
    input_path: Path,
    output_dir: Path,
) -> Path:
    """
        Run OCR by calling a remote OCR Universal microservice

        Required env vars:
            - OCR_SERVICE_URL

        Assumptions:
            - The service exposes an HTTP endpoint that accepts a file upload and returns text
            - We keep this generic and easy to adapt

        Args:
            input_path (Path): Path to the file to OCR
            output_dir (Path): Directory where OCR output will be stored

        Returns:
            Path: Path to the generated text file

        Raises:
            FileNotFoundError: If input_path does not exist
            ValueError: If OCR_SERVICE_URL is missing
            RuntimeError: If the service call fails
    """
    
    settings = get_settings()

    if not input_path.exists():
        raise FileNotFoundError(f"OCR input file not found: {input_path}")

    if not settings.ocr_service_url:
        raise ValueError("OCR_SERVICE_URL is not set in .env for remote_service mode.")

    ensure_directories(output_dir)

    url = settings.ocr_service_url.rstrip("/") + "/ocr"
    out_path = _default_text_output_path(input_path, output_dir)

    logger.info(f"Calling OCR microservice: {url}")
    with open(input_path, "rb") as f:
        files = {"file": (input_path.name, f)}
        response = requests.post(url, files=files, timeout=300)

    if response.status_code != 200:
        logger.error(f"OCR service failed: status={response.status_code}")
        logger.error(f"Response: {response.text[:1000]}")
        raise RuntimeError("Remote OCR service call failed.")

    ## We assume response contains plain text
    out_path.write_text(response.text, encoding="utf-8", errors="ignore")
    logger.info(f"OCR text saved: {out_path}")
    return out_path

## ============================================================
## PUBLIC API
## ============================================================
def ocr_document(
    input_path: Path,
    output_dir: Optional[Path] = None,
) -> OcrResult:
    """
        Run OCR on a document using the configured mode

        If the input is already a text file, this function will copy it to the output dir
        and treat it as "already OCR'ed"

        Args:
            input_path (Path): Input file path
            output_dir (Optional[Path]): Output directory. Defaults to settings.tmp_dir

        Returns:
            OcrResult: OCR result container
    """
    
    settings = get_settings()

    start = time.time()
    local_output_dir = output_dir if output_dir else settings.tmp_dir
    ensure_directories(local_output_dir)

    ## If already text, simply copy to tmp output for consistent downstream logic
    if _is_text_file(input_path):
        out_path = _default_text_output_path(input_path, local_output_dir)
        shutil.copyfile(str(input_path), str(out_path))
        duration = time.time() - start
        logger.info("Input is already a text file. Skipping OCR.")
        return OcrResult(
            input_path=input_path,
            text_path=out_path,
            duration_s=duration,
            mode="skip_text",
        )

    mode = (settings.ocr_mode or "local_docker").strip().lower()

    if mode == "local_docker":
        text_path = run_ocr_local_docker(input_path=input_path, output_dir=local_output_dir)
    elif mode == "remote_service":
        text_path = run_ocr_remote_service(input_path=input_path, output_dir=local_output_dir)
    else:
        raise ValueError(f"Unsupported OCR_MODE: {settings.ocr_mode}")

    duration = time.time() - start
    logger.info(f"OCR completed in {duration:.2f}s | mode={mode}")

    return OcrResult(
        input_path=input_path,
        text_path=text_path,
        duration_s=duration,
        mode=mode,
    )