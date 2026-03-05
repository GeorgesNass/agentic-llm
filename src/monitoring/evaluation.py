'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Evaluation utilities: compute basic answer metrics and optional LLM-as-a-judge scoring with structured outputs."
'''

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.core.errors import EvaluationError, ValidationError
from src.orchestrator.routing import route_chat_completion
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str, _safe_json_loads
from src.utils.validation_utils import _must_be_non_empty, _clamp_float
from src.utils.text_utils import _count_words, _detect_urls, _detect_code_block

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class BasicMetrics:
    """
        Basic text metrics

        Args:
            length_chars: Character length
            length_words: Word count
            contains_numbers: Whether answer contains any digits
            contains_code_block: Whether answer contains ``` block
            contains_urls: Whether answer contains http(s) like string
    """

    length_chars: int
    length_words: int
    contains_numbers: bool
    contains_code_block: bool
    contains_urls: bool

@dataclass(frozen=True)
class JudgeResult:
    """
        LLM-as-a-judge result

        Args:
            verdict: pass or fail
            score: 0..10 score
            rationale: Short rationale
            issues: Issues list
            confidence: 0..1 confidence
            metadata: Extra metadata
    """

    verdict: str
    score: float
    rationale: str
    issues: List[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class EvaluationReport:
    """
        Full evaluation report

        Args:
            query: User query
            answer: Produced answer
            basic: BasicMetrics
            judge: Optional JudgeResult
            metadata: Extra metadata
    """

    query: str
    answer: str
    basic: BasicMetrics
    judge: Optional[JudgeResult]
    metadata: Dict[str, Any]

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _compute_basic_metrics(answer: str) -> BasicMetrics:
    """
        Compute basic metrics from answer text

        Args:
            answer: Answer string

        Returns:
            BasicMetrics
    """

    ## Compute counts
    length_chars = len(answer)
    length_words = _count_words(answer)

    ## Detect simple signals
    contains_numbers = bool(re.search(r"\d", answer))
    contains_code_block = _detect_code_block(answer)
    contains_urls = _detect_urls(answer)

    return BasicMetrics(
        length_chars=length_chars,
        length_words=length_words,
        contains_numbers=contains_numbers,
        contains_code_block=contains_code_block,
        contains_urls=contains_urls,
    )

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
        Extract JSON object from judge output

        Args:
            text: Model output

        Returns:
            Dict
    """

    ## Direct JSON path
    direct = _safe_json_loads(text.strip())
    if direct:
        return direct

    ## Regex path
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    return _safe_json_loads(match.group(0))

def _build_judge_prompt(user_query: str, answer: str) -> List[Dict[str, Any]]:
    """
        Build LLM judge prompt

        Args:
            user_query: Query
            answer: Answer text

        Returns:
            Chat messages
    """

    ## Keep schema strict for stable parsing
    system = (
        "You are a strict evaluator for an autonomous AI platform.\n"
        "Return ONLY valid JSON.\n"
        "Score from 0 to 10.\n"
        "verdict must be pass or fail.\n"
        "JSON schema:\n"
        "{\n"
        '  "verdict": "pass|fail",\n'
        '  "score": 0.0,\n'
        '  "rationale": "short string",\n'
        '  "issues": ["string"],\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Rules:\n"
        "- Fail if the answer makes claims not supported by tool outputs.\n"
        "- Fail if the answer ignores key constraints from the user.\n"
        "- Prefer concise, actionable answers.\n"
    )

    user = f"User query:\n{user_query}\n\nAnswer:\n{answer}"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _parse_judge_result(data: Dict[str, Any], raw_text: str) -> JudgeResult:
    """
        Parse JudgeResult from dict

        Args:
            data: Parsed JSON dict
            raw_text: Raw model text

        Returns:
            JudgeResult
    """

    verdict = str(data.get("verdict", "fail")).strip().lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"

    score = _clamp_float(data.get("score"), default=0.0, min_v=0.0, max_v=10.0)

    rationale = str(data.get("rationale", "")).strip()
    if not rationale:
        rationale = "n/a"

    issues_raw = data.get("issues", [])
    issues = [str(x) for x in issues_raw] if isinstance(issues_raw, list) else []

    confidence = _clamp_float(data.get("confidence"), default=0.5, min_v=0.0, max_v=1.0)

    return JudgeResult(
        verdict=verdict,
        score=score,
        rationale=rationale,
        issues=issues[:12],
        confidence=confidence,
        metadata={"raw": _safe_str(raw_text[:4000])},
    )

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def evaluate_answer(
    user_query: str = "",
    answer: str = "",
    *,
    query: str = "",
    use_llm_judge: bool = True,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """
        Evaluate an answer with basic metrics and optional LLM judge

        Args:
            user_query: User query
            answer: Produced answer
            query: Backward-compatible alias for user_query
            use_llm_judge: Whether to run LLM-as-a-judge
            prefer_local: Prefer local backend for judge
            use_gpu: GPU flag for local judge
            **kwargs: Extra ignored keyword arguments (compat)

        Returns:
            EvaluationReport
    """

    ## Ignore unused kwargs for compatibility with tests/callers
    _ = kwargs

    ## Backward-compatible mapping: allow query=... instead of user_query=...
    if not user_query:
        user_query = query

    ## Validate inputs
    uq = _must_be_non_empty(user_query, "user_query")
    ans = _must_be_non_empty(answer, "answer")

    ## Compute basic metrics
    basic = _compute_basic_metrics(ans)

    judge: Optional[JudgeResult] = None

    ## Run LLM judge if enabled
    if use_llm_judge:
        messages = _build_judge_prompt(uq, ans)

        try:
            start = time.perf_counter()

            out = route_chat_completion(
                messages=messages,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                temperature=0.0,
                top_p=1.0,
                max_tokens=500,
            )

            duration = time.perf_counter() - start

            raw_text = str(out.get("text", "")).strip()
            data = _extract_first_json_object(raw_text)

            judge = _parse_judge_result(data, raw_text)

            ## Attach judge call metadata
            judge_meta = dict(judge.metadata)
            judge_meta.update(
                {
                    "duration_sec": duration,
                    "provider": out.get("provider"),
                    "model": out.get("model"),
                    "usage": out.get("usage", {}),
                }
            )

            judge = JudgeResult(
                verdict=judge.verdict,
                score=judge.score,
                rationale=judge.rationale,
                issues=judge.issues,
                confidence=judge.confidence,
                metadata=judge_meta,
            )

        except Exception as exc:
            ## Do not break the pipeline: wrap into EvaluationError
            raise EvaluationError(
                message="LLM judge evaluation failed",
                error_code="evaluation_error",
                details={"cause": _safe_str(exc)},
                origin="evaluation",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    ## Build report
    report = EvaluationReport(
        query=uq,
        answer=ans,
        basic=basic,
        judge=judge,
        metadata={"judge_enabled": use_llm_judge},
    )

    ## Log summary
    logger.info(
        "EvaluationReport | judge=%s | score=%s | words=%s",
        bool(judge),
        judge.score if judge else "n/a",
        basic.length_words,
    )

    ## Unit-test friendly output: tests expect a dict when judge is disabled
    if not use_llm_judge:
        payload = report_to_dict(report)

        ## Unit-test friendly alias: some callers expect "metrics"
        payload["metrics"] = dict(payload.get("basic", {}))

        return payload
        
    return report
    
@log_execution_time
def report_to_dict(report: EvaluationReport) -> Dict[str, Any]:
    """
        Convert EvaluationReport to a JSON-serializable dict

        Args:
            report: EvaluationReport

        Returns:
            Dict
    """

    ## Convert dataclasses
    payload = asdict(report)

    ## Ensure nested dataclasses are safe
    payload["basic"] = asdict(report.basic)
    payload["judge"] = asdict(report.judge) if report.judge else None

    return payload