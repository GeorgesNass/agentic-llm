'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for autonomous AI platform: tasks, agents, tools, prompts, dependencies and execution coherence"
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    compute_quality_score,
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_consistency")

## ============================================================
## ISSUE HANDLING
## ============================================================
def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Append issue and log it

        Args:
            issues: Issue list
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            None
    """

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    issues.append(issue)

    ## Log depending on severity
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_tasks(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate tasks structure and dependencies

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    tasks = data.get("tasks")

    ## Tasks must exist
    if not tasks:
        _add_issue(issues, "tasks_missing", "error", "Tasks are required")
        return

    if not isinstance(tasks, list):
        _add_issue(issues, "tasks_type", "error", "Tasks must be a list")
        return

    task_ids = set()

    for idx, task in enumerate(tasks):

        ## Validate task format
        if not isinstance(task, dict):
            _add_issue(issues, "task_format", "error", "Task must be dict", {"index": idx})
            continue

        task_id = task.get("id")
        agent = task.get("agent")
        prompt = task.get("prompt")
        depends_on = task.get("depends_on", [])

        ## ID validation
        if not task_id:
            _add_issue(issues, "task_id_missing", "error", "Task id missing", {"index": idx})
        else:
            task_ids.add(task_id)

        ## Agent validation
        if not agent:
            _add_issue(issues, "task_agent_missing", "error", "Agent missing", {"index": idx})

        ## Prompt validation
        if not isinstance(prompt, str) or not prompt.strip():
            _add_issue(issues, "task_prompt_invalid", "error", "Invalid prompt", {"index": idx})

        ## Dependencies validation
        if not isinstance(depends_on, list):
            _add_issue(issues, "task_depends_type", "error", "depends_on must be list", {"index": idx})

    ## Validate dependency coherence
    for task in tasks:
        for dep in task.get("depends_on", []):
            if dep not in task_ids:
                _add_issue(
                    issues,
                    "task_dependency_invalid",
                    "error",
                    "Dependency refers to unknown task",
                    {"dependency": dep},
                )

def _validate_agents(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate agents configuration

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    agents = data.get("agents")

    ## Agents must exist
    if not agents:
        _add_issue(issues, "agents_missing", "error", "Agents are required")
        return

    if not isinstance(agents, dict):
        _add_issue(issues, "agents_type", "error", "Agents must be dict")
        return

    for name, agent in agents.items():

        ## Each agent must define a model
        if not isinstance(agent, dict):
            _add_issue(issues, "agent_format", "error", "Agent must be dict", {"agent": name})
            continue

        if "model" not in agent:
            _add_issue(issues, "agent_model_missing", "error", "Agent model missing", {"agent": name})

def _validate_tools(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate tools configuration

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    tools = data.get("tools")

    ## Tools are optional
    if tools is None:
        return

    if not isinstance(tools, dict):
        _add_issue(issues, "tools_type", "error", "Tools must be dict")
        return

    for name, tool in tools.items():

        if not isinstance(tool, dict):
            _add_issue(issues, "tool_format", "error", "Tool must be dict", {"tool": name})
            continue

        if "type" not in tool:
            _add_issue(issues, "tool_type_missing", "error", "Tool type missing", {"tool": name})

def _validate_structure(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate schema and types

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    ## Schema validation
    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    ## Type validation
    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(
    data: Dict[str, Any],
) -> float:
    """
        Compute quality score

        Args:
            data: Input data

        Returns:
            float
    """

    return compute_quality_score(data)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_consistency(
    data: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data consistency pipeline for autonomous AI platform

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Dict[str, Any]
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize input payload
        data = normalize_data(data)

        ## Validate orchestration
        _validate_tasks(data, issues)
        _validate_agents(data, issues)
        _validate_tools(data, issues)

        ## Validate schema/types
        _validate_structure(data, issues)

        ## Compute quality score
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
        }

        ## Strict mode blocks pipeline
        if strict and errors:
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc