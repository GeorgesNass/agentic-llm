'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for autonomous AI platform: normalization, validation of tasks, agents, tools and quality"
'''

from __future__ import annotations

from typing import Any, Dict, List

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("data_utils")

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Normalize input payload

        Args:
            data: Input dictionary

        Returns:
            Dict[str, Any]
    """

    normalized = {}

    for key, value in data.items():

        ## Normalize text fields
        if isinstance(value, str):
            logger.debug(f"Normalizing string: {key}")
            value = value.strip()

        ## Normalize tasks
        if key == "tasks" and isinstance(value, list):
            logger.debug("Normalizing tasks")

            normalized_tasks = []

            for task in value:
                if isinstance(task, dict):
                    normalized_tasks.append({
                        "id": str(task.get("id", "")).strip(),
                        "agent": str(task.get("agent", "")).strip(),
                        "prompt": str(task.get("prompt", "")).strip(),
                        "depends_on": task.get("depends_on", []),
                    })

            value = normalized_tasks

        ## Normalize agents
        if key == "agents" and isinstance(value, dict):
            logger.debug("Normalizing agents")

            normalized_agents = {}

            for name, agent in value.items():
                if isinstance(agent, dict):
                    normalized_agents[name] = {
                        "model": str(agent.get("model", "")).strip(),
                        **agent,
                    }

            value = normalized_agents

        ## Normalize tools
        if key == "tools" and isinstance(value, dict):
            logger.debug("Normalizing tools")

            normalized_tools = {}

            for name, tool in value.items():
                if isinstance(tool, dict):
                    normalized_tools[name] = {
                        "type": str(tool.get("type", "")).strip(),
                        **tool,
                    }

            value = normalized_tools

        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields for autonomous pipeline

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## Tasks required
    if "tasks" not in data:
        logger.error("Missing tasks field")
        issues.append({
            "rule": "schema_tasks",
            "level": "error",
            "message": "tasks are required",
        })

    ## Agents required
    if "agents" not in data:
        logger.error("Missing agents field")
        issues.append({
            "rule": "schema_agents",
            "level": "error",
            "message": "agents are required",
        })

    return issues

def validate_types(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate field types

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## Tasks type
    if "tasks" in data and not isinstance(data["tasks"], list):
        logger.error("Invalid tasks type")
        issues.append({
            "rule": "type_tasks",
            "level": "error",
            "message": "tasks must be list",
        })

    ## Agents type
    if "agents" in data and not isinstance(data["agents"], dict):
        logger.error("Invalid agents type")
        issues.append({
            "rule": "type_agents",
            "level": "error",
            "message": "agents must be dict",
        })

    ## Tools type
    if "tools" in data and not isinstance(data["tools"], dict):
        logger.error("Invalid tools type")
        issues.append({
            "rule": "type_tools",
            "level": "error",
            "message": "tools must be dict",
        })

    return issues

def validate_business_rules(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate simple orchestration rules

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    tasks = data.get("tasks", [])

    ## At least one task required
    if isinstance(tasks, list) and len(tasks) == 0:
        logger.warning("No tasks defined")
        issues.append({
            "rule": "tasks_empty",
            "level": "warning",
            "message": "No tasks defined",
        })

    return issues

def compute_quality_score(data: Dict[str, Any]) -> float:
    """
        Compute quality score based on prompts

        Args:
            data: Input dictionary

        Returns:
            float
    """

    tasks = data.get("tasks", [])

    if not tasks:
        logger.warning("No tasks for scoring")
        return 0.0

    total_length = 0

    for task in tasks:
        if isinstance(task, dict):
            prompt = task.get("prompt", "")
            if isinstance(prompt, str):
                total_length += len(prompt)

    if total_length == 0:
        return 0.0

    ## Normalize score
    score = min(total_length / 1000.0, 1.0)

    logger.debug(f"Quality score: {score}")

    return score