"""Task-specific prompts for different molecular design tasks"""

from typing import Dict, Any
from .base_templates import (
    get_base_scientist_prompt,
    get_base_scientist_prompt_with_review,
    get_base_reviewer_prompt,
    get_base_scientist_prompt_with_double_checker_review,
    get_base_double_checker_prompt
)
from utils.task_dicts import get_task_to_condition_dict

TASK_CONDITIONS = get_task_to_condition_dict()

def get_task_specific_prompt(task_name: str, prompt_type: str, **kwargs) -> str:
    """Get task-specific prompt based on task name and prompt type"""
    if task_name not in TASK_CONDITIONS:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Get base template
    if prompt_type == "scientist":
        return get_base_scientist_prompt(
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "scientist_with_review":
        return get_base_scientist_prompt_with_review(
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "reviewer":
        return get_base_reviewer_prompt(
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "scientist_with_double_checker":
        return get_base_scientist_prompt_with_double_checker_review(
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "double_checker":
        return get_base_double_checker_prompt(
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")