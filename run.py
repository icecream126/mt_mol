from json.decoder import JSONDecodeError

import dill
import hashlib
from json import dumps
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
import importlib.util
import sys
import traceback
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Union, Tuple
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage
from rdkit import Chem
from utils.args import parse_args, return_API_keys
import utils.utils
from utils.metrics import *
import pandas as pd
from typing import Set
from openai import OpenAI
from typing import List, Tuple
import utils.auc
from utils.task_dicts import get_task_to_condition_dict, get_task_to_dataset_path_dict, get_task_to_score_dict, get_task_to_scientist_prompt_dict, get_task_to_scientist_prompt_with_review_dict, get_task_to_reviewer_prompt_dict, get_task_to_scientist_prompt_with_double_checker_dict, get_task_to_double_checker_prompt_dict, get_task_to_functional_group_dict
from prompts.task_prompts.task_specific_prompts import get_task_specific_prompt

import json
import os
import wandb
import re
import datetime
from dataclasses import dataclass
from pathlib import Path
import logging
import random
import numpy as np
import torch

DOUBLE_CHECKER_COUNT = 0
IN_DOUBLE_CHECKING_PROCESS = False

def save_graphstate(state, path):
    with open(path, "wb") as f:
        dill.dump(dict(state), f)

def load_graphstate(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return dill.load(f)
    else:
        return create_initial_state()

@dataclass
class Config:
    """Configuration class to hold all global settings"""
    task: str
    max_iter: int
    api_num: int
    tool_call_model_name: str
    tool_call_temperature: float
    scientist_model_name: str
    scientist_temperature: float
    reviewer_model_name: str
    reviewer_temperature: float
    double_checker_model_name: str
    double_checker_temperature: float
    run_id: str
    
    def __post_init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_dir = Path(f"./logs/{self.current_time}-{self.run_id}/")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # API Keys
        api_keys = return_API_keys()
        self.openai_api_key = api_keys["OPENAI_API_KEY"]
        self.deepseek_api_key = api_keys[f"DEEPSEEK_API_KEY_{self.api_num}"]
        
        # File paths
        self.log_path = self.log_dir / "log.txt"
        self.unique_smiles_history_log_path = self.log_dir / "unique_smiles_history.txt"
        self.smiles_history_log_path = self.log_dir / "smiles_history.txt"
        self.best_smiles_log_path = self.log_dir / "best_smiles.txt"

class Logger:
    """Centralized logging class"""
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )
        
    def log(self, msg: str):
        """Log a message to both file and console"""
        logging.info(msg)
        with open(self.config.log_path, "a") as f:
            f.write(msg + "\n")
            
    def log_best_smiles(self, msg: str):
        """Log best SMILES to best SMILES log file"""
        with open(self.config.best_smiles_log_path, "a") as f:
            f.write(msg + "\n")
            
    def log_smiles_history(self, msg: str):
        """Log SMILES history"""
        with open(self.config.smiles_history_log_path, "a") as f:
            f.write(msg + "\n")

    def log_unique_smiles_history(self, msg: str):
        """Log unique SMILES history"""
        with open(self.config.unique_smiles_history_log_path, "a") as f:
            f.write(msg + "\n")
            
# --------------------------
# Logging helper
# --------------------------
def log(msg: str):
    log(msg)

def SMILES_history_log(msg: str):
    logger.log_smiles_history(msg)

def BEST_SMILES_log(msg: str):
    logger.log_best_smiles(msg)

def unique_SMILES_history_log(msg: str):
    logger.log_unique_smiles_history(msg)
# --------------------------
# Graph State
# --------------------------
class GraphState(TypedDict):
    """State class for the LangGraph workflow"""
    # Core state
    iteration: int
    max_iterations: int
    task: Union[str, List[str]]
    score: float
    generated_smiles: str
    best_score: float
    best_smiles: str
    double_checker_count: int
    
    # Thinking states
    scientist_think_dict: Dict[str, str]
    reviewer_think_dict: Dict[str, str]
    double_checker_think_dict: Dict[str, str]
    
    # SMILES tracking
    smiles_history: List[str]
    unique_smiles_history: Set[str]
    topk_smiles: List[Tuple[str, float]]
    smiles_scores: List[Tuple[str, float]]
    parsed_smiles: List[str]
    
    # Analysis results
    functional_groups: List[str]
    target_functional_groups: str
    
    # Control flags
    json_output: bool
    in_double_checking_process: bool
    redundant_smiles: bool
    
    # Tool configuration
    basic_tools: List[Dict[str, str]]
    electronic_tools: List[Dict[str, str]]
    fragment_based_tools: List[Dict[str, str]]
    identifier_tools: List[Dict[str, str]]
    other_tools: List[Dict[str, str]]
    structural_tools: List[Dict[str, str]]
    tools_to_use: List[Dict[str, str]]

# Constants
TASK_TO_CONDITION = get_task_to_condition_dict()
TASK_TO_DATASET_PATH = get_task_to_dataset_path_dict()
TASK_TO_SCORING_FUNCTION = get_task_to_score_dict()
TASK_TO_SCIENTIST_PROMPT = get_task_to_scientist_prompt_dict()
TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW = get_task_to_scientist_prompt_with_review_dict()
TASK_TO_REVIEWER_PROMPT = get_task_to_reviewer_prompt_dict()
TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER = get_task_to_scientist_prompt_with_double_checker_dict()
TASK_TO_DOUBLE_CHECKER_PROMPT = get_task_to_double_checker_prompt_dict()
TASK_TO_FUNCTIONAL_GROUP = get_task_to_functional_group_dict()

# --------------------------
# Global Variables
# --------------------------

SMILES = ""
BEST_SMILES = ""

oracle_buffer = []
BEST_TOP_10_AUC_ALL = 0.0
BEST_TOP_10_AUC_NO_1 = 0.0
# smiles_history = set()

class MoleculeError(Exception):
    """Custom exception for molecule-related errors"""
    pass


def run_pubchem_functions(function_names, smiles_list, utils_path="/home/anonymous/chemiloop/utils/tools.py"):
    """
    Run specified functions on a list of SMILES strings and return a structured text summary.

    Args:
        function_names (list of str): Names of functions to run.
        smiles_list (list of dict): List of dicts with key 'smiles'.
        utils_path (str): Path to the module with functions.

    Returns:
        str: Text summary of results per SMILES.
    """
    module_name = "chem_utils"
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = utils_module
    spec.loader.exec_module(utils_module)

    output = []
    # Check if smiles is string or list
    if isinstance(smiles_list, str):
        smiles_list = [{"smiles": smiles_list}]

    for smiles_dict in smiles_list:
        
        smiles = smiles_dict.get("smiles", "")
        if not smiles:
            continue

        output.append(f"=== Results for SMILES: {smiles} ===")
        for object in function_names:
            fn_name = object.get("tool_name", "").lower()
            purpose = object.get("purpose", "")
            if hasattr(utils_module, fn_name):
                func = getattr(utils_module, fn_name)
                try:
                    result = func(smiles)
                    formatted = f"[{fn_name}] - {purpose}\n Result:{str(result).strip()}\n"
                except Exception as e:
                    formatted = f"[{fn_name}] - {purpose} - ERROR: {e}\n"
            else:
                formatted = f"[{fn_name}] - {purpose} - NOT FOUND\n"

            output.append(formatted)

        output.append("")  # Blank line between SMILES blocks
    
    return "\n".join(output)


def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def get_task_name(state: GraphState) -> str:
    """Get the task name from state, handling both string and list cases"""
    return state["task"][0] if isinstance(state["task"], list) else state["task"]

def format_smiles_history(smiles_history: set) -> str:
    """Format SMILES history for display"""
    if not smiles_history:
        return "Currently no history"
    return "\n".join(sorted(smiles_history))

def update_best_score(current_score: float, current_smiles: str, best_score: float, best_smiles: str) -> Tuple[float, str]:
    """Update best score and SMILES if current score is better"""
    if current_score > best_score:
        return current_score, current_smiles
    return best_score, best_smiles

def calculate_metrics(smiles_scores: List[Tuple[str, float]], iteration: int, max_iterations: int) -> Dict[str, float]:
    """Calculate various metrics for the current state"""
    finish = iteration + 1 >= max_iterations
    auc_top10_all, _ = utils.auc.compute_topk_auc(smiles_scores, top_k=10, max_oracle_calls=1000, freq_log=1, buffer_max_idx=iteration+1, finish=finish)
    auc_top1_all, _ = utils.auc.compute_topk_auc(smiles_scores, top_k=1, max_oracle_calls=1000, freq_log=1, buffer_max_idx=iteration+1, finish=finish)
    
    sorted_all = sorted(smiles_scores, key=lambda x: x[1], reverse=True)
    top_10_all = sum(score for _, score in sorted_all[:10]) / 10
    
    return {
        "top_10_avg_score_all": top_10_all,
        "auc_top1_all": auc_top1_all,
        "auc_top10_all": auc_top10_all
    }


def tool_electronic_topological_descriptors(state: GraphState) -> GraphState:
    """Node for selecting tools to use in the workflow"""
    tool_path = "/home/anonymous/chemiloop/tool_jsons/Electronic_Topological_Descriptors.json"
    
    try:
        with open(tool_path, "r") as tool_json:
            tool_specs = json.load(tool_json)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "electronic_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "electronic_tools": []}

    task = get_task_name(state)
    system_prompt = """You are a professional AI chemistry assistant specialized in calculating electronic and topological descriptors from molecular graphs using RDKit.
    Your job is to identify relevant descriptors such as Chi indices, BCUT, Crippen logP/MR, or topological polar surface area that impact task-specific molecular behavior.

    Follow this structured reasoning process step-by-step:

    Step 1. Analyze the molecule design condition which is the goal of the task.
    Step 2. Carefully think about the topological or electronic features that are chemically meaningful for the task.
        - Explain why each feature (e.g., topological index, surface area) matters for the task.
    Step 3. Select appropriate tools that calculate those features.
    Step 4. Output your final answer in the provided JSON format."""



    user_prompt = f"""This is a molecule design condition of the {task} task:
{TASK_TO_CONDITION[task]}
                
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "tools_to_use": [
    {{
      "tool_name": "fr_Ar_OH",
      "purpose": "Detect aromatic hydroxyl groups, similar to those in albuterol."
    }},
    ...
  ]
}}
```"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.tool_call_model_name,
            llm_temperature=config.tool_call_temperature,
            max_retries=10,
            sleep_sec=2,
            tools=tool_specs
        )
        response = raw_response.choices[0].message.content
        tool_json = json.loads(response)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        return {**state, "electronic_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        return {**state, "electronic_tools": []}
        
    
    logger.log(f"\n==== Electronic Topological Descriptors Tool call Node - {state['iteration']} ==")
    logger.log("Prompt to tool call node:")
    logger.log(str(prompt))
    logger.log("\nResponse from tool call node:")
    logger.log(str(response))
    state["electronic_tools"] = tool_json.get("tools_to_use", [])

    with open(f"./logs/{config.current_time}-{config.run_id}/electronic_tools.txt", "a") as f:
        f.write(str(state["electronic_tools"]) + "\n")

    save_graphstate(state, f"{config.log_dir}/graphstate.pkl")

    return {
        **state,
        "electronic_tools": tool_json.get("tools_to_use", []),
    }

def tool_fragment_based_functional_groups(state: GraphState) -> GraphState:
    """Node for selecting tools to use in the workflow"""
    tool_path = "/home/anonymous/chemiloop/tool_jsons/Fragment_Based_Functional_Groups.json"
    
    try:
        with open(tool_path, "r") as tool_json:
            tool_specs = json.load(tool_json)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "fragment_based_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "fragment_based_tools": []}


    task = get_task_name(state)
    system_prompt = """You are a professional AI chemistry assistant specializing in analyzing functional groups and substructures using SMARTS-based RDKit fragment tools.
    Your job is to detect and count specific functional groups that are relevant to the molecule design condition.

    Follow this structured reasoning process step-by-step:

    Step 1. Analyze the molecule design condition which is the goal of the task.
    Step 2. Carefully think about the functional groups or substructures likely to influence the desired molecular property (e.g., hydroxyl, carboxylic acid, halogens).
        - Justify the role of each group (e.g., H-bonding, reactivity, metabolic stability).
    Step 3. Select the tools that detect those groups from your toolkit.
    Step 4. Output your final answer in the provided JSON format."""




    user_prompt = f"""This is a molecule design condition of the {task} task:
{TASK_TO_CONDITION[task]}
                
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "tools_to_use": [
    {{
      "tool_name": "fr_Ar_OH",
      "purpose": "Detect aromatic hydroxyl groups, similar to those in albuterol."
    }},
    ...
  ]
}}
```"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.tool_call_model_name,
            llm_temperature=config.tool_call_temperature,
            max_retries=10,
            sleep_sec=2,
            tools=tool_specs
        )
        response = raw_response.choices[0].message.content
        tool_json = json.loads(response)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "fragment_based_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "fragment_based_tools": []}
        
    
    logger.log(f"\n==== Fragment Based Functional Groups Tool call Node - {state['iteration']} ==")
    logger.log("Prompt to tool call node:")
    logger.log(str(prompt))
    logger.log("\nResponse from tool call node:")
    logger.log(str(response))

    state["fragment_based_tools"] = tool_json.get("tools_to_use", [])

    with open(f"./logs/{config.current_time}-{config.run_id}/fragment_based_tools.txt", "a") as f:
        f.write(str(state["fragment_based_tools"]) + "\n")
    
    save_graphstate(state, f"{config.log_dir}/graphstate.pkl")

    return {
        **state,
        "fragment_based_tools": tool_json.get("tools_to_use", []),
    }

def tool_other_descriptors(state: GraphState) -> GraphState:
    """Node for selecting tools to use in the workflow"""
    tool_path = "/home/anonymous/chemiloop/tool_jsons/Other_Descriptors.json"
    
    try:
        with open(tool_path, "r") as tool_json:
            tool_specs = json.load(tool_json)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "other_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "other_tools": []}


    task = get_task_name(state)
    system_prompt = """You are a professional AI chemistry assistant that provides advanced or miscellaneous molecular descriptors that don’t fall into standard categories.
    Your job is to analyze complex molecular tasks and identify any useful descriptors that might be overlooked by traditional tools.

    Follow this structured reasoning process step-by-step:

    Step 1. Analyze the molecule design condition which is the goal of the task.
    Step 2. Based on expert-level chemical knowledge, think about the unusual or rarely used descriptors (e.g., plane of best fit, MQNs) that may aid this task.
        - Explain the niche or advanced utility of each descriptor.
    Step 3. Choose the most appropriate tools from your toolkit.
    Step 4. Output your final answer in the provided JSON format."""




    user_prompt = f"""This is a molecule design condition of the {task} task:
{TASK_TO_CONDITION[task]}
                
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "tools_to_use": [
    {{
      "tool_name": "fr_Ar_OH",
      "purpose": "Detect aromatic hydroxyl groups, similar to those in albuterol."
    }},
    ...
  ]
}}
```"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.tool_call_model_name,
            llm_temperature=config.tool_call_temperature,
            max_retries=10,
            sleep_sec=2,
            tools=tool_specs
        )
        response = raw_response.choices[0].message.content
        tool_json = json.loads(response)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "other_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "other_tools": []}
        
    
    logger.log(f"\n==== Other Descriptors Tool call Node - {state['iteration']} ==")
    logger.log("Prompt to tool call node:")
    logger.log(str(prompt))
    logger.log("\nResponse from tool call node:")
    logger.log(str(response))

    state["other_tools"] = tool_json.get("tools_to_use", [])

    with open(f"./logs/{config.current_time}-{config.run_id}/other_tools.txt", "a") as f:
        f.write(str(state["other_tools"]) + "\n")
    
    save_graphstate(state, f"{config.log_dir}/graphstate.pkl")

    return {
        **state,
        "other_tools": tool_json.get("tools_to_use", []),
    }

def tool_structural_descriptors(state: GraphState) -> GraphState:
    """Node for selecting tools to use in the workflow"""
    tool_path = "/home/anonymous/chemiloop/tool_jsons/Structural_Descriptors.json"
    
    try:
        with open(tool_path, "r") as tool_json:
            tool_specs = json.load(tool_json)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "structural_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "structural_tools": []}

    task = get_task_name(state)
    system_prompt = """You are a professional AI chemistry assistant with expertise in analyzing molecular topology and structure using RDKit-based atom and bond count tools.
    Your job is to identify structural features such as number of atoms, rings, rotatable bonds, or stereocenters that influence the design task.

    Follow this structured reasoning process step-by-step:

    Step 1. Analyze the molecule design condition which is the goal of the task.
    Step 2. Carefully think about which **structural characteristics** are important for this task.
        - Examples: rigidity, size, flexibility, 3D complexity.
    Step 3. Select relevant tools to compute those descriptors.
    Step 4. Output your final answer in the provided JSON format."""


    user_prompt = f"""This is a molecule design condition of the {task} task:
{TASK_TO_CONDITION[task]}
                
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "tools_to_use": [
    {{
      "tool_name": "fr_Ar_OH",
      "purpose": "Detect aromatic hydroxyl groups, similar to those in albuterol."
    }},
    ...
  ]
}}
```"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.tool_call_model_name,
            llm_temperature=config.tool_call_temperature,
            max_retries=10,
            sleep_sec=2,
            tools=tool_specs
        )
        response = raw_response.choices[0].message.content
        tool_json = json.loads(response)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "structural_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "structural_tools": []}
        
    
    logger.log(f"\n==== Structural Descriptors Tool call Node - {state['iteration']} ==")
    logger.log("Prompt to tool call node:")
    logger.log(str(prompt))
    logger.log("\nResponse from tool call node:")
    logger.log(str(response))

    state["structural_tools"] = tool_json.get("tools_to_use", [])

    with open(f"./logs/{config.current_time}-{config.run_id}/structural_tools.txt", "a") as f:
        f.write(str(state["structural_tools"]) + "\n")

    save_graphstate(state, f"{config.log_dir}/graphstate.pkl")

    return {
        **state,
        "structural_tools": tool_json.get("tools_to_use", []),
    }

def tool_identifiers_and_representations(state: GraphState) -> GraphState:
    """Node for selecting tools to use in the workflow"""
    tool_path = "/home/anonymous/chemiloop/tool_jsons/Identifiers_and_Representations.json"
    
    try:
        with open(tool_path, "r") as tool_json:
            tool_specs = json.load(tool_json)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "identifier_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "identifier_tools": []}

    task = get_task_name(state)
    system_prompt = """You are a professional AI chemistry assistant specialized in resolving molecular identifiers and representations using RDKit-based tools.
    Your job is to identify how to retrieve standardized molecular information such as CIDs, InChI, and canonical SMILES for downstream processing.

    Follow this structured reasoning process step-by-step:

    Step 1. Analyze the molecule design condition which is the goal of the task.
    Step 2. Parse **list of all valid SMILES strings** mentioned anywhere in the user prompt and output them in the provided JSON format.
    Step 3. Based on your chemical knowledge, explain why standardizing identifiers and resolving canonical formats might be important for this task.
        - E.g., checking uniqueness, linking to external data, verifying molecular identity.
    Step 4. Choose **as many tools as necessary** from the identifier toolset that help you access consistent molecular representations or external references.
    Step 5. Output your final answer in the provided JSON format."""



    user_prompt = f"""This is a molecule design condition of the {task} task:
{TASK_TO_CONDITION[task]}
                
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "parsed_smiles": [
    {{
      "smiles": "Parsed SMILES string",
    }}  ,
    ...
  ],
  "tools_to_use": [
    {{
      "tool_name": "fr_Ar_OH",
      "purpose": "Detect aromatic hydroxyl groups, similar to those in albuterol."
    }},
    ...
  ]
}}
```"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.tool_call_model_name,
            llm_temperature=config.tool_call_temperature,
            max_retries=10,
            sleep_sec=2,
            tools=tool_specs
        )
        response = raw_response.choices[0].message.content
        tool_json = json.loads(response)
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "identifier_tools": []}
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        traceback.print_exc()
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return {**state, "identifier_tools": []}
    
    logger.log(f"\n==== Identifiers and Representations Tool call Node - {state['iteration']} ==")
    logger.log("Prompt to tool call node:")
    logger.log(str(prompt))
    logger.log("\nResponse from tool call node:")
    logger.log(str(response))

    state["parsed_smiles"] = tool_json.get("parsed_smiles", [])
    state["identifier_tools"] = tool_json.get("tools_to_use", [])

    with open(f"./logs/{config.current_time}-{config.run_id}/identifier_tools.txt", "a") as f:
        f.write(str(state["identifier_tools"]) + "\n")

    save_graphstate(state, f"{config.log_dir}/graphstate.pkl")

    state["tools_to_use"] = state["electronic_tools"] + state["fragment_based_tools"] + state["other_tools"] + state["structural_tools"]
    state["target_functional_groups"] = run_pubchem_functions(state["tools_to_use"]+state["identifier_tools"], state["parsed_smiles"])



    return {
        **state,
        "identifier_tools": tool_json.get("tools_to_use", []),
    }

def retrieval_node(state: GraphState) -> GraphState:
    
    # Load pre-computed top-k dataset by task
    # TODO: Extend this to entire train dataset
    # TODO: Add more tasks 
    # TODO: If not pre-computed, compute the top-k dataset
    task = state["task"][0] if type(state["task"])==list else state["task"]
    dataset_path = TASK_TO_DATASET_PATH.get(task, None)
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    topk_smiles = []
    for data in dataset:
        smiles = data["smiles"]
        score = data.get(f"{task}_score", 0)
        topk_smiles.append((smiles, score))

    print(f"[Retrieval Node] Retrieved SMILES: ", str(topk_smiles))
    
    save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
    return {
        **state,
        "topk_smiles": topk_smiles,
    }

def scientist_node(state: GraphState) -> GraphState:
    if state["tools_to_use"] == []:
        state["tools_to_use"] = state["electronic_tools"] + state["fragment_based_tools"] + state["other_tools"] + state["structural_tools"]

    """Generate new SMILES based on current state"""
    global SMILES
    
    logger.log("Scientist node is thinking...")
    
    # Format SMILES history and top-k SMILES
    smiles_history = format_smiles_history(state["unique_smiles_history"])
    topk_smiles = utils.utils.format_topk_smiles(state["topk_smiles"])
    
    # Get task name and functional groups
    task_name = get_task_name(state)

    system_prompt = f"You are a skilled chemist."
    # TODO: Define target functional groups by analyzing the state["parsed_smiles"] via state["tools_to_use"]
    # TODO: Use the function utils.utils.run_pubchem_functions(function_names, smiles) to get the functional groups
    
    
    # state["target_functional_groups"] = run_pubchem_functions(state["tools_to_use"]+state["identifier_tools"], state["parsed_smiles"])

    # Get appropriate prompt based on state
    if IN_DOUBLE_CHECKING_PROCESS:
        user_prompt = get_task_specific_prompt(
            task_name=task_name,
            prompt_type="scientist_with_double_checker",
            functional_groups=state["functional_groups"],
            previous_thinking=state["scientist_think_dict"],
            previous_smiles=state["generated_smiles"],
            double_checker_feedback=state["double_checker_think_dict"],
            smiles_history=smiles_history, 
            target_functional_groups=state["target_functional_groups"]
        )
    elif state.get("reviewer_think_dict"):

        user_prompt = get_task_specific_prompt(
            task_name=task_name,
            prompt_type="scientist_with_review",
            functional_groups=state["functional_groups"],
            target_functional_groups=state["target_functional_groups"],
            scientist_think_dict=state["scientist_think_dict"],
            reviewer_feedback_dict=state["reviewer_think_dict"],
            previous_smiles=state["generated_smiles"],
            score=state["score"],
            smiles_history=smiles_history,
            topk_smiles=topk_smiles, 
        )
    else:
        user_prompt = get_task_specific_prompt(
            task_name=task_name,
            prompt_type="scientist",
            topk_smiles=topk_smiles,
            target_functional_groups=state["target_functional_groups"]
        )
    
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.scientist_model_name,
            llm_temperature=config.scientist_temperature,
            max_retries=10,
            sleep_sec=2,
        )
        response = raw_response.choices[0].message.content
        
                # Try parsing response
        parsed = json.loads(response)

        # Fill in missing steps and consistency
        default_keys = ["step1", "step2", "step3", "smiles"]
        state["scientist_think_dict"] = {key: parsed.get(key, "") for key in default_keys}
        
        if "smiles" not in state["scientist_think_dict"]:
            logger.log("No SMILES generated in response")
        
        state["generated_smiles"] = state["scientist_think_dict"]["smiles"]

        logger.log(f"\n==== Scientist Node - {state['iteration']} ===")
        logger.log("Prompt to scientist node:")
        logger.log(str(prompt))
        logger.log("\nResponse from scientist node:")
        logger.log(str(response))

        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in double checker node: {e}")
        traceback.print_exc()
        state["scientist_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
            "smiles": "INVALID"
        }
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state
    except Exception as e:
        logger.log(f"General error in double checker node: {str(e)}")
        traceback.print_exc()
        state["scientist_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
            "smiles": "INVALID"
        }
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state

def double_checker_node(state: GraphState) -> GraphState:
    """Verify scientist's reasoning and SMILES generation"""
    logger.log("Double checker node is thinking...")
    
    # Get task name
    task_name = get_task_name(state)
    if not validate_smiles(state["generated_smiles"]):
        state["functional_groups"] = "No functional groups because the SMILES is invalid."
    else:
        state["functional_groups"] = run_pubchem_functions(state["tools_to_use"], state["generated_smiles"])

    system_prompt = f"You are a meticulous double-checker LLM. Your task is to verify whether each step of the scientist’s reasoning is chemically valid and faithfully and logically reflected in the final SMILES string."  # (include full system instructions here)
    
    # Get double checker prompt
    user_prompt = get_task_specific_prompt(
        task_name=task_name,
        prompt_type="double_checker",
        functional_groups=state["functional_groups"],
        target_functional_groups=state["target_functional_groups"],
        thinking=state["scientist_think_dict"],
        smiles=state["generated_smiles"]
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:

        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.double_checker_model_name,
            llm_temperature=config.double_checker_temperature,
            max_retries=10,
            sleep_sec=2,
        )
        response = raw_response.choices[0].message.content
        logger.log(f"Double checker node response: {response}")
        

        # Try parsing response
        parsed = json.loads(response)

        # Fill in missing steps and consistency
        default_keys = ["step1", "step2", "step3", "consistency"]
        state["double_checker_think_dict"] = {key: parsed.get(key, "") for key in default_keys}


        logger.log(f"\n==== Double Checker Node - {state['iteration']} ==")
        logger.log("Prompt to double checker node:")
        logger.log(str(prompt))
        logger.log("\nResponse from double checker node:")
        logger.log(str(response))

        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in double checker node: {e}")
        traceback.print_exc()
        state["double_checker_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
            "consistency": "Inconsistent"
        }
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state

    except Exception as e:
        logger.log(f"General error in double checker node: {str(e)}")
        traceback.print_exc()
        state["double_checker_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
            "consistency": "Inconsistent"
        }
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state

def reviewer_node(state: GraphState) -> GraphState:
    state["iteration"] += 1
    """Review and score generated SMILES"""
    logger.log("Reviewer node is thinking...")
    
    # Get task name and scoring function
    task_name = get_task_name(state)
    # Validate SMILES
    if not validate_smiles(state["generated_smiles"]):
        logger.log("Invalid SMILES generated")
        state["score"] = 0.0
        state["functional_groups"] = "No functional groups because the SMILES is invalid."
        
    else:
        scoring_function = TASK_TO_SCORING_FUNCTION[task_name]
        state["score"] = scoring_function(state["generated_smiles"])
        state["functional_groups"] = run_pubchem_functions(state["tools_to_use"], state["generated_smiles"])
    
    if not isinstance(state["generated_smiles"], str):
        logger.log("Generated SMILES is not a string, setting to empty string.")
        state["generated_smiles"] = ""

    state["smiles_scores"].append((state["generated_smiles"], state["score"]))
    
    state["unique_smiles_history"].add(state["generated_smiles"])
    state["smiles_history"].append([state["generated_smiles"]])

    unique_SMILES_history_log(str(state["unique_smiles_history"]))
    SMILES_history_log(str(state["generated_smiles"]+","+str(state["score"])))
    
    # Update best score if needed
    state["best_score"], state["best_smiles"] = update_best_score(state["score"], state["generated_smiles"], state["best_score"], state["best_smiles"])
    
    # Calculate metrics
    metrics = calculate_metrics(state["smiles_scores"], state["iteration"], state["max_iterations"])
    wandb.log(metrics)
    
    system_prompt="You are a rigorous chemistry reviewer.\n"
    # Get reviewer prompt
    user_prompt = get_task_specific_prompt(
        task_name=task_name,
        prompt_type="reviewer",
        functional_groups=state["functional_groups"],
        target_functional_groups=state["target_functional_groups"],
        scientist_think_dict=state["scientist_think_dict"],
        score=state["score"],
    )   
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=llm,
            llm_type=config.reviewer_model_name,
            llm_temperature=config.reviewer_temperature,
            max_retries=10,
            sleep_sec=2,
        )
        response = raw_response.choices[0].message.content
        logger.log(f"Reviewer node response: {response}")

        # Try parsing response
        parsed = json.loads(response)

        # Fill in missing steps and consistency
        default_keys = ["step1", "step2", "step3"]
        state["reviewer_think_dict"] = {key: parsed.get(key, "") for key in default_keys}



        logger.log(f"\n==== Reviewer Node - {state['iteration']} ==")
        logger.log("Prompt to reviewer node:")
        logger.log(str(prompt))
        logger.log("\nResponse from reviewer node:")
        logger.log(str(response))

        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state
    except JSONDecodeError as e:
        logger.log(f"JSONDecodeError in double checker node: {e}")
        traceback.print_exc()
        state["reviewer_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
        }
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state
    except Exception as e:
        logger.log(f"General error in double checker node: {str(e)}")
        traceback.print_exc()
        state["reviewer_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
        }
        save_graphstate(state, f"{config.log_dir}/graphstate.pkl")
        return state
# Conditional entry: only first iteration runs tools
def entry_router(state: GraphState) -> str:
    return "tool_electronic" if state["iteration"] == 0 else "scientist_node"

def should_continue(state: GraphState) -> str:
    print(f"Iteration: {state['iteration']}")
    if state["iteration"] >= state["max_iterations"]:
        return False
    else:
        return True


def route_after_double_checker(state: GraphState) -> str:
    global DOUBLE_CHECKER_COUNT
    global IN_DOUBLE_CHECKING_PROCESS
    DOUBLE_CHECKER_COUNT += 1
    if state["double_checker_think_dict"]["consistency"].strip().lower() == "consistent":
        logger.log("Double checker consistency is consistent, returning to reviewer node")
        DOUBLE_CHECKER_COUNT = 0
        IN_DOUBLE_CHECKING_PROCESS = False
        return "reviewer_node"
    elif DOUBLE_CHECKER_COUNT >= 3:
        logger.log("Double checker count >= 3, returning to reviewer node. ")
        IN_DOUBLE_CHECKING_PROCESS = False
        DOUBLE_CHECKER_COUNT = 0
        return "reviewer_node"
    else:
        logger.log(f"Double checker consistency is inconsistent, returning to scientist node.\nDouble checker count: {DOUBLE_CHECKER_COUNT}")
        IN_DOUBLE_CHECKING_PROCESS = True
        return "scientist_node"

def create_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # Add all tool expert nodes
    builder.add_node("tool_electronic", tool_electronic_topological_descriptors)
    builder.add_node("tool_fragment", tool_fragment_based_functional_groups)
    builder.add_node("tool_identifier", tool_identifiers_and_representations)
    builder.add_node("tool_other", tool_other_descriptors)
    builder.add_node("tool_structural", tool_structural_descriptors)

    # Add core reasoning nodes
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("scientist_node", scientist_node)
    builder.add_node("double_checker_node", double_checker_node)
    builder.add_node("reviewer_node", reviewer_node)

    # Entry point
    builder.set_conditional_entry_point(entry_router)

    # # Parallel fan-out from tool_basic to all other tool experts
    
    builder.add_edge("tool_electronic", "tool_fragment")
    builder.add_edge("tool_fragment", "tool_other")
    builder.add_edge("tool_other", "tool_structural")
    builder.add_edge("tool_structural", "tool_identifier")
    builder.add_edge("tool_identifier", "retrieval_node")
    builder.add_edge("retrieval_node", "scientist_node")
    builder.add_edge("scientist_node", "double_checker_node")
    builder.add_conditional_edges("double_checker_node", route_after_double_checker)
    builder.add_conditional_edges("reviewer_node", should_continue, {True: "scientist_node", False: END})


    
    return builder.compile()

def create_initial_state() -> GraphState:
    """Create the initial state for the graph"""
    return {
            "iteration": 0,
            "max_iterations": config.max_iter,
            "scientist_think_dict": {},
            "reviewer_think_dict": {},
            "double_checker_think_dict": {},
            "task": config.task,
            "score": 0.0,
            "functional_groups": "",
            "generated_smiles": "",
            "json_output": True,
            "topk_smiles": [],
            "smiles_scores": [],
            "scientist_message": [],
            "reviewer_message": [],
            "smiles_history": [],
            "unique_smiles_history": set(),
            "redundant_smiles": False,
            "best_score": 0.0,
            "best_smiles": "",
            "target_functional_groups": "",
            "parsed_smiles": [],
            "basic_tools": [],
            "electronic_tools": [],
            "fragment_based_tools": [],
            "identifier_tools": [],
            "other_tools": [],
            "structural_tools": [],
            "tools_to_use": [],
            "checkpoint_id": hashlib.md5(config.task.encode()).hexdigest(),
            "thread_id": 0,
            "checkpoint_ns": "",
    }

if __name__ == "__main__":
    args = parse_args()
    
    seed = args.seed
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Initialize wandb
    wandb.init(project="test", name=args.task[0] if isinstance(args.task, list) else args.task, config=vars(args))#  , mode="disabled")

    
    config = Config(
        task=args.task[0] if isinstance(args.task, list) else args.task,
        max_iter=args.max_iter,
        api_num=args.api_num,
        tool_call_model_name=args.tool_call_model_name,
        tool_call_temperature=args.tool_call_temperature,
        scientist_model_name=args.scientist_model_name,
        scientist_temperature=args.scientist_temperature,
        reviewer_model_name=args.reviewer_model_name,
        reviewer_temperature=args.reviewer_temperature,
        double_checker_model_name=args.double_checker_model_name,
        double_checker_temperature=args.double_checker_temperature,
        run_id=wandb.run.id,

    )

    logger = Logger(config)

    llm = OpenAI(
        api_key=config.deepseek_api_key,  # Replace with your actual API key
        base_url="https://api.deepseek.com"
    )
    
    smiles_history_LOG_PATH = f"{config.log_dir}smiles_history.txt"
    if not args.graph_state_path:
        graph_state_path = f"{config.log_dir}graphstate.pkl"
    else:
        graph_state_path = args.graph_state_path

    print("smiles history path: ", smiles_history_LOG_PATH)
    print("graph state path: ", graph_state_path)

    try:
        if args.graph_state_path:
            logger.log("Loading graph state")
            graph_state = load_graphstate(graph_state_path)
        else:
            logger.log("Creating graph state")
            graph_state = create_initial_state()
        
        graph = create_graph()

        final_state = graph.invoke(graph_state, {"recursion_limit": 9999})
        
        # Print final state
        logger.log("\nFinal State:")
        for k, v in final_state.items():
            logger.log(f"{k}: {v}")
            
    except Exception as e:
        logger.log(f"Error in main execution: {e}")
    finally:
        wandb.finish()
