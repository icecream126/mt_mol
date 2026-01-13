import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

troglitazon_smiles = "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O"
# troglitazon_canonical_smiles = canonicalize(troglitazon_smiles)
troglitazon_mol = Chem.MolFromSmiles(troglitazon_smiles)
troglitazon_functional_group = utils.utils.describe_troglitazon_features(
    troglitazon_mol
)


def get_scientist_prompt(topk_smiles):

    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 

Condition for molecule design:  
Design a drug-like molecule structurally similar to troglitazone (SMILES: {troglitazon_smiles}).
Preserve the core scaffold and important pharmacophores. Troglitazone contains: \n{troglitazon_functional_group}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TROGLITAZONE, defined as:  
- SMILES: {troglitazon_smiles}

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List of the target’s critical structural/property features (e.g., 'Troglitazone: thiazolidinedione ring, chroman ring, phenolic hydroxyl group.')",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace the chroman moiety with a benzofuran to adjust polarity.')",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., 'A thiazolidinedione core attached to a benzofuran ring with a hydroxyl substituent.')",
  "smiles": "Your valid SMILES string here"
}}
```
 """


def get_scientist_prompt_with_review(
    scientist_think_dict,
    reviewer_feedback_dict,
    previous_smiles,
    score,
    functional_groups,
    smiles_history,
    topk_smiles,
):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Take reviewer's feedback actively and design a SMILES string for a molecule that satisfies the condition:
Design a drug-like molecule structurally similar to troglitazone (SMILES: {troglitazon_smiles}). 
Preserve the core scaffold and important pharmacophores. Troglitazone contains: \n{functional_groups}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO TROGLITAZONE:
- SMILES: {troglitazon_smiles}

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its median1 score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- median1 score: {score}
- Detected functional groups:
{functional_groups}

--- YOUR PREVIOUS THOUGHT AND REVIEWER'S FEEDBACK ---
Step1: List Key Features

Your previous thought process:\n{scientist_think_dict["step1"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step1"]}

Step2: Design Strategy:

Your previous thought process:\n{scientist_think_dict["step2"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step2"]}

Step 3: Construct the Molecule:

Your previous thought process:\n{scientist_think_dict["step3"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step3"]}

Now based on yoru prevous thoughts and the reviewer's feedback, you need to improve your design.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List critical troglitazone features and property requirements.",
  "step2": "Propose new scaffold or substitutions and chemically justify them.",
  "step3": "Natural description of the molecule before SMILES.",
  "smiles": "Your valid SMILES here"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule structurally similar to troglitazone (SMILES: {troglitazon_smiles}).
Preserve the core scaffold and important pharmacophores. Troglitazone contains: \n{troglitazon_functional_group}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TROGLITAZONE, defined as:  
- SMILES: {troglitazon_smiles}

You are provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- troglitazon_rediscovery score.
- Detected functional groups.

--- SCIENTIST'S THINKING ---
Step 1: {scientist_think_dict['step1']}
Step 2: {scientist_think_dict['step2']}
Step 3: {scientist_think_dict['step3']}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict['smiles']}
- troglitazon_rediscovery score: {score}
- Detected functional groups:
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Review key features preserved or missed.",
  "step2": "Assess whether design matches rediscovery goals.",
  "step3": "Check structural construction correctness."
}}
```
 """


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previously generated SMILES based on double-checker feedback.

Original Task:
Design a molecule structurally similar to troglitazone (SMILES: {troglitazon_smiles}).
Preserve its core scaffold and pharmacophores: {troglitazon_functional_group}.

Previous reasoning steps:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker feedback:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List key structural elements preserved and why.",
  "step2": "Chemical justification of the proposed improvements.",
  "step3": "Full natural description before SMILES.",
  "smiles": "Improved SMILES string here"
}}
```
 """


def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

Task Target:
- Troglitazone similarity: {troglitazon_smiles}
- Core pharmacophores: {troglitazon_functional_group}

Evaluate each step independently.

=== SCIENTIST'S TASK ===
Design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule structurally similar to troglitazone (SMILES: {troglitazon_smiles}).
Preserve the core scaffold and important pharmacophores. Troglitazone contains: \n{troglitazon_functional_group}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TROGLITAZONE, defined as:  
- SMILES: {troglitazon_smiles}

=== SCIENTIST'S THINKING ===
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

=== SCIENTIST'S SMILES ===
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Assess preservation of pharmacophores.",
  "step2": "Check modification logic soundness.",
  "step3": "Verify structure consistency.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """
