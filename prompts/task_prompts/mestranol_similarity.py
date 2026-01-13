import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

mestranol_smiles = "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1"
# mestranol_canonical_smiles = canonicalize(mestranol_smiles)
mestranol_mol = Chem.MolFromSmiles(mestranol_smiles)
mestranol_functional_group = utils.utils.describe_albuterol_features(mestranol_mol)


def get_scientist_prompt(topk_smiles):

    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 

Condition for molecule design:  
Design a drug-like molecule structurally similar to mestranol (SMILES: {mestranol_smiles}). 
Preserve the core scaffold and key functional groups. Mestranol contains: {mestranol_functional_group}.
  
IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO MESTRANOL, defined as:  
- SMILES: {mestranol_smiles}

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
  "step1": "List of the target’s critical structural/property features (e.g., 'Mestranol: steroid core, ethinyl group, methoxy group')\nIf property-based, specify requirements (e.g., 'high lipophilicity').",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace methoxy with hydroxyl for better metabolism').\n Justify each change chemically (e.g., 'Improves hydrophilicity while retaining key interactions').",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., 'A steroid backbone with an ethynyl group at position 17 and hydroxyl group at position 3.')",
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
Design a drug-like molecule structurally similar to mestranol (SMILES: {mestranol_smiles}).
Preserve the core scaffold and key functional groups. 
Mestranol contains the following functional groups: {mestranol_functional_group}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO MESTRANOL:
- SMILES: {mestranol_smiles}

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its mestranol_similarity score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- mestranol_similarity score: {score}
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
  "step1": "List critical mestranol features and structural requirements.",
  "step2": "Propose scaffold or decoration changes with chemical justifications.",
  "step3": "Natural language description of the final structure before SMILES.",
  "smiles": "Your improved and valid SMILES string"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule structurally similar to mestranol (SMILES: {mestranol_smiles}).
Preserve the core scaffold and key functional groups. Mestranol contains: {mestranol_functional_group}.
  
IMPORTANT CONSTRAINT:  
MUST NOT GENERATE A MOLECULE IDENTICAL TO MESTRANOL, defined as:  
- SMILES: {mestranol_smiles}

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. mestranol_similarity score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- mestranol_similarity score: {score}
- Functional groups detected:
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Review preservation of mestranol’s critical features.",
  "step2": "Check if design strategy is chemically sound and meets similarity objectives.",
  "step3": "Check full structure correctness and adherence to task."
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
Design a molecule structurally similar to mestranol (SMILES: {mestranol_smiles}).
Preserve its core scaffold and key functional groups ({mestranol_functional_group}).

Your previous reasoning steps:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-Checker Feedback:
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
  "step1": "List mestranol critical features and property constraints.",
  "step2": "Propose improvements to structure meeting chemical and task goals.",
  "step3": "Describe the new molecule naturally before SMILES.",
  "smiles": "Your improved and chemically valid SMILES"
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

Task target:
- Similarity to mestranol: {mestranol_smiles}
- Preserve critical groups: {mestranol_functional_group}

=== SCIENTIST'S TASK ===
Design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule structurally similar to mestranol (SMILES: {mestranol_smiles}).
Preserve the core scaffold and key functional groups. Mestranol contains: {mestranol_functional_group}.
  
IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO MESTRANOL, defined as:  
- SMILES: {mestranol_smiles}

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
  "step1": "Consistency check for core feature preservation.",
  "step2": "Consistency check for scaffold or modification logic.",
  "step3": "Consistency check for full structure assembly.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """
