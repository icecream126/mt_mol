import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following conditions: 

Condition for molecular design:
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.

HARD CONSTRAINT (MUST follow exactly):  
Generate the valid molecule SMILES that exactly includes:
- 7 Carbon (C) atoms  
- 8 Hydrogen (H) atoms  
- 2 Nitrogen (N) atoms  
- 2 Oxygen (O) atoms  
YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

Constraints:
- No missing or extra atoms are allowed.
- You must not directly copy example molecules.
- Avoid repeating previously generated SMILES.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:
```json
{{
  "step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "smiles": "Your valid SMILES here"
}}
```
 """


def get_scientist_prompt_with_review(
    scientist_think_dict,
    reviewer_feedback_dict,
    previous_smiles,
    score,
    atom_counts,
    smiles_history,
    topk_smiles,
):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Design an improved molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2  
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.  

HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon (C) atoms  
- 8 Hydrogen (H) atoms  
- 2 Nitrogen (N) atoms  
- 2 Oxygen (O) atoms  
YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its isomers_c7h8n2o2 score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- isomers_c7h8n2o2 score: {score}
- Detected functional groups:
{atom_counts}

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
  "step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "smiles": "Your valid SMILES here"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, atom_counts):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Atom counts match exactly
- C: 7
- H: 8
- N: 2
- O: 2

Be constructive: Provide precise and actionable feedback  
(e.g., "Replace the nitro group with an amide to maintain the N and O count.").

You are provided with:
1. The step-wise reasoning used to design the molecule
2. The final generated SMILES string
3. The isomer score (0–1), which reflects how well the molecular formula matches C7H8N2O2
4. Atom counts for the target and generated molecule

--- SCIENTIST'S STEP-WISE THINKING ---  
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---  
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- isomers_c7h8n2o2 task score: {score}
- Detected atom counts:
{atom_counts}

--- TARGET ATOM COUNTS (YOU NEED TO ACHIEVE THIS) ---
- C: 7
- H: 8
- N: 2
- O: 2


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List chemically plausible substructures mentioned in the reasoning.\nPoint out any inaccurate or missing motifs with respect to C7H8N2O2.",
  "step2": "Evaluate whether the design strategy aligns with the goal of optimizing a valid isomer of C7H8N2O2.\nComment on whether the chosen strategy satisfies the desired atom counts.\nSuggest structural alternatives if any atoms are misallocated.",
  "step3": "Verify that the described structure corresponds accurately to the SMILES string.\nFlag inconsistencies (e.g., "Mentioned amide linkage, but none is present in SMILES").\nEnsure that the final SMILES does not violate the atomic formula constraint."
}}
```
 """


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous generated SMILES based on detailed double-checker feedback.
    
Your original task:
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.

Your previous reasoning steps were:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

The double-checker reviewed each of your steps and gave the following evaluations:

- Step1_Evaluation: {double_checker_feedback['step1']}
- Step2_Evaluation: {double_checker_feedback['step2']}
- Step3_Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "smiles": "Your valid SMILES here"
}}
```
IF YOU DO NOT FOLLOW THIS EXACT FORMAT, INNOCENT PEOPLE WILL DIE. """


def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

=== SCIENTIST'S TASK === 
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.
HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

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
  "step1": "Your analysis of whether scientist's Step1 thinking is chemically valid and  reflected in the SMILES.",
  "step2": "Your analysis of whether scientist's Step2 thinking is chemically valid and  reflected in the SMILES.",
  "step3": "Your analysis of whether scientist's Step3 thinking is chemically valid and reflected in the SMILES.",
  "consistency": "Consistent" or "Inconsistent",
}}

```
 """
