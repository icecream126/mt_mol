def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition:
Create an isomer of molecular formula **C9H10N2O2PF2Cl**.

HARD CONSTRAINT (MUST follow exactly): 
Generate the valid molecule SMILES that exactly includes:
  - 9 Carbon (C) atoms
  - 10 Hydrogen (H) atoms
  - 2 Nitrogen (N) atoms
  - 2 Oxygen (O) atoms
  - 1 Phosphorus (P) atom
  - 2 Fluorine (F) atoms
  - 1 Chlorine (Cl) atom
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

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List the required atom counts exactly (C9H10N2O2PF2Cl).",
  "step2": "Describe your design strategy to arrange the atoms chemically plausibly.",
  "step3": "Describe the structure of your designed molecule in natural language before SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
"""

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Revise your design based on reviewer feedback to better match the molecular formula C9H10N2O2PF2Cl.
Your task is to design a SMILES string for a molecule that satisfies the following condition:
Create an isomer of molecular formula C9H10N2O2PF2Cl.

HARD CONSTRAINT (MUST follow exactly): 
Generate the valid molecule SMILES that exactly includes:
  - 9 Carbon (C) atoms
  - 10 Hydrogen (H) atoms
  - 2 Nitrogen (N) atoms
  - 2 Oxygen (O) atoms
  - 1 Phosphorus (P) atom
  - 2 Fluorine (F) atoms
  - 1 Chlorine (Cl) atom
YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its isomer score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---
MOLECULE SMILES: {previous_smiles}
- isomers_c9h10n2o2pf2cl score: {score}
- functional groups detected: 
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
  "step1": "List the atom counts correctly: C9H10N2O2PF2Cl.",
  "step2": "Propose a corrected atom arrangement strategy.",
  "step3": "Describe the final molecule clearly in natural language.",
  "smiles": "Your improved SMILES string here"
}}
```
"""

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Atom counts match exactly
- C: 9
- H: 10
- N: 2
- O: 2
- P: 1
- F: 2
- Cl: 1 

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. Isomer score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- isomers_c9h10n2o2pf2cl score: {score}
- Atom counts detected:
{functional_groups}

--- TARGET ATOM COUNTS (YOU NEED TO ACHIEVE THIS) ---
- C: 9
- H: 10
- N: 2
- O: 2
- P: 1
- F: 2
- Cl: 1 


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Assess whether atom counts were correctly listed and matched.",
  "step2": "Evaluate the chemical plausibility of the design strategy.",
  "step3": "Check if structure description matches the SMILES produced."
}}
```
# """

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous molecule based on detailed double-checker feedback.

Original Objective:
- Exact atom counts: C9H10N2O2PF2Cl.
- Chemically valid structure.

Your Previous Reasoning:
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
  "step1": "Reconfirm and explicitly list atom counts as C9H10N2O2PF2Cl.",
  "step2": "Propose corrected design adjustments to respect atom composition.",
  "step3": "Describe the structure clearly before writing the SMILES.",
  "smiles": "Your corrected SMILES string here"
}}
```
# """

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

--- SCIENTIST'S TASK ---
Design a SMILES string for a molecule that satisfies the following condition:
Create an isomer of molecular formula **C9H10N2O2PF2Cl**.

HARD CONSTRAINT (MUST follow exactly): 
Generate the valid molecule SMILES that exactly includes:
  - 9 Carbon (C) atoms
  - 10 Hydrogen (H) atoms
  - 2 Nitrogen (N) atoms
  - 2 Oxygen (O) atoms
  - 1 Phosphorus (P) atom
  - 2 Fluorine (F) atoms
  - 1 Chlorine (Cl) atom
YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

Constraints:
- No missing or extra atoms are allowed.
- You must not directly copy example molecules.
- Avoid repeating previously generated SMILES.

--- SCIENTIST'S THINKING ---
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

--- SCIENTIST'S SMILES ---
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Evaluate whether Step1 lists correct atom counts.",
  "step2": "Evaluate if Step2 proposes a chemically plausible isomer.",
  "step3": "Evaluate Step3 description versus actual SMILES generated.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
# """