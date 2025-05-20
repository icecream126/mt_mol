import json

def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Maximize the probability of binding to the DRD2 receptor (Dopamine Receptor D2).

IMPORTANT CONSTRAINTS:
- Design drug-like molecules.
- Maximize the DRD2 binding score as high as possible.
- Avoid generating identical structures to provided examples.
- Avoid repeating molecules you already generated.

You are provided with:
- Top-5 example molecules with high DRD2 binding scores below.
- A list of previously generated SMILES you must NOT repeat.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Identify chemical features known to favor DRD2 binding (e.g., aromatic systems, basic nitrogen atoms, specific ring systems).",
  "step2": "Propose scaffold or functional group additions/modifications that improve binding potential.",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
"""

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Improve your molecule based on reviewer's feedback to maximize DRD2 binding.
Improve a SMILES string for a molecule that satisfies the following condition: 
Maximize the probability of binding to the DRD2 receptor (Dopamine Receptor D2).

You must:
- Improve chemical features for DRD2 affinity.
- Increase the probability of binding compared to your previous SMILES.
- Avoid previously generated molecules.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

MOLECULE SMILES TO IMPROVE:
MOLECULE SMILES: {previous_smiles}
- DRD2 Binding Score: {score}
- Detected functional groups:
{functional_groups}

--- YOUR PREVIOUS THOUGHTS AND REVIEWER FEEDBACK ---
Step1:
{scientist_think_dict['step1']}
Reviewer's feedback:
{reviewer_feedback_dict['step1']}

Step2:
{scientist_think_dict['step2']}
Reviewer's feedback:
{reviewer_feedback_dict['step2']}

Step3:
{scientist_think_dict['step3']}
Reviewer's feedback:
{reviewer_feedback_dict['step3']}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Update your list of critical features for DRD2 binding.",
  "step2": "Describe your improved chemical design strategy.",
  "step3": "Describe the final structure in words before giving SMILES.",
  "smiles": "Your improved SMILES string"
}}
```
"""

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM's molecule for:
- Chemical features that enhance or reduce DRD2 binding potential.
- Logical consistency between reasoning steps and final SMILES.
- Adherence to rediscovery objective (maximize DRD2 binding).

Reference:
- Higher DRD2 probability is better.

Scientist's Reasoning:
Step 1: {scientist_think_dict['step1']}
Step 2: {scientist_think_dict['step2']}
Step 3: {scientist_think_dict['step3']}

Scientist's generated SMILES:
{scientist_think_dict['smiles']}
- DRD2 Binding Score: {score}
- Functional Groups detected: 
{functional_groups}


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Features correctly identified and missed.",
  "step2": "Evaluation of proposed design strategy.",
  "step3": "Review structure and check consistency with the design.",
}}
```
"""

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Revise your molecule based on the detailed double-checker feedback.

Original Task:
Design a molecule maximizing DRD2 binding probability.

Your previous reasoning:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES that you need to improve:
{previous_smiles}

Double-checker's Feedback:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Update critical features important for DRD2 affinity.",
  "step2": "Describe improved strategy.",
  "step3": "Describe the updated molecule structure before SMILES.",
  "smiles": "Your improved SMILES"
}}
```
"""

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will evaluate:
- Scientist's thought process.
- Whether it matches the final SMILES for DRD2 binding maximization.

Scientist's Task:
Scientist's task is to design a SMILES string for a molecule that satisfies the following condition: 
Maximize the probability of binding to the DRD2 receptor (Dopamine Receptor D2).

Scientist's Thinking:
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

Generated SMILES:
{improved_smiles}


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Does Step1 reasoning match SMILES?",
  "step2": "Does Step2 reasoning match SMILES?",
  "step3": "Does Step3 description match SMILES?",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""
