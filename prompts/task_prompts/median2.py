import json

# SMILES for median2 task reference molecules
tadalafil_smiles = "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C"
sildenafil_smiles = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that is simultaneously similar to two reference molecules:

Two reference molecules:    
- Tadalafil SMILES: {tadalafil_smiles}
- Sildenafil SMILES: {sildenafil_smiles}

Achieve balanced similarity to both.

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
  "step1": "Identify shared critical features between tadalafil and sildenafil.",
  "step2": "Propose scaffold or substitutions to capture the key properties.",
  "step3": "Describe the full designed structure before writing the SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
"""


def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that is simultaneously similar to two reference molecules:
- Tadalafil (SMILES: {tadalafil_smiles})
- Sildenafil (SMILES: {sildenafil_smiles})

You will also see:
1. Molecule SMILES to improve
2. Its median2 score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- median2 score: {score}
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
  "step1": "Identify shared critical features between tadalafil and sildenafil.",
  "step2": "Propose scaffold or substitutions to better capture key properties.",
  "step3": "Describe the new designed structure.",
  "smiles": "Your revised valid SMILES string here"
}}
```
"""


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that is simultaneously similar to two reference molecules:
- Tadalafil (SMILES: {tadalafil_smiles})
- Sildenafil (SMILES: {sildenafil_smiles})

Reference SMILES:
- Tadalafil: {tadalafil_smiles}
- Sildenafil: {sildenafil_smiles}


Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. median2 score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- median2 score: {score}
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
  "step1": "List correct/missing critical features.",
  "step2": "Assess whether strategy captured both molecules' key aspects.",
  "step3": "Review structural construction accuracy."
}}
```
"""


def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous SMILES based on double-checker feedback.

Original Task:
Design a molecule balancing similarity between:
- Tadalafil (SMILES: {tadalafil_smiles})
- Sildenafil (SMILES: {sildenafil_smiles})

Previous Reasoning:
- Step 1: {previous_thinking['step1']}
- Step 2: {previous_thinking['step2']}
- Step 3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker Feedback:
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
  "step1": "List updated critical shared features.",
  "step2": "Propose corrected design strategy.",
  "step3": "Describe improved designed molecule.",
  "smiles": "New valid SMILES string"
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

--- SCIENTIST'S TASK ---
Design a SMILES string for a molecule that is simultaneously similar to two reference molecules:
- Tadalafil (SMILES: {tadalafil_smiles})
- Sildenafil (SMILES: {sildenafil_smiles})

Achieve balanced similarity to both.

--- SCIENTIST'S THINKING ---
- Step 1: {thinking['step1']}
- Step 2: {thinking['step2']}
- Step 3: {thinking['step3']}

--- SCIENTIST'S SMILES ---
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Assess if critical features are properly captured.",
  "step2": "Assess the design strategy alignment.",
  "step3": "Check consistency between reasoning and structure.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""
