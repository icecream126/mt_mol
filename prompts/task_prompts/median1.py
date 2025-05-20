import json

median1_description = "Target molecules that are simultaneously similar to both camphor and menthol based on ECFP4 fingerprint similarity."
camphor_smiles = "CC1(C)C2CCC1(C)C(=O)C2"
menthol_smiles = "CC(C)C1CCC(C)CC1O"


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition:
Create a drug-like molecule that is simultaneously similar to both camphor (SMILES: {camphor_smiles}) and menthol (SMILES: {menthol_smiles}), based on ECFP4 fingerprint similarity.

IMPORTANT CONSTRAINT:
- Optimize the structure so that it maintains good similarity to both target molecules.

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
  "step1": "List the critical features derived from camphor and menthol (e.g., fused ring systems, alcohol groups, methyl substituents).",
  "step2": "Propose a structure that captures these key features while ensuring synthetic plausibility.",
  "step3": "Describe the structure clearly before writing the SMILES (e.g., 'A fused bicyclic core with a tertiary alcohol group and several methyl groups attached.'),",
  "smiles": "Your valid SMILES string here"
}}
```
"""


def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that satisfies the following condition:
Create a drug-like molecule that is simultaneously similar to both camphor (SMILES: {camphor_smiles}) and menthol (SMILES: {menthol_smiles}).

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
  "step1": "Updated key features to focus on.",
  "step2": "New or adjusted design strategy.",
  "step3": "Natural language description of the revised molecule.",
  "smiles": "Your improved SMILES string."
}}
```
"""


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the scientist's molecule for:
- Capturing key features from both camphor and menthol.
- Structural plausibility and novelty.
- Alignment with simultaneous similarity goal.

You are provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- median1 score.
- Detected functional groups.

--- SCIENTIST'S THINKING ---
Step1: {scientist_think_dict['step1']}
Step2: {scientist_think_dict['step2']}
Step3: {scientist_think_dict['step3']}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict['smiles']}
- median1 score: {score}
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
  "step1": "Assess identification of critical features.",
  "step2": "Evaluate if the design aligns with task.",
  "step3": "Comment on accuracy and consistency of the structure.",
}}
```
"""


def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your molecule based on double-checker feedback.

Original Task:
Create a molecule similar to both camphor and menthol.

Previous Reasoning:
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
  "step1": "Key features adjustment.",
  "step2": "Refined modification plan.",
  "step3": "Clear description before SMILES.",
  "smiles": "Your revised SMILES string."
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

--- Scientist's Task ---
Your task is to design a SMILES string for a molecule that satisfies the following condition:
Create a drug-like molecule that is simultaneously similar to both camphor (SMILES: {camphor_smiles}) and menthol (SMILES: {menthol_smiles}), based on ECFP4 fingerprint similarity.

IMPORTANT CONSTRAINT:
- Optimize the structure so that it maintains good similarity to both target molecules.

--- Scientist's Thinking ---
- Step1: {thinking['step1']}
- Step2: {thinking['step2']}
- Step3: {thinking['step3']}

--- Scientist's SMILES ---
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Does Step1 correctly identify the key features?",
  "step2": "Does Step2 properly design modifications?",
  "step3": "Does Step3 match the SMILES structure?",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""
