def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 

Condition for molecule design:
Maximize the QED (Quantitative Estimation of Drug-likeness) score of the molecule.

IMPORTANT CONSTRAINTS:
- QED score must be as high as possible (close to 1).
- Avoid simply copying example molecules.
- You must NOT generate molecules that are unrealistic or synthetically infeasible.

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
  "step1": "List molecular features known to improve QED (e.g., moderate size, good balance of hydrophilicity/hydrophobicity, absence of problematic groups).",
  "step2": "Describe specific strategies you will use to maximize QED (e.g., 'Introduce a hydroxyl group to improve polarity while maintaining low molecular weight').",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
"""

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Take the reviewer's feedback actively and design a better SMILES string maximizing the QED score.

IMPORTANT:
- QED must be improved while maintaining reasonable chemical realism.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its qed score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- qed score: {score}
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
  "step1": "List features favoring high QED and identify weak points from the previous design.",
  "step2": "Propose improvements or scaffold changes to boost QED.",
  "step3": "Describe the improved molecule in natural language.",
  "smiles": "Your newly improved SMILES here"
}}
```
"""

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that satisfies the following condition: 
Maximize the QED (Quantitative Estimation of Drug-likeness) score of the molecule.

IMPORTANT CONSTRAINTS:
- QED score must be as high as possible (close to 1).
- Avoid simply copying example molecules.
- You must NOT generate molecules that are unrealistic or synthetically infeasible.

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. qed score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- qed score: {score}
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
  "step1": "Evaluate if identified features truly promote high QED and if any key aspects were missed.",
  "step2": "Evaluate whether the proposed strategy aligns with maximizing QED.",
  "step3": "Evaluate the chemical plausibility and realism of the final design."
}}
```
"""

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your molecule based on the double-checker's feedback.

Original Task:
Maximize QED (Quantitative Estimation of Drug-likeness) score.

Previous Steps:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

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
  "step1": "List important chemical features boosting QED.",
  "step2": "Propose improved design ideas that address previous mistakes.",
  "step3": "Describe the final designed molecule in natural language.",
  "smiles": "Your improved SMILES string here"
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

--- SCIENTIST's TASK ---
Design a SMILES string for a molecule that satisfies the following condition: 
Maximize the QED (Quantitative Estimation of Drug-likeness) score of the molecule.

IMPORTANT CONSTRAINTS:
- QED score must be as high as possible (close to 1).
- Avoid simply copying example molecules.
- You must NOT generate molecules that are unrealistic or synthetically infeasible.

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
  "step1": "Evaluate if Step1 thinking matches final molecule.",
  "step2": "Evaluate if Step2 strategy is applied logically.",
  "step3": "Evaluate if Step3 molecule description matches the SMILES.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""