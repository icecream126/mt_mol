import json

# GSK3B task LLM prompts

def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Condition:    
Design a molecule that achieves high predicted binding affinity to the GSK3B target.

IMPORTANT:
- GSK3B activity is evaluated by a predictive ML model trained on bioactivity data.
- Your goal is to maximize the predicted binding probability (score between 0 and 1).

You are provided with:
- Top-5 example molecules with high GSK3B activity, listed below. Use these for inspiration, but DO NOT COPY them exactly.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Example Molecules (SMILES, score):
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Identify key substructures and features associated with GSK3B activity.",
  "step2": "Propose design strategies or modifications to maximize activity.",
  "step3": "Describe the designed molecule’s structure before providing the SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
"""

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Improve your previous design to maximize GSK3B activity, based on reviewer feedback.

--- MOLECULE SMILES TO IMPROVE ---
MOLECULE SMILES: {previous_smiles}
- gsk3b task score: {score}
- Detected functional groups:
{functional_groups}

--- YOUR PREVIOUS THINKING AND REVIEWER'S FEEDBACK ---
Step1 (Key Features):
{scientist_think_dict['step1']}
Feedback:
{reviewer_feedback_dict['step1']}

Step2 (Design Strategy):
{scientist_think_dict['step2']}
Feedback:
{reviewer_feedback_dict['step2']}

Step3 (Construction):
{scientist_think_dict['step3']}
Feedback:
{reviewer_feedback_dict['step3']}

Use the feedback to revise your design.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List preserved critical features and new ideas.",
  "step2": "Propose improved strategy for GSK3B binding.",
  "step3": "Describe your improved molecule.",
  "smiles": "Your revised SMILES string"
}}
```
"""

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning and SMILES generation for:
- Chemical validity
- Biological relevance to GSK3B activity
- Alignment with maximizing GSK3B predicted score

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict['step1']}

Step2: {scientist_think_dict['step2']}

Step3: {scientist_think_dict['step3']}

--- SCIENTIST'S GENERATED SMILES ---
SMILES: {scientist_think_dict['smiles']}
- GSK3B task score: {score}
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
  "step1": "List features correctly captured and any missing important features.",
  "step2": "Assess if the design strategy matches binding optimization goals.",
  "step3": "Check if the final molecule structurally matches the design intent."
}}
```
"""

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous molecule design based on double-checker feedback.

Your original task:
Maximize the predicted binding affinity to GSK3B by designing a novel SMILES.

Your previous reasoning:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Your previous SMILES to improve:
{previous_smiles}

--- DOUBLE CHECKER FEEDBACK ---
Step1 Feedback: {double_checker_feedback['step1']}
Step2 Feedback: {double_checker_feedback['step2']}
Step3 Feedback: {double_checker_feedback['step3']}

Now revise your design and molecule.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Identify preserved and newly added key features.",
  "step2": "Explain improved design strategies.",
  "step3": "Describe the updated molecular structure.",
  "smiles": "New improved SMILES"
}}
```
"""

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""Evaluate the Scientist’s new molecule design logically against their reasoning.

Original task: Maximize GSK3B binding activity.

--- SCIENTIST'S TASK ---
Scientist's task is to design a SMILES string for a molecule that achieves high predicted binding affinity to the GSK3B target.

--- SCIENTIST'S THINKING ---
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

--- NEW SMILES ---
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Check if Step1 thinking is correctly reflected in the SMILES.",
  "step2": "Check if Step2 strategy is achieved.",
  "step3": "Check if Step3 description matches the final molecule.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""