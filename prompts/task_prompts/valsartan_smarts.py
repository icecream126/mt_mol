def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition:

Condition for molecule design:
Maximize the valsartan_SMARTS score.
A high valsartan_SMARTS score means:
- Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Your molecule must have:
  - A logP (lipophilicity) similar to ~2.0
  - A TPSA (Topological Polar Surface Area) around ~95
  - A Bertz complexity close to ~800
  
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
  "step1": "List the critical features: presence of the SMARTS motif, target range for logP (~2.0), TPSA (~95), Bertz complexity (~800).",
  "step2": "Propose strategies to introduce the SMARTS motif while tuning molecular properties (e.g., hydrophobicity, polarity).",
  "step3": "Describe the full structure in natural language before writing the SMILES.",
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

Your task is to design a SMILES string for a molecule that satisfies the following condition:
Maximize the valsartan_SMARTS score.

A high valsartan_SMARTS score means:
- Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Your molecule must have:
  - A logP (lipophilicity) similar to ~2.0
  - A TPSA (Topological Polar Surface Area) around ~95
  - A Bertz complexity close to ~800

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its valsartan_smarts score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- valsartan_smarts score: {score}
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
  "step1": "List essential features: SMARTS matching, target logP/TPSA/Bertz ranges.",
  "step2": "Describe scaffold/property adjustments to better match the objectives.",
  "step3": "Describe the improved structure in natural language.",
  "smiles": "Your improved SMILES string here"
}}
```
"""


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that satisfies the following condition:
Maximize the valsartan_SMARTS score.

A high valsartan_SMARTS score means:
- Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Your molecule must have:
  - A logP (lipophilicity) similar to ~2.0
  - A TPSA (Topological Polar Surface Area) around ~95
  - A Bertz complexity close to ~800
  
You are provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- valsartan_smarts score.
- Detected functional groups.

--- SCIENTIST'S THINKING ---
Step 1: {scientist_think_dict['step1']}
Step 2: {scientist_think_dict['step2']}
Step 3: {scientist_think_dict['step3']}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict['smiles']}
- valsartan_smarts score: {score}
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
  "step1": "Assess whether SMARTS and property targets were correctly identified and addressed.",
  "step2": "Evaluate the proposed design strategy for chemical plausibility and improvement.",
  "step3": "Review structure description accuracy and SMILES matching."
}}
```
"""


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous molecule based on detailed double-checker feedback.

Your task is to design a SMILES string for a molecule that satisfies the following condition:
Maximize the valsartan_SMARTS score.

A high valsartan_SMARTS score means:
- Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Your molecule must have:
  - A logP (lipophilicity) similar to ~2.0
  - A TPSA (Topological Polar Surface Area) around ~95
  - A Bertz complexity close to ~800

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
  "step1": "List features ensuring SMARTS match and desired molecular property ranges.",
  "step2": "Explain corrections and enhancements to boost valsartan_SMARTS score.",
  "step3": "Describe your revised molecule in natural language.",
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

--- SCIENTIST'S TASK ---
Design a SMILES string for a molecule that satisfies the following condition:
Maximize the valsartan_SMARTS score.

A high valsartan_SMARTS score means:
- Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Your molecule must have:
  - A logP (lipophilicity) similar to ~2.0
  - A TPSA (Topological Polar Surface Area) around ~95
  - A Bertz complexity close to ~800
  
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
  "step1": "Evaluate Step1 alignment with SMARTS and properties.",
  "step2": "Evaluate if the Step2 strategy fixes previous mistakes.",
  "step3": "Evaluate Step3 description versus SMILES outcome.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""
