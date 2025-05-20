import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

# ScaffoldHop shared task-specific information
pharmacophor_smiles = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
deco_target_smarts = "[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1"
scaffold_to_remove_smarts = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"

scaffoldhop_description = "remove the original scaffold matching SMARTS pattern: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12 while preserving the key decoration matching: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1"

def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition:
Design a drug-like molecule that **removes the original scaffold** while **preserving critical decorations**.

Chemical Constraints:
- REMOVE scaffold SMARTS: {scaffold_to_remove_smarts}
- PRESERVE decoration SMARTS: {deco_target_smarts}
- Maintain pharmacophore similarity with SMILES: {pharmacophor_smiles} (similarity capped at 0.75).

IMPORTANT CONSTRAINTS:
- REMOVE the core scaffold but PRESERVE key decorations.
- Modify the scaffold creatively to maintain drug-likeness.
- DO NOT repeat molecules already generated.

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
  "step1": "List which decoration features must be preserved and which core structure must be removed.",
  "step2": "Propose a new scaffold and explain chemically why it works with the preserved decorations.",
  "step3": "Describe the complete structure of the molecule naturally before writing SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
 """

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Improve your molecule based on reviewer feedback while following scaffold hopping constraints.

Chemical Constraints:
- REMOVE scaffold SMARTS: {scaffold_to_remove_smarts}
- PRESERVE decoration SMARTS: {deco_target_smarts}
- Maintain pharmacophore similarity with SMILES: {pharmacophor_smiles} (similarity capped at 0.75).

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its scaffold_hop score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- scaffold_hop score: {score}
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
  "step1": "List preserved decorations and confirm scaffold removal.",
  "step2": "Propose and justify a new scaffold chemically.",
  "step3": "Describe the final designed molecule before SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
 """

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that satisfies the following condition:
Design a drug-like molecule that **removes the original scaffold** while **preserving critical decorations**.

Chemical Constraints:
- REMOVE scaffold SMARTS: {scaffold_to_remove_smarts}
- PRESERVE decoration SMARTS: {deco_target_smarts}
- Maintain pharmacophore similarity with SMILES: {pharmacophor_smiles} (similarity capped at 0.75).

IMPORTANT CONSTRAINTS:
- REMOVE the core scaffold but PRESERVE key decorations.
- Modify the scaffold creatively to maintain drug-likeness.
- DO NOT repeat molecules already generated.

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. scaffold_hop score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- scaffold_hop score: {score}
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
  "step1": "Check if forbidden scaffold was removed and decorations preserved.",
  "step2": "Evaluate chemical validity of scaffold replacement.",
  "step3": "Evaluate full molecule construction and logic."
}}
```
 """

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous SMILES design based on detailed double-checker feedback.

Chemical Constraints:
- REMOVE scaffold SMARTS: {scaffold_to_remove_smarts}
- PRESERVE decoration SMARTS: {deco_target_smarts}
- Maintain pharmacophore similarity with SMILES: {pharmacophor_smiles} (similarity capped at 0.75).

Your previous reasoning:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker Feedback:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}

Now, improve your design based on the feedback.
You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "State preserved decorations and confirm scaffold removal.",
  "step2": "Propose new scaffold and justify chemically.",
  "step3": "Describe the final structure naturally before SMILES.",
  "smiles": "Your improved SMILES here"
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

== SCIENTIST'S TASK ==
Design a SMILES string for a molecule that satisfies the following condition:
Design a drug-like molecule that **removes the original scaffold** while **preserving critical decorations**.

Chemical Constraints:
- REMOVE scaffold SMARTS: {scaffold_to_remove_smarts}
- PRESERVE decoration SMARTS: {deco_target_smarts}
- Maintain pharmacophore similarity with SMILES: {pharmacophor_smiles} (similarity capped at 0.75).

IMPORTANT CONSTRAINTS:
- REMOVE the core scaffold but PRESERVE key decorations.
- Modify the scaffold creatively to maintain drug-likeness.
- DO NOT repeat molecules already generated.

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
  "step1": "Consistency check for preserved decorations and scaffold removal.",
  "step2": "Consistency check for new scaffold proposal.",
  "step3": "Consistency check for final structure assembly.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """

