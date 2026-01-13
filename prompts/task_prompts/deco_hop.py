import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

# DecoHop specific information
pharmacophor_smiles = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
scaffold_target_smarts = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12]"
forbidden_deco1_smarts = "CS([#6])(=O)=O"
forbidden_deco2_smarts = "[#7]-c1ccc2ncsc2c1"

# Scaffold description
# decohop_core_description = "the fixed scaffold corresponding to the SMARTS pattern: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule that preserves the **fixed core scaffold** while modifying peripheral decorations to explore chemical diversity.

Chemical Constraints:
- Preserve the scaffold matching SMARTS: {scaffold_target_smarts}
- Avoid the forbidden SMARTS patterns:
  - {forbidden_deco1_smarts}
  - {forbidden_deco2_smarts}
- Maintain moderate similarity to the reference pharmacophore:
  - SMILES: {pharmacophor_smiles} (similarity capped at 0.85)

IMPORTANT CONSTRAINTS:
- DO NOT modify the preserved scaffold.
- You are encouraged to creatively modify the peripheral decorations (side groups).
- DO NOT repeat molecules already generated.

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List and explain the preserved scaffold features, and indicate which parts can be modified.",
  "step2": "Propose specific peripheral modifications while ensuring forbidden SMARTS are avoided.\nJustify each change chemically (e.g., 'Improves solubility, enhances binding.').",
  "step3": "Describe your final designed molecule in natural language before writing the SMILES (e.g., 'The scaffold remains unchanged while R1 is modified with a fluorine group.').",
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
Design a drug-like molecule that preserves the **fixed core scaffold** while modifying peripheral decorations to explore chemical diversity.

Chemical Constraints:
- Preserve the scaffold matching SMARTS: {scaffold_target_smarts}
- Avoid the forbidden SMARTS patterns:
  - {forbidden_deco1_smarts}
  - {forbidden_deco2_smarts}
- Maintain moderate similarity to the reference pharmacophore:
  - SMILES: {pharmacophor_smiles} (similarity capped at 0.85)

You are provided with:
- Top-5 example molecules related to the task, which you may use as inspiration (but DO NOT copy).
- Previously generated SMILES, which you MUST NOT repeat.

Top-5 Relevant SMILES Examples:
{topk_smiles}

Also provided:
1. Your previous SMILES.
2. Diversity-preservation score (higher is better).
3. Detected functional groups.

--- MOLECULE SMILES TO IMPROVE ---
SMILES: {previous_smiles}
- deco_hop task score: {score}
- Detected functional groups:
{functional_groups}

--- PREVIOUS THOUGHTS AND REVIEWER FEEDBACK ---
Step1 (Core + Decorations):

Your previous thinking:\n{scientist_think_dict["step1"]}
Reviewer's feedback:\n{reviewer_feedback_dict["step1"]}

Step2 (Decoration Strategy):

Your previous thinking:\n{scientist_think_dict["step2"]}
Reviewer's feedback:\n{reviewer_feedback_dict["step2"]}

Step3 (Molecule Construction):

Your previous thinking:\n{scientist_think_dict["step3"]}
Reviewer's feedback:\n{reviewer_feedback_dict["step3"]}

Now, based on your previous thought and the reviewer's feedback, improve your design.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List of preserved core scaffold and allowed modification points.",
  "step2": "Specific proposed peripheral modifications and chemical justification.",
  "step3": "Natural language description of the molecule before SMILES.",
  "smiles": "Your improved and valid SMILES string"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM's molecule design reasoning.

Your evaluation should check:
- Did the molecule **preserve** the scaffold matching SMARTS: {scaffold_target_smarts}?
- Did the molecule **avoid forbidden motifs**: {forbidden_deco1_smarts}, {forbidden_deco2_smarts}?
- Are the decoration modifications chemically valid and meaningful?
- Is the molecule reasonably similar to the reference pharmacophore ({pharmacophor_smiles}) without exceeding the similarity cap (0.85)?

You are provided with:
- The scientist's step-by-step reasoning
- The final generated SMILES
- The diversity-preservation score
- Detected functional groups

--- SCIENTIST'S STEP-WISE THINKING ---
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- SCIENTIST'S FINAL SMILES ---
SMILES: {scientist_think_dict["smiles"]}
- deco_hop task score: {score}
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
  "step1": "Whether the core scaffold was correctly preserved and forbidden motifs were avoided.",
  "step2": "Evaluation of the chemical appropriateness of peripheral modifications.",
  "step3": "Assessment of molecule construction completeness and correctness."
}}
```
 """


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous SMILES design based on detailed double-checker feedback.

Original Task:
Design a molecule that preserves the **fixed core scaffold** and modifies peripheral decorations.

Chemical Constraints:
- Preserve the scaffold matching SMARTS: {scaffold_target_smarts}
- Avoid the forbidden SMARTS patterns:
  - {forbidden_deco1_smarts}
  - {forbidden_deco2_smarts}
- Maintain moderate similarity to the reference pharmacophore:
  - SMILES: {pharmacophor_smiles} (similarity capped at 0.85)

Your previous reasoning:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}


Double-checker Feedback:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}

Now, update your molecule according to the feedback.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List preserved scaffold features and allowed decoration points.",
  "step2": "Propose specific peripheral modifications with chemical reasoning.",
  "step3": "Describe the full structure naturally before writing SMILES.",
  "smiles": "Your improved and chemically valid SMILES"
}}
```
 """


def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be provided with:
- User task: Preserve the core scaffold and modify decorations.
- The scientist's step-wise reasoning.
- The final SMILES molecule.

You must verify:
- Is the fixed scaffold preserved (SMARTS: {scaffold_target_smarts})?
- Are forbidden motifs ({forbidden_deco1_smarts}, {forbidden_deco2_smarts}) absent?
- Are the modifications chemically reasonable and meaningful?

Evaluate each step independently.

=== SCIENTIST'S TASK ===
- Preserve the scaffold matching SMARTS: {scaffold_target_smarts}
- Avoid the forbidden SMARTS patterns:
  - {forbidden_deco1_smarts}
  - {forbidden_deco2_smarts}
- Maintain moderate similarity to the reference pharmacophore:
  - SMILES: {pharmacophor_smiles} (similarity capped at 0.85)


=== SCIENTIST'S THINKING ===
Step 1: {thinking['step1']}
Step 2: {thinking['step2']}
Step 3: {thinking['step3']}

=== SCIENTIST'S SMILES ===
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Evaluate consistency of Step1 (preservation of scaffold and SMARTS).",
  "step2": "Evaluate consistency of Step2 (proper peripheral modifications).",
  "step3": "Evaluate consistency of Step3 (full molecule construction).",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """
