import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

albuterol_smiles = "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O"
albuterol_mol = Chem.MolFromSmiles(albuterol_smiles)
albuterol_functional_group = utils.utils.describe_albuterol_features(albuterol_mol)


def get_scientist_prompt(topk_smiles):

    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 

Condition for molecule design:
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}. 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.
  
IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL, defined as:  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O  
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

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
  "step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
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
    return f"""Previously generated SMILES. YOU MUST NOT REPEAT ANY OF THEM:
{smiles_history}
Task: Take reviewer's feedback actively and design a SMILES string for a molecule that satisfies the condition:

Condition for molecule design:    
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}). 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO ALBUTEROL with :  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You will be provided with:
1. Previous SMILES string
2. Tanimoto similarity score (0–1) to albuterol CC(C)(C)NCC(O)c1ccc(O)c(CO)c1 based on canonical SMILES
3. Detected functional groups in your previous molecule 

--- MOLECULE SMILES TO IMPROVE ---
MOLECULE SMILES: {previous_smiles}
- Tanimoto similarity score: {score} (0–1)
- Functional groups detected:
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
  "step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
  "smiles": "Your valid SMILES string here"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}. 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.

Be constructive: Provide fixes for issues (e.g., "Replace C=O=C with O=C=O for carbon dioxide").

You are provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- Tanimoto similarity score to albuterol: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
- Detected functional groups in the generated molecule.

--- SCIENTIST'S STEP-WISE THINKING ---
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- SCIENTIST-MOLECULE SMILES ---
SMILES: {scientist_think_dict["smiles"]}
- Tanimoto similarity score: {score} (range: 0 to 1)
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
  "step1": "List accurate features and functional groups identified.\nMention any critical features and functional groups that were missed or misinterpreted.",
  "step2": "Evaluate if the proposed design strategy aligns with the structural and functional similarity goal.\nComment on whether the design aligns with the initial objectives.\nSuggest improvements or alternatives if needed.",
  "step3": "Review the structural construction and positional assignments.\nCheck for missing elements or mismatches in reasoning. (e.g., "Claimed ‘para hydroxyl’ but SMILES places it meta")",
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
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}. 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.

Your previous reasoning steps that you should improve were:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES that you need to improve:
{previous_smiles}

The double-checker reviewed each of your previous reasoning steps and gave the following evaluations:

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
  "step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
  "smiles": "Your improved and valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS EXACT FORMAT, INNOCENT PEOPLE WILL DIE. """


def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step4,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all four steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If **any** step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

=== SCIENTIST'S TASK === 
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}). 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.

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

Use the following format.
Take a deep breath and think carefully before writing your answer. 
```json
{{
  "step1": "Your analysis of whether scientist's Step1 thinking is chemically valid and  reflected in the SMILES.",
  "step2": "Your analysis of whether scientist's Step2 thinking is chemically valid and  reflected in the SMILES.",
  "step3": "Your analysis of whether scientist's Step3 thinking is chemically valid and reflected in the SMILES.",
  "consistency": "Consistent" or "Inconsistent",
}}

```
 """
