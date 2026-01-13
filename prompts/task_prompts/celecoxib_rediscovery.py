import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils


celecoxib_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
# celecoxib_canonical_smiles = canonicalize(celecoxib_smiles)
celecoxib_mol = Chem.MolFromSmiles(celecoxib_smiles)
celecoxib_functional_group = utils.utils.describe_celecoxib_features(celecoxib_mol)


def get_scientist_prompt(topk_smiles):

    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
  
Condition for molecule design:
Design a drug-like molecule structurally similar to celecoxib (SMILES: {celecoxib_smiles}). 
Preserve the core scaffold and important pharmacophores. Celecoxib contains: \n{celecoxib_functional_group}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO CELECOXIB, defined as:  
- SMILES: {celecoxib_smiles}

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
  "step1": "List of the target’s critical structural/property features (e.g., 'Celecoxib: pyrazole core, sulfonamide group, phenyl rings for hydrophobicity.')",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace one phenyl ring with a thiophene to modulate hydrophobicity.')",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., 'A sulfonamide-substituted pyrazole ring connected to a thiophene and phenyl group.')",
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
Design a drug-like molecule structurally similar to celecoxib (SMILES: {celecoxib_smiles}). 
Preserve the core scaffold and important pharmacophores. Celecoxib contains: \n{functional_groups}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO CELECOXIB with :  
- SMILES: {celecoxib_smiles}

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You will be provided with:
1. Previous SMILES string
2. Tanimoto similarity score (0–1) to celecoxib {celecoxib_smiles} based on canonical SMILES
3. Detected functional groups in your previous molecule 

--- MOLECULE SMILES TO IMPROVE ---
MOLECULE SMILES: {previous_smiles}
- Celecoxib rediscovery task score: {score} (0–1)
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

Now based on your previous thoughts and the reviewer's feedback, you need to improve your design.

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
  "step1": "List of the target’s critical structural/property features (e.g., 'Celecoxib: pyrazole core, sulfonamide group, two phenyl rings.')\nIf property-based, specify requirements (e.g., 'Maintain hydrophobic aromatic groups to enhance binding').",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace one phenyl with a thiophene for metabolic stability.').\nJustify each change chemically (e.g., 'Maintains binding but enhances polarity.').",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., 'A pyrazole core with a sulfonamide and two substituted aromatic rings.')",
  "smiles": "Your valid SMILES string here"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition

Be constructive: Provide fixes for issues (e.g., "Replace C=O=C with O=C=O for carbon dioxide").

You are provided with:
1. The scientist's step-wise reasoning.
2. The final generated SMILES.
3. The Tanimoto similarity score to celecoxib: {celecoxib_smiles}
4. The detected functional groups in the generated molecule.

Note that celecoxib contains the following functional groups: \n{celecoxib_functional_group}

--- SCIENTIST'S STEP-WISE THINKING ---
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- SCIENTIST'S MOLECULE SMILES ---
SMILES: {scientist_think_dict["smiles"]}
- Celecoxib rediscovery task score: {score} (0–1)
- Detected functional groups:
{functional_groups}

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
  "step1": "List accurate features and functional groups identified.\nMention any critical features and functional groups that were missed or misinterpreted.",
  "step2": "Evaluate if the proposed design strategy aligns with the structural and functional similarity goal.\nComment on whether the design aligns with the initial objectives.\nSuggest improvements or alternatives if needed.",
  "step3": "Review the structural construction and positional assignments.\nCheck for missing elements or mismatches in reasoning. (e.g., 'Claimed sulfonamide at para but SMILES places it meta')",
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
Design a drug-like molecule structurally similar to celecoxib (SMILES: {celecoxib_smiles}). 
Preserve the core scaffold and important pharmacophores. Celecoxib contains: {celecoxib_functional_group}.

Your previous reasoning steps were:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previously generated SMILES that you must improve:
{previous_smiles}

The double-checker reviewed each of your steps and gave the following evaluations:

- Step1_Evaluation: {double_checker_feedback['step1']}
- Step2_Evaluation: {double_checker_feedback['step2']}
- Step3_Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.

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
  "step1": "List of the target’s critical structural/property features (e.g., 'Celecoxib: pyrazole core, sulfonamide group, two phenyl rings.')\nIf property-based, specify requirements (e.g., 'Maintain aromaticity, preserve sulfonamide for binding.')",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace a phenyl group with a furan ring to enhance polarity.').\n Justify each change chemically.",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., 'A pyrazole ring attached to a sulfonamide and substituted with two aromatic rings.')",
  "smiles": "Your improved and valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS EXACT FORMAT, INNOCENT PEOPLE WILL DIE. """


def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If **any** step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

=== SCIENTIST'S TASK === 
Design a drug-like molecule structurally similar to celecoxib (SMILES: {celecoxib_smiles}). 
Preserve the core scaffold and important pharmacophores. Celecoxib contains: \n{celecoxib_functional_group}.

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

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Your analysis of whether scientist's Step1 thinking is chemically valid and reflected in the SMILES.",
  "step2": "Your analysis of whether scientist's Step2 thinking is chemically valid and reflected in the SMILES.",
  "step3": "Your analysis of whether scientist's Step3 thinking is chemically valid and reflected in the SMILES.",
  "consistency": "Consistent" or "Inconsistent",
}}
```
 """
