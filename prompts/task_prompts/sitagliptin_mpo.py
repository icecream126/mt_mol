from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

sitagliptin_smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)
sitagliptin_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(sitagliptin_mol)
sitagliptin_logP = Descriptors.MolLogP(sitagliptin_mol)
sitagliptin_TPSA = Descriptors.TPSA(sitagliptin_mol)


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string that satisfies the following conditions:
Conditions: 
- Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
- Match the molecular formula C16H15F6N5O (no missing or extra atoms).
- Design a molecule with:
  - logP similar to sitagliptin: {round(sitagliptin_logP, 4)}
  - TPSA (polar surface area) similar to sitagliptin: {round(sitagliptin_TPSA, 4)}
  - Tanimoto similarity score: 0.7

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List the required molecular formula and key physicochemical targets (e.g., 'C16H15F6N5O, logP ~ target, TPSA ~ target').",
  "step2": "Propose a chemical strategy balancing similarity, formula matching, and property control.",
  "step3": "Describe the designed molecule's structure before presenting the SMILES.",
  "smiles": "Your designed SMILES string here"
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
    mol = Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(test_fp, sitagliptin_fp)
    else:
        logp = "Invalid SMILES"
        tpsa = "Invalid SMILES"
        similarity = "Invalid SMILES"
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Revise your SMILES design based on reviewer feedback to better meet the sitagliptin_mpo objectives.

Your task is to design a SMILES string that satisfies the following conditions:
Conditions: 
- Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
- Match the molecular formula C16H15F6N5O (no missing or extra atoms).
- Design a molecule with:
  - logP similar to sitagliptin: {round(sitagliptin_logP, 4)}
  - TPSA (polar surface area) similar to sitagliptin: {round(sitagliptin_TPSA, 4)}


Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its sitagliptin_mpo score
3. Its detailed sitagliptin_mpo score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- ranolazine_mpo score: {score}
- detailed ranolazine_mpo score:
  - Target logP ≈ {round(sitagliptin_logP, 4)} (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ {round(sitagliptin_TPSA, 4)} (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target Tanimoto similarity score: 0.7 (your design: {round(similarity, 4) if isinstance(similarity, float) else similarity})
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
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Reaffirm the molecular formula and property targets.",
  "step2": "Update your design strategy considering reviewer feedback.",
  "step3": "Describe the new molecule clearly before SMILES.",
  "smiles": "Your improved SMILES string here"
}}
```
"""


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    mol = Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(test_fp, sitagliptin_fp)
    else:
        logp = "Invalid SMILES"
        tpsa = "Invalid SMILES"
        similarity = "Invalid SMILES"
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
  - Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
  - Match the molecular formula C16H15F6N5O (no missing or extra atoms).
  - Design a molecule with:
    - Target logP ≈ {round(sitagliptin_logP, 4)} (current design: {round(logp, 4) if isinstance(logp, float) else logp})
    - Target TPSA ≈ {round(sitagliptin_TPSA, 4)} (current design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})


Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. sitagliptin_mpo score
4. Detailed ranolazine_mpo score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- sitagliptin_mpo score: {score}
- Detailed sitagliptin_mpo score:
  - Target logP ≈ {round(sitagliptin_logP, 4)} (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ {round(sitagliptin_TPSA, 4)} (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target Tanimoto similarity score: 0.7 (your design: {round(similarity, 4) if isinstance(similarity, float) else similarity})
- Functional groups detected:
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Assess if the molecular formula and property targets were correctly understood.",
  "step2": "Evaluate whether the design strategy appropriately balances formula, property, and similarity.",
  "step3": "Review structure description accuracy and its reflection in SMILES."
}}
```
"""


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    mol = Chem.MolFromSmiles(previous_smiles)
    if mol is not None:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(test_fp, sitagliptin_fp)
    else:
        logp = "Invalid SMILES"
        tpsa = "Invalid SMILES"
        similarity = "Invalid SMILES"
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous molecule based on the double-checker's detailed feedback.

Original task:
Design a SMILES string that satisfies the following conditions:
Conditions: 
- Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
- Match the molecular formula C16H15F6N5O (no missing or extra atoms).
- Design a molecule with:
  - Target logP ≈ {round(sitagliptin_logP, 4)} (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ {round(sitagliptin_TPSA, 4)} (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target Tanimoto similarity score: 0.7 (your design: {round(similarity, 4) if isinstance(similarity, float) else similarity})

Your Previous Steps:
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
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Correct and explicitly list formula and property targets.",
  "step2": "Revise the design strategy to better meet chemical and physicochemical goals.",
  "step3": "Describe the new structure before presenting the SMILES.",
  "smiles": "Your corrected SMILES here"
}}
```
"""


def get_double_checker_prompt(thinking, improved_smiles):
    mol = Chem.MolFromSmiles(improved_smiles)
    if mol is not None:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(test_fp, sitagliptin_fp)
    else:
        logp = "Invalid SMILES"
        tpsa = "Invalid SMILES"
        similarity = "Invalid SMILES"
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step independently, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

---- SCIENTIST'S TASK ---
Your task is to design a SMILES string that satisfies the following conditions:
Conditions: 
- Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
- Match the molecular formula C16H15F6N5O (no missing or extra atoms).
- Design a molecule with:
  - Target logP ≈ {round(sitagliptin_logP, 4)} (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ {round(sitagliptin_TPSA, 4)} (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target Tanimoto similarity score: 0.7 (your design: {round(similarity, 4) if isinstance(similarity, float) else similarity})


--- SCIENTIST'S THINKING ---
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

--- SCIENTIST'S SMILES ---
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Evaluate if Step1 correctly targets formula and property goals.",
  "step2": "Assess if Step2 proposes a good design strategy respecting all constraints.",
  "step3": "Check whether Step3's description matches the generated SMILES.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""
