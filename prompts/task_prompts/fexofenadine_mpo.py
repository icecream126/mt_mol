from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs

# Reference molecule
fexofenadine_smiles = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
fexofenadine_mol = Chem.MolFromSmiles(fexofenadine_smiles)
fexofenadine_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
    fexofenadine_mol
)


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- Achieve high structural similarity to fexofenadine (SMILES: {fexofenadine_smiles}).
- Target a Topological Polar Surface Area (TPSA) around 90.
- Aim for moderate lipophilicity with a LogP value close to 4.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO FEXOFENADINE.

You are provided with:
- Top-5 example molecules highly relevant to this task (use them for inspiration, but do not copy them).
- A list of previously generated SMILES (you MUST NOT repeat).

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List the key structural features of fexofenadine (e.g., carboxylic acid group, multiple aromatic rings, hydroxyl groups, piperidine ring).",
  "step2": "Propose scaffold or functional group modifications that maintain similarity while tuning TPSA and LogP. Justify each chemical change.",
  "step3": "Describe the full designed molecule naturally before giving the SMILES (e.g., 'A piperidine core linked to a hydroxylated triaryl system with carboxyl functionality.').",
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
    mol = Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(fp, fexofenadine_fp)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
    else:
        similarity = tpsa = logp = "Invalid SMILES"
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Task: Improve your molecule design by carefully incorporating reviewer's feedback.

Design Objectives:
1. Structural similarity to fexofenadine:
   - Moderate atom-pair similarity is encouraged.
   - Similarity scores above 0.8 are not rewarded.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Physicochemical properties:
   - TPSA should be close to 90.
     - Your current TPSA: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa}
   - logP should be close to 4.
     - Your current logP: {round(logp, 4) if isinstance(logp, float) else logp}

You must reflect reviewer's advice into your improved molecule.

Resources:
- Top-5 relevant examples (for inspiration)
- Previously generated SMILES (avoid duplication)
- Previous similarity score and detected functional groups

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

=== MOLECULE SMILES TO IMPROVE ===
MOLECULE SMILES: {previous_smiles}
- fexofenadine_mpo task score: {score}
- Detected functional groups:
{functional_groups}

=== YOUR PREVIOUS THINKING AND REVIEWER FEEDBACK ===
Step 1:
- Your thought: {scientist_think_dict["step1"]}
- Reviewer feedback: {reviewer_feedback_dict["step1"]}

Step 2:
- Your thought: {scientist_think_dict["step2"]}
- Reviewer feedback: {reviewer_feedback_dict["step2"]}

Step 3:
- Your thought: {scientist_think_dict["step3"]}
- Reviewer feedback: {reviewer_feedback_dict["step3"]}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Updated key features of fexofenadine and molecular property targets (TPSA, LogP).",
  "step2": "Refined design strategy ensuring high similarity while tuning TPSA and LogP properly.",
  "step3": "Natural description of the improved molecule before the SMILES.",
  "smiles": "Your corrected valid SMILES string"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    mol = Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(fp, fexofenadine_fp)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
    else:
        similarity = tpsa = logp = "Invalid SMILES"

    return f"""Evaluate the Scientist LLM’s reasoning and final molecule for:

- Validity: Are the chemical modifications plausible and scientifically sound?
- Preservation: Are fexofenadine’s important structural elements retained?
- Objective Achievement: Is TPSA around 90 and LogP near 4?

Note that scientist LLM's task is:
Design Objectives:
1. Structural similarity to fexofenadine:
   - Moderate atom-pair similarity is encouraged.
   - Similarity scores above 0.8 are not rewarded.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Physicochemical properties:
   - TPSA should be close to 90.
     - Your current TPSA: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa}
   - logP should be close to 4.
     - Your current logP: {round(logp, 4) if isinstance(logp, float) else logp}

Provided:
- Scientist’s detailed stepwise thinking
- Final generated SMILES
- Tanimoto similarity score to fexofenadine
- Detected functional groups

Important Features of Fexofenadine:
- Carboxylic acid group
- Hydroxyl groups
- Multiple aromatic rings
- Piperidine ring

=== SCIENTIST'S STEP-WISE THINKING ===
Step 1: {scientist_think_dict["step1"]}
Step 2: {scientist_think_dict["step2"]}
Step 3: {scientist_think_dict["step3"]}

=== SCIENTIST'S GENERATED SMILES ===
SMILES: {scientist_think_dict["smiles"]}
- fexofenadine_mpo task score: {score}
- Detected functional groups:
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Evaluate whether key fexofenadine features and property goals were correctly captured.",
  "step2": "Assess whether the design matches TPSA and LogP objectives. Suggest better strategies if needed.",
  "step3": "Check if final structure is consistent with the scientist's reasoning. Identify any inconsistencies."
}}
```
 """


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    mol = Chem.MolFromSmiles(previous_smiles)
    if mol is not None:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(fp, fexofenadine_fp)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
    else:
        similarity = tpsa = logp = "Invalid SMILES"
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previously designed molecule based on detailed double-checker feedback.

Design Objectives:
1. Structural similarity to fexofenadine:
   - Moderate atom-pair similarity is encouraged.
   - Similarity scores above 0.8 are not rewarded.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Physicochemical properties:
   - TPSA should be close to 90.
     - Your current TPSA: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa}
   - logP should be close to 4.
     - Your current logP: {round(logp, 4) if isinstance(logp, float) else logp}

Your previous thinking:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker feedback:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List corrected critical features and property constraints for fexofenadine (TPSA ≈ 90, LogP ≈ 4).",
  "step2": "Propose improved chemical modifications that preserve similarity and molecular properties.",
  "step3": "Natural language description of the final molecule before writing SMILES.",
  "smiles": "Your improved valid SMILES string."
}}
```
 """


def get_double_checker_prompt(thinking, improved_smiles):
    mol = Chem.MolFromSmiles(improved_smiles)
    if mol is not None:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        similarity = DataStructs.TanimotoSimilarity(fp, fexofenadine_fp)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
    else:
        similarity = tpsa = logp = "Invalid SMILES"
    return f"""You will evaluate the scientist LLM's thinking and final improved molecule critically:

Checklist:
- Does it retain important fexofenadine features?
- Does it achieve TPSA ≈ 90 and LogP ≈ 4?
- Is the final SMILES consistent with the scientist’s detailed reasoning?

Scientist LLM's task is:
Design Objectives:
1. Structural similarity to fexofenadine:
   - Moderate atom-pair similarity is encouraged.
   - Similarity scores above 0.8 are not rewarded.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Physicochemical properties:
   - TPSA should be close to 90.
     - Your current TPSA: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa}
   - logP should be close to 4.
     - Your current logP: {round(logp, 4) if isinstance(logp, float) else logp}

Inputs:
- Original prompt
- Scientist’s updated stepwise thinking
- Final SMILES

Target: Fexofenadine rediscovery.

=== SCIENTIST'S TASK ===
Scientist's task is to design a SMILES string for a molecule that satisfies the following conditions:
- Achieve high structural similarity to fexofenadine (SMILES: {fexofenadine_smiles}).
- Target a Topological Polar Surface Area (TPSA) around 90.
- Aim for moderate lipophilicity with a LogP value close to 4.

=== SCIENTIST'S STEP-WISE THINKING ===
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

=== FINAL SMILES ===
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Analyze if key structural and property objectives (TPSA, LogP) were preserved.",
  "step2": "Evaluate the chemical rationality and rediscovery alignment.",
  "step3": "Confirm structural consistency with the proposed design strategy.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """
