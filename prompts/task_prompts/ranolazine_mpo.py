from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

ranolazine_smiles = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
ranolazine_mol = Chem.MolFromSmiles(ranolazine_smiles)
ranolazine_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(ranolazine_mol)

def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following conditions:

Condition for molecule design:
- High structural Tanimoto similarity to ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2).
- Achieve a Topological Polar Surface Area (TPSA) around 95.
- Maintain a lipophilicity (LogP) around 7.
- Include approximately 1 fluorine atom.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to ranolazine.

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
  "step1": "List ranolazine's key structural/property features (e.g., ether linker, secondary amines, one fluorine atom, moderate TPSA ~95, high LogP ~7).",
  "step2": "Propose modifications to optimize TPSA, LogP, and maintain 1 fluorine atom. Justify chemically.",
  "step3": "Describe the designed molecule naturally before writing SMILES (e.g., 'A benzyl ether scaffold linked to a piperazine derivative with a fluorinated aromatic moiety.').",
  "smiles": "Your valid SMILES string here"
}}
```
 """

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    mol=Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
      logp = Descriptors.MolLogP(mol)
      tpsa = Descriptors.TPSA(mol)
      num_fluorine = sum(1 for atom in Chem.MolFromSmiles(scientist_think_dict["smiles"]).GetAtoms() if atom.GetSymbol() == 'F')
      test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
      similarity = DataStructs.TanimotoSimilarity(test_fp, ranolazine_fp)
    else:
      logp = "Invalid SMILES"
      tpsa = "Invalid SMILES"
      num_fluorine = "Invalid SMILES"
      similarity = "Invalid SMILES"
    
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- High structural Tanimoto similarity to ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2).
- Achieve a Topological Polar Surface Area (TPSA) around 95.
- Maintain a lipophilicity (LogP) around 7.
- Include approximately 1 fluorine atom.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to ranolazine.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its ranolazine_mpo score
3. Its detailed ranolazine_mpo score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- ranolazine_mpo score: {score}
- detailed ranolazine_mpo score:
  - Target logP ≈ 7 (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ 95 (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target number of fluorine atoms: 1 (your design: {num_fluorine})
  - Target Tanimoto similarity score: 0.7 (your design: {round(similarity, 4) if isinstance(similarity, float) else similarity})
- Detected functional groups:
{functional_groups}

=== YOUR PREVIOUS THOUGHT AND REVIEWER'S FEEDBACK ===
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
  "step1": "List updated structural and property features (TPSA ≈ 95, LogP ≈ 7, 1 fluorine atom).",
  "step2": "Refine the modification strategies balancing similarity, polarity, hydrophobicity, and fluorination.",
  "step3": "Natural description of the updated structure before writing SMILES.",
  "smiles": "Your corrected valid SMILES string"
}}
```
 """

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    mol=Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
      logp = Descriptors.MolLogP(mol)
      tpsa = Descriptors.TPSA(mol)
      num_fluorine = sum(1 for atom in Chem.MolFromSmiles(scientist_think_dict["smiles"]).GetAtoms() if atom.GetSymbol() == 'F')
      test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
      similarity = DataStructs.TanimotoSimilarity(test_fp, ranolazine_fp)
    else:
      logp = "Invalid SMILES"
      tpsa = "Invalid SMILES"
      num_fluorine = "Invalid SMILES"
      similarity = "Invalid SMILES"
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Preservation of ranolazine's scaffold
Achievement of TPSA (~95), LogP (~7), and 1 fluorine atom

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. Ranolazine_mpo score
4. Detailed ranolazine_mpo score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- ranolazine_mpo score: {score}
- Detailed ranolazine_mpo score:
  - Target logP ≈ 7 (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ 95 (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target number of fluorine atoms: 1 (your design: {num_fluorine})
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
  "step1": "Check if ranolazine’s important structural/property features were correctly identified.",
  "step2": "Evaluate if the design strategy meets TPSA, LogP, and fluorine constraints.",
  "step3": "Verify logical consistency between design reasoning and final SMILES structure."
}}
```
 """

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- High structural Tanimoto similarity to ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2).
- Achieve a Topological Polar Surface Area (TPSA) around 95.
- Maintain a lipophilicity (LogP) around 7.
- Include approximately 1 fluorine atom.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to ranolazine.

Your previous steps:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker feedback:
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
  "step1": "Corrected list of ranolazine’s key features and property targets.",
  "step2": "Revised scaffold/functional group tuning strategy.",
  "step3": "Natural description of final corrected molecule before SMILES.",
  "smiles": "Your improved SMILES string."
}}
```
 """

def get_double_checker_prompt(thinking, improved_smiles):
    mol=Chem.MolFromSmiles(improved_smiles)
    if mol is not None:
      logp = Descriptors.MolLogP(mol)
      tpsa = Descriptors.TPSA(mol)
      num_fluorine = sum(1 for atom in Chem.MolFromSmiles(improved_smiles).GetAtoms() if atom.GetSymbol() == 'F')
      test_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
      similarity = DataStructs.TanimotoSimilarity(test_fp, ranolazine_fp)
    else:
      logp = "Invalid SMILES"
      tpsa = "Invalid SMILES"
      num_fluorine = "Invalid SMILES"
      similarity = "Invalid SMILES"
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step independently, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.


=== SCIENTIST'S TASK ===
Design a SMILES string for a molecule that satisfies the following conditions:
- High structural Tanimoto similarity to ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2).

  - Target logP ≈ 7 (your design: {round(logp, 4) if isinstance(logp, float) else logp})
  - Target TPSA ≈ 95 (your design: {round(tpsa, 4) if isinstance(tpsa, float) else tpsa})
  - Target number of fluorine atoms: 1 (your design: {num_fluorine})
  - Target Tanimoto similarity score: 0.7 (your design: {round(similarity, 4) if isinstance(similarity, float) else similarity})

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to ranolazine.

=== SCIENTIST'S THINKING ===
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

=== SCIENTIST'S SMILES ===
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a guideline, not the answer.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Assess if key structural/property features and fluorine were addressed.",
  "step2": "Check if the proposed design is chemically reasonable and goal-oriented.",
  "step3": "Confirm logical consistency between reasoning steps and final molecule.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """