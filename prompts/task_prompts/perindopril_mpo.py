from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs

# Reference molecule
perindopril_smiles = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
perindopril_mol = Chem.MolFromSmiles(perindopril_smiles)
perindopril_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
    perindopril_mol, radius=2, nBits=2048
)


def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following conditions:

Condition for molecule design:    
- High structural similarity to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).
- The molecule should contain approximately 2 aromatic rings.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).

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
  "step1": "List key structural and property features of perindopril (e.g., bicyclic core, amide bonds, carboxylate group, around 2 aromatic rings).",
  "step2": "Propose modifications while preserving core motifs and maintaining about 2 aromatic rings. Provide chemical reasoning for each modification.",
  "step3": "Describe your designed molecule in natural language before writing the SMILES (e.g., 'A bicyclic lactam system extended with hydrophobic groups.').",
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
        test_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )
        similarity = DataStructs.TanimotoSimilarity(test_fp, perindopril_fp)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    else:
        similarity = num_aromatic_rings = "Invalid SMILES"
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
Design Objectives:
1. Structural similarity to perindopril:
   - The molecule should be structurally similar based on ECFP4 (circular fingerprint) similarity.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Aromatic ring count:
   - The molecule should contain approximately 2 aromatic rings, which balances molecular complexity and desired scaffold features.
   - Your current aromatic ring count: {num_aromatic_rings}


Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its perindopril_mpo score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- perindopril_mpo score: {score}
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
  "step1": "List updated key features and aromatic ring constraint (~2 rings).",
  "step2": "Refine design modifications to better match perindopril and property constraints.",
  "step3": "Natural description of the corrected structure before writing SMILES.",
  "smiles": "Your corrected valid SMILES string"
}}
```
 """


def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    mol = Chem.MolFromSmiles(scientist_think_dict["smiles"])
    if mol is not None:
        test_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )
        similarity = DataStructs.TanimotoSimilarity(test_fp, perindopril_fp)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    else:
        similarity = num_aromatic_rings = "Invalid SMILES"
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:

Your task is to give feedbacks to scientist LLM to design a SMILES string for a molecule that satisfies the following conditions:
Design Objectives:
1. Structural similarity to perindopril:
   - The molecule should be structurally similar based on ECFP4 (circular fingerprint) similarity.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Aromatic ring count:
   - The molecule should contain approximately 2 aromatic rings, which balances molecular complexity and desired scaffold features.
   - Your current aromatic ring count: {num_aromatic_rings}

IMPORTANT CONSTRAINT:  
MUST NOT generate a molecule identical to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. perindopril_mpo score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- perindopril_mpo score: {score}
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
  "step1": "Check whether key structural features and the ~2 aromatic rings constraint were captured.",
  "step2": "Assess if the molecule modification aligns with rediscovery objectives.",
  "step3": "Verify logical consistency between reasoning and final molecule."
}}
```
 """


def get_scientist_prompt_with_double_checker_review(
    previous_thinking, previous_smiles, double_checker_feedback, smiles_history
):
    mol = Chem.MolFromSmiles(previous_smiles)
    if mol is not None:
        test_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )
        similarity = DataStructs.TanimotoSimilarity(test_fp, perindopril_fp)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    else:
        similarity = num_aromatic_rings = "Invalid SMILES"
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
Design Objectives:
1. Structural similarity to perindopril:
   - The molecule should be structurally similar based on ECFP4 (circular fingerprint) similarity.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Aromatic ring count:
   - The molecule should contain approximately 2 aromatic rings, which balances molecular complexity and desired scaffold features.
   - Your current aromatic ring count: {num_aromatic_rings}

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
  "step1": "List corrected critical features for perindopril rediscovery (~2 aromatic rings).",
  "step2": "Propose chemically improved design strategies based on feedback.",
  "step3": "Natural language description of final structure before SMILES.",
  "smiles": "Your corrected valid SMILES string"
}}
```
 """


def get_double_checker_prompt(thinking, improved_smiles):
    mol = Chem.MolFromSmiles(improved_smiles)
    if mol is not None:
        test_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )
        similarity = DataStructs.TanimotoSimilarity(test_fp, perindopril_fp)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    else:
        similarity = num_aromatic_rings = "Invalid SMILES"
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step independently, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

--- SCIENTIST'S TASK ---
Design a SMILES string for a molecule that satisfies the following conditions:
Design Objectives:
1. Structural similarity to perindopril:
   - The molecule should be structurally similar based on ECFP4 (circular fingerprint) similarity.
   - Your current similarity score: {round(similarity, 4) if isinstance(similarity, float) else similarity}

2. Aromatic ring count:
   - The molecule should contain approximately 2 aromatic rings, which balances molecular complexity and desired scaffold features.
   - Your current aromatic ring count: {num_aromatic_rings}

IMPORTANT CONSTRAINT:  
MUST NOT generate a molecule identical to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).

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
  "step1": "Assess whether critical features and 2 aromatic ring constraint were correctly reflected.",
  "step2": "Analyze the chemical rationality of the modifications.",
  "step3": "Confirm consistency between design steps and final SMILES.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
 """
