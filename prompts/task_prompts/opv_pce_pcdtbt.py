def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition:

Condition for Molecular Design:
Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

Objective = PCE_PCDTBT - SAscore, where:
- PCE_PCDTBT: Power Conversion Efficiency of the molecule when paired with PCDTBT as the donor.
- SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

Your molecule should:
- Achieve high PCE_PCDTBT in both settings.
- Have low SAscore (simple, stable, synthetically feasible structure).

 Desirable features to increase PCE_PCDTBT and decrease SAscore:
- Strong Donor-Acceptor (D-A) character for charge separation.
- Extended conjugation for charge transport.
- Planar structure for π-π stacking.
- Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
- Avoid excessive rings or rare functional groups that increase synthetic complexity.
- Use commonly studied OPV substructures (see below).

Helpful Building Blocks:
- Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
- Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
- Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

Constraints:
- DO NOT repeat any previously generated SMILES.
- DO NOT copy top SMILES directly — draw inspiration only.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List desired structural/property features (e.g., 'Donor-acceptor backbone, planar, conjugated, C8 side chains').",
  "step2": "Describe your scaffold and justify how it enhances PCE_PCDTBT and keeps SAscore low (e.g., 'BT core flanked by thiophenes with octyl chains; conjugation and solubility optimized').",
  "step3": "Describe the full molecular structure in words (e.g., 'central benzothiadiazole with two thiophene rings and solubilizing C8 chains').",
  "smiles": "Your valid SMILES string here"
}}
```"""

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Condition for Molecular Design:
Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

Objective = PCE_PCDTBT - SAscore, where:
- PCE_PCDTBT: Power Conversion Efficiency of the molecule when paired with PCDTBT as the donor.
- SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

Your molecule should:
- Achieve high PCE_PCDTBT in both settings.
- Have low SAscore (simple, stable, synthetically feasible structure).

 Desirable features to increase PCE_PCDTBT and decrease SAscore:
- Strong Donor-Acceptor (D-A) character for charge separation.
- Extended conjugation for charge transport.
- Planar structure for π-π stacking.
- Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
- Avoid excessive rings or rare functional groups that increase synthetic complexity.
- Use commonly studied OPV substructures (see below).

Helpful Building Blocks:
- Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
- Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
- Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You will be provided with:
1. Previous SMILES string
2. PCE_PCDTBT - SAscore of the previous SMILES
3. Detected functional groups in your previous molecule 

--- MOLECULE SMILES TO IMPROVE ---
SMILES: {previous_smiles}
PCE_PCDTBT - SAscore Score: {score}
Functional Groups:
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
  "step1": "List desired structural/property features (e.g., 'Donor-acceptor backbone, planar, conjugated, C8 side chains').",
  "step2": "Describe your scaffold and justify how it enhances PCE_PCDTBT and keeps SAscore low (e.g., 'BT core flanked by thiophenes with octyl chains; conjugation and solubility optimized').",
  "step3": "Describe the full molecular structure in words (e.g., 'central benzothiadiazole with two thiophene rings and solubilizing C8 chains').",
  "smiles": "Your valid SMILES string here"
}}
```"""

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""You are a reviewer evaluating a candidate OPV molecule.

Condition for Molecular Design:
Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

Objective = PCE_PCDTBT - SAscore, where:
- PCE_PCDTBT: Power Conversion Efficiency of the molecule when paired with PCDTBT as the donor.
- SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

Your molecule should:
- Achieve high PCE_PCDTBT in both settings.
- Have low SAscore (simple, stable, synthetically feasible structure).

 Desirable features to increase PCE_PCDTBT and decrease SAscore:
- Strong Donor-Acceptor (D-A) character for charge separation.
- Extended conjugation for charge transport.
- Planar structure for π-π stacking.
- Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
- Avoid excessive rings or rare functional groups that increase synthetic complexity.
- Use commonly studied OPV substructures (see below).

Helpful Building Blocks:
- Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
- Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
- Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).

You will be provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- PCE_PCDTBT - SAscore of the previous SMILES.
- Detected functional groups in your previous molecule .

--- SCIENTIST'S THINKING ---
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- SCIENTIST-MOLECULE SMILES ---
SMILES: {scientist_think_dict["smiles"]}
- PCE_PCDTBT - SAscore: {score} 
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
  "step1": "Evaluate how well the described features (e.g., D-A character, conjugation, planarity) match the SMILES.\nList strengths and any overlooked design principles.",
  "step2": "Assess the scaffold and substitution strategy.\nDoes it align with boosting PCE_PCDTBT or reducing SAscore?\nMention if better alternatives exist.",
  "step3": "Check if the natural-language description matches the SMILES structure.\nNote any inconsistencies in chain length, placement, or substructure."
}}
```"""

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previous generated SMILES based on detailed double-checker feedback.
Goal: Improve your OPV molecule to increase PCE_PCDTBT - SAscore.

Your original task:
Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

Objective = PCE_PCDTBT - SAscore, where:
- PCE_PCDTBT: Power Conversion Efficiency of the molecule when paired with PCDTBT as the donor.
- SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

Your molecule should:
- Achieve high PCE_PCDTBT in both settings.
- Have low SAscore (simple, stable, synthetically feasible structure).

 Desirable features to increase PCE_PCDTBT and decrease SAscore:
- Strong Donor-Acceptor (D-A) character for charge separation.
- Extended conjugation for charge transport.
- Planar structure for π-π stacking.
- Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
- Avoid excessive rings or rare functional groups that increase synthetic complexity.
- Use commonly studied OPV substructures (see below).

Helpful Building Blocks:
- Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
- Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
- Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).

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
  "step1": "List desired structural/property features (e.g., 'Donor-acceptor backbone, planar, conjugated, C8 side chains').",
  "step2": "Describe your scaffold and justify how it enhances PCE_PCDTBT and keeps SAscore low (e.g., 'BT core flanked by thiophenes with octyl chains; conjugation and solubility optimized').",
  "step3": "Describe the full molecular structure in words (e.g., 'central benzothiadiazole with two thiophene rings and solubilizing C8 chains').",
  "smiles": "Your valid SMILES string here"
}}
```"""

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step4,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all four steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If **any** step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.


--- SCIENTIST'S TASK ---    
Condition for Molecular Design:
Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

Objective = PCE_PCDTBT - SAscore, where:
- PCE_PCDTBT: Power Conversion Efficiency of the molecule when paired with PCDTBT as the donor.
- SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

Your molecule should:
- Achieve high PCE_PCDTBT in both settings.
- Have low SAscore (simple, stable, synthetically feasible structure).

 Desirable features to increase PCE_PCDTBT and decrease SAscore:
- Strong Donor-Acceptor (D-A) character for charge separation.
- Extended conjugation for charge transport.
- Planar structure for π-π stacking.
- Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
- Avoid excessive rings or rare functional groups that increase synthetic complexity.
- Use commonly studied OPV substructures (see below).

Helpful Building Blocks:
- Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
- Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
- Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).

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

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Does the SMILES reflect the described target features (e.g., donor-acceptor backbone, planarity, conjugation)?\nList present or missing features and assess their impact on performance.",
  "step2": "Are the scaffold and substitutions described in Step2 clearly visible in the SMILES?\nComment on whether the design supports the claimed improvement in PCE_PCDTBT and synthetic simplicity.",
  "step3": "Does the SMILES match the structural description?\nMention any mismatches in atom placement, branching, or chain lengths.",
  "consistency": "Consistent" or "Inconsistent",
}}
```"""