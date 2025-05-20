def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule suitable as an efficient organic emitter for OLED applications.

Condition for Molecular Design:
Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

Objective 1
- Name: Oscillator strength 
- Notation: f12
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN 2.97)

Objective 2
- Name: Singlet-triplet energy gap
- Notation: ΔE(S1 - T1)
- Goal: SMALLER IS BETTER (MUST BE AT LEAST SMALLER THAN 0.02)

Objective 3
- Name: Composite Objective
- Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN -0.04)

Where ΔE(S0 - S1): Excitation energy

Your molecule should:
- Emit light efficiently (maximize f12),
- Minimize the singlet-triplet gap (ΔE(S1 - T1) ≈ 0 eV),
- Target excitation energy around 3.2 eV for blue light emission,
- Avoid overly complex or synthetically inaccessible motifs (e.g., large rings, rare atoms),
- Be stable and realistically synthesizable (implicitly guided by structure).

Helpful Design Principles:
- Planar conjugated systems increase f12 and stabilize excited states,
- Rigid aromatic rings and π-bridges promote high emission and reduce vibrational loss,
- Small ΔE(S1-T1) enhances TADF via reverse intersystem crossing (RISC),
- Electron-donating and withdrawing groups can tune excitation properties.

Example building blocks (SMILES):
- Electron donors: triphenylamine C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC=CC=C3, carbazole C1=CC=C2C(=C1)C3=CC=CC=C3N2
- Electron acceptors: benzothiadiazole C1=CC2=NSN=C2C=C1, triazine C1=CN=NN=C1
- π-spacers: thiophene C1=CSC=C1

Constraints:
- DO NOT repeat previously generated SMILES.
- DO NOT copy top examples directly — use them as inspiration.

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
  "step1": "List desired structural/property features (e.g., 'donor-acceptor-donor system, rigid π-bridge, planar structure, minimal ΔE(S1 - T1)').",
  "step2": "Describe your scaffold and justify how it enhances f12, minimizes ΔE(S1 - T1), and aligns ΔE(S0 - S1) to ~3.2 eV (e.g., 'carbazole core flanked by phenylene bridges and BT unit').",
  "step3": "Describe the molecule in words (e.g., 'central triazine core with donor triphenylamine groups and conjugated phenylene bridges').",
  "smiles": "Your valid SMILES string here"
}}
```"""
def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Condition for Molecular Design:
Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

Objective 1
- Name: Oscillator strength 
- Notation: f12
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN 2.97)

Objective 2
- Name: Singlet-triplet energy gap
- Notation: ΔE(S1 - T1)
- Goal: SMALLER IS BETTER (MUST BE AT LEAST SMALLER THAN 0.02)

Objective 3
- Name: Composite Objective
- Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN -0.04)

Your molecule should:
- Emit light efficiently (maximize f12),
- Minimize the singlet-triplet gap (ΔE(S1 - T1) ≈ 0 eV),
- Target excitation energy around 3.2 eV for blue light emission,
- Avoid overly complex or synthetically inaccessible motifs (e.g., large rings, rare atoms),
- Be stable and realistically synthesizable (implicitly guided by structure).

Helpful Design Principles:
- Planar conjugated systems increase f12 and stabilize excited states,
- Rigid aromatic rings and π-bridges promote high emission and reduce vibrational loss,
- Small ΔE(S1-T1) enhances TADF via reverse intersystem crossing (RISC),
- Electron-donating and withdrawing groups can tune excitation properties.

Example building blocks (SMILES):
- Electron donors: triphenylamine C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC=CC=C3, carbazole C1=CC=C2C(=C1)C3=CC=CC=C3N2
- Electron acceptors: benzothiadiazole C1=CC2=NSN=C2C=C1, triazine C1=CN=NN=C1
- π-spacers: thiophene C1=CSC=C1

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

--- PREVIOUS SMILES TO IMPROVE ---
SMILES: {previous_smiles}  
Score: {score}  
Detected Functional Groups: {functional_groups}

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
  "step1": "Refine key features to include e.g., 'D-A-D backbone, conjugated bridge, rigid planarity, minimized singlet-triplet gap'.",
  "step2": "Explain your scaffold and its benefits to emission (e.g., 'phenylene bridge and BT acceptor enhance f12 and lower ΔE(S1 - T1)').",
  "step3": "Describe molecule in plain terms (e.g., 'central triazine core with flanking TPA units and bridging phenylene rings').",
  "smiles": "Your valid SMILES string here"
}}
```"""
def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""You are a reviewer evaluating a candidate organic emitter molecule for OLEDs.

Condition for Molecular Design:
Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

Objective 1
- Name: Oscillator strength 
- Notation: f12
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN 2.97)

Objective 2
- Name: Singlet-triplet energy gap
- Notation: ΔE(S1 - T1)
- Goal: SMALLER IS BETTER (MUST BE AT LEAST SMALLER THAN 0.02)

Objective 3
- Name: Composite Objective
- Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN -0.04)

Constraints to Ensure Validity:
- The SMILES **must be syntactically valid and parsable by RDKit**.
- It must support **hydrogen addition using `rdkit.Chem.AddHs()`** without error.
- Avoid:
  - Unbalanced rings or valences (e.g., C with 5 bonds).
  - Uncommon elements or radicals (e.g., [Se], [Na], charged atoms like [N+]).
  - Multi-bridge fusions or uncommon fused polycyclic cores.

Use only **safe and verified building blocks**:
- Electron Donors:
  - Triphenylamine — C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC=CC=C3  
  - Carbazole — C1=CC=C2C(=C1)C3=CC=CC=C3N2
- Electron Acceptors:
  - Benzothiadiazole — C1=CC2=NSN=C2C=C1  
  - Triazine — C1=CN=NN=C1
- π-Spacers:
  - Thiophene — C1=CSC=C1  
  - Phenylene — C1=CC=CC=C1

--- SCIENTIST'S STEP-WISE THINKING ---
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- SCIENTIST-MOLECULE SMILES ---
SMILES: {scientist_think_dict["smiles"]}
- Score: {score}
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
  "step1": "Do the listed features support strong emission and low ΔE(S1 - T1)? What’s missing?",
  "step2": "Evaluate the scaffold and substitutions — do they improve emission and color tuning? Any better alternatives?",
  "step3": "Does the structural description clearly match the SMILES? Any mismatches or unrealistic claims?"
}}
```"""
def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}
Condition for Molecular Design:
Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

Objective 1
- Name: Oscillator strength 
- Notation: f12
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN 2.97)

Objective 2
- Name: Singlet-triplet energy gap
- Notation: ΔE(S1 - T1)
- Goal: SMALLER IS BETTER (MUST BE AT LEAST SMALLER THAN 0.02)

Objective 3
- Name: Composite Objective
- Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN -0.04)

Constraints to Ensure Validity:
- The SMILES **must be syntactically valid and parsable by RDKit**.
- It must support **hydrogen addition using `rdkit.Chem.AddHs()`** without error.
- Avoid:
  - Unbalanced rings or valences (e.g., C with 5 bonds).
  - Uncommon elements or radicals (e.g., [Se], [Na], charged atoms like [N+]).
  - Multi-bridge fusions or uncommon fused polycyclic cores.

Use only **safe and verified building blocks**:
- Electron Donors:
  - Triphenylamine — C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC=CC=C3  
  - Carbazole — C1=CC=C2C(=C1)C3=CC=CC=C3N2
- Electron Acceptors:
  - Benzothiadiazole — C1=CC2=NSN=C2C=C1  
  - Triazine — C1=CN=NN=C1
- π-Spacers:
  - Thiophene — C1=CSC=C1  
  - Phenylene — C1=CC=CC=C1

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
  "step1": "Refined features (e.g., planarity, conjugation, TADF-friendly groups).",
  "step2": "Updated scaffold rationale (e.g., 'replaced donor to enhance f12 while keeping ΔE(S1 - T1) low').",
  "step3": "Full description of molecule structure in words.",
  "smiles": "Your new valid SMILES string here"
}}
```"""
def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You are the double-checker validating the logical consistency between molecular design reasoning and the proposed SMILES.

--- SCIENTIST'S TASK ---
Condition for Molecular Design:
Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

Objective 1
- Name: Oscillator strength 
- Notation: f12
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN 2.97)

Objective 2
- Name: Singlet-triplet energy gap
- Notation: ΔE(S1 - T1)
- Goal: SMALLER IS BETTER (MUST BE AT LEAST SMALLER THAN 0.02)

Objective 3
- Name: Composite Objective
- Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
- Goal: HIGHER IS BETTER (MUST BE AT LEAST HIGHER THAN -0.04)

Constraints to Ensure Validity:
- The SMILES **must be syntactically valid and parsable by RDKit**.
- It must support **hydrogen addition using `rdkit.Chem.AddHs()`** without error.
- Avoid:
  - Unbalanced rings or valences (e.g., C with 5 bonds).
  - Uncommon elements or radicals (e.g., [Se], [Na], charged atoms like [N+]).
  - Multi-bridge fusions or uncommon fused polycyclic cores.

Use only **safe and verified building blocks**:
- Electron Donors:
  - Triphenylamine — C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC=CC=C3  
  - Carbazole — C1=CC=C2C(=C1)C3=CC=CC=C3N2
- Electron Acceptors:
  - Benzothiadiazole — C1=CC2=NSN=C2C=C1  
  - Triazine — C1=CN=NN=C1
- π-Spacers:
  - Thiophene — C1=CSC=C1  
  - Phenylene — C1=CC=CC=C1

Review the following:

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
  "step1": "Do the described features match the SMILES? Are they realistic for emission?",
  "step2": "Does the scaffold and substitution plan fit the emission and color tuning goals?",
  "step3": "Does the structure match the description exactly?",
  "consistency": "Consistent" or "Inconsistent"
}}
```"""
