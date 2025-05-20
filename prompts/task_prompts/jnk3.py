import json

# jnk3_smiles = "C1=CC=C(C=C1)C2=NC(=CS2)N"
# jnk3_canonical_smiles = canonicalize(jnk3_smiles)
# jnk3_mol = Chem.MolFromSmiles(jnk3_smiles)
# jnk3_functional_group = utils.utils.describe_jnk3_features(jnk3_mol)

def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule with high predicted JNK3 inhibitory activity.
Maximize the model-predicted probability of JNK3 inhibition.

IMPORTANT:
- Design chemically valid, realistic molecules.
- Preserve critical features related to JNK3 binding.

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
  "step1": "List of structural features important for JNK3 activity (e.g., specific heterocycles, H-bond donors/acceptors)",
  "step2": "Propose scaffold or substituent designs to enhance JNK3 binding.",
  "step3": "Describe your designed molecule in natural language before writing the SMILES.",
  "smiles": "Your valid SMILES string here"
}}
```
"""

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, smiles_history, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule with high predicted JNK3 inhibitory activity.
Maximize the model-predicted probability of JNK3 inhibition.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its jnk3 score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---
MOLECULE SMILES: {previous_smiles}
- jnk3 score: {score}
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
  "step1": "Updated list of structural features for JNK3 inhibition.",
  "step2": "New scaffold/substituent strategies and justifications.",
  "step3": "Natural language description before SMILES.",
  "smiles": "Your improved valid SMILES string here"
}}
```
"""

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
- Potential for strong JNK3 inhibition
- Scientific soundness

You are provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- JNK3 score.
- Detected functional groups.

--- SCIENTIST'S THINKING ---
Step 1: {scientist_think_dict['step1']}
Step 2: {scientist_think_dict['step2']}
Step 3: {scientist_think_dict['step3']}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict['smiles']}
- jnk3 score: {score}
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
  "step1": "Accurate features detected and important omissions.",
  "step2": "Evaluation of scaffold and substituent strategies.",
  "step3": "Check consistency between description and SMILES.",
}}
```
"""

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, smiles_history):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{smiles_history}

Improve your previously generated SMILES based on double-checker feedback for JNK3 inhibition.

Your original task:
Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule with high predicted JNK3 inhibitory activity.
Maximize the model-predicted probability of JNK3 inhibition.


Your previous reasoning steps were:
Step1: {previous_thinking['step1']}
Step2: {previous_thinking['step2']}
Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

The double-checker reviewed each of your steps and gave the following evaluations:
Step1 Evaluation: {double_checker_feedback['step1']}
Step2 Evaluation: {double_checker_feedback['step2']}
Step3 Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List updated structural features critical for JNK3.",
  "step2": "Propose better scaffold or substituent strategy.",
  "step3": "Describe the molecule before writing the SMILES.",
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


=== SCIENTIST'S TASK ===
Design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule with high predicted JNK3 inhibitory activity.
Maximize the model-predicted probability of JNK3 inhibition.

IMPORTANT:
- Design chemically valid, realistic molecules.
- Preserve critical features related to JNK3 binding.

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
  "step1": "Analysis of whether step1 thinking is reflected in the SMILES.",
  "step2": "Analysis of whether step2 strategy matches SMILES design.",
  "step3": "Consistency check between final description and SMILES.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
"""