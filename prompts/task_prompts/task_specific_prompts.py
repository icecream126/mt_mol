"""Task-specific prompts for different molecular design tasks"""

from typing import Dict, Any
from .base_templates import (
    get_base_scientist_prompt,
    get_base_scientist_prompt_with_review,
    get_base_reviewer_prompt,
    get_base_scientist_prompt_with_double_checker_review,
    get_base_double_checker_prompt,
)
from utils.task_dicts import get_task_to_condition_dict

# Task-specific conditions and constraints
# TASK_CONDITIONS = {
#     "albuterol_similarity":f"""Condition for molecule design:
#     Design a drug-like molecule structurally similar to albuterol (SMILES:  CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O, canonical: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1).
#     Preserve the core scaffold and key functional groups.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL, defined as:
#     - SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O
#     - canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
#     """,

#         "isomers_c7h8n2o2": """Condition for molecular design:
#     Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.

#     HARD CONSTRAINT (MUST follow exactly):
#     Generate the valid molecule SMILES that exactly includes:
#     - 7 Carbon (C) atoms
#     - 8 Hydrogen (H) atoms
#     - 2 Nitrogen (N) atoms
#     - 2 Oxygen (O) atoms
#     YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

#     Constraints:
#     - No missing or extra atoms are allowed.
#     - You must not directly copy example molecules.
#     - Avoid repeating previously generated SMILES.""",

#     "isomers_c9h10n2o2pf2cl": """Your task is to design a SMILES string for a molecule that satisfies the following condition:
#     Create an isomer of molecular formula **C9H10N2O2PF2Cl**.

#     HARD CONSTRAINT (MUST follow exactly):
#     Generate the valid molecule SMILES that exactly includes:
#     - 9 Carbon (C) atoms
#     - 10 Hydrogen (H) atoms
#     - 2 Nitrogen (N) atoms
#     - 2 Oxygen (O) atoms
#     - 1 Phosphorus (P) atom
#     - 2 Fluorine (F) atoms
#     - 1 Chlorine (Cl) atom
#     YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

#     Constraints:
#     - No missing or extra atoms are allowed.
#     - You must not directly copy example molecules.
#     - Avoid repeating previously generated SMILES.""",
#     "amlodipine_mpo": """Conditions:
#     - Achieve high structural similarity to amlodipine (SMILES: Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC).
#     - Preferably maintain around **3 rings** in the molecular structure to preserve desired complexity.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO AMLODIPINE: Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC.""",

#     "celecoxib_rediscovery": """Condition for molecule design:
#     Design a drug-like molecule structurally similar to celecoxib (SMILES: CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F, canonical: Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1). ,
#     Preserve the core scaffold and important pharmacophores.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO CELECOXIB, defined as:
#     - SMILES: CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
#     - canonical SMILES: Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1""",

#     "deco_hop":"""Design a drug-like molecule that preserves the **fixed core scaffold** while modifying peripheral decorations to explore chemical diversity.

#     Chemical Constraints:
#     - Preserve the scaffold matching SMARTS: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12]
#     - Avoid the forbidden SMARTS patterns:
#     - CS([#6])(=O)=O
#     - [#7]-c1ccc2ncsc2c1
#     - Maintain moderate similarity to the reference pharmacophore:
#     - SMILES: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C (similarity capped at 0.85)

#     IMPORTANT CONSTRAINTS:
#     - DO NOT modify the preserved scaffold.
#     - You are encouraged to creatively modify the peripheral decorations (side groups).
#     - DO NOT repeat molecules already generated.
#     """,

#     "zaleplon_mpo":"""Condition for molecule design:
#     - Achieve high structural similarity to zaleplon (SMILES: O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1).
#     - Match the molecular formula **C19H17N3O2** exactly (correct atom counts).

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ZALEPLON (SMILES: O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1).""",

#     "valsartan_smarts":"""Condition for molecule design:
#     Maximize the valsartan_SMARTS score.
#     A high valsartan_SMARTS score means:
#     - Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
#     - Your molecule must have:
#     - A logP (lipophilicity) similar to ~2.0
#     - A TPSA (Topological Polar Surface Area) around ~95
#     - A Bertz complexity close to ~800""",

#     "troglitazon_rediscovery":"""Condition for molecule design:
#     Design a drug-like molecule structurally similar to troglitazone (SMILES: Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O, canonical: Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2). ,
#     Preserve the core scaffold and important pharmacophores.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TROGLITAZONE, defined as:
#     - SMILES: Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O
#     - canonical SMILES: Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2""",

#     "thiothixene_rediscovery":"""Condition for molecule design:
#     Design a drug-like molecule structurally similar to thiothixene (SMILES: CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1, canonical: CN1CCN(CCC=C2c3ccccc3Sc3ccc(S(=O)(=O)N(C)C)cc32)CC1).
#     Preserve the core scaffold and important pharmacophores.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO THIOTHIXENE, defined as:
#     - SMILES: CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1
#     - canonical SMILES: CN1CCN(CCC=C2c3ccccc3Sc3ccc(S(=O)(=O)N(C)C)cc32)CC1""",

#     "sitagliptin_mpo":"""Conditions:
#     - Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
#     - Match the **molecular formula** C16H15F6N5O (no missing or extra atoms).
#     - Design a molecule with:
#     - logP similar to sitagliptin: {round(sitagliptin_logP, 4)}
#     - TPSA (polar surface area) similar to sitagliptin: {round(sitagliptin_TPSA, 4)}
#     - Encourage chemical diversity: avoid being too structurally identical to sitagliptin.""",

#     "scaffold_hop":"""Design a drug-like molecule that **removes the original scaffold** while **preserving critical decorations**.

#     Chemical Constraints:
#     - REMOVE scaffold SMARTS: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
#     - PRESERVE decoration SMARTS: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1
#     - Maintain pharmacophore similarity with SMILES: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C (similarity capped at 0.75).

#     IMPORTANT CONSTRAINTS:
#     - REMOVE the core scaffold but PRESERVE key decorations.
#     - Modify the scaffold creatively to maintain drug-likeness.
#     - DO NOT repeat molecules already generated.""",

#     "ranolazine_mpo":"""Condition for molecule design:
#     - High structural Tanimoto similarity to ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2).
#     - Achieve a Topological Polar Surface Area (TPSA) around **95**.
#     - Maintain a lipophilicity (LogP) around **7**.
#     - Include approximately **1 fluorine atom**.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT generate a molecule identical to ranolazine.""",

#     "qed":"""Condition for molecule design:
#     Maximize the QED (Quantitative Estimation of Drug-likeness) score of the molecule.

#     IMPORTANT CONSTRAINTS:
#     - QED score must be as high as possible (close to 1).
#     - Avoid simply copying example molecules.
#     - You must NOT generate molecules that are unrealistic or synthetically infeasible.""",

#     "perindopril_mpo":"""Condition for molecule design:
#     - High structural similarity to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).
#     - The molecule should contain approximately **2 aromatic rings**.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT generate a molecule identical to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).""",

#     "osimertinib_mpo":"""Condition for molecule design:
#     - High structural similarity to osimertinib (SMILES: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34").
#     - Achieve a Topological Polar Surface Area (TPSA) close to **100**.
#     - Maintain a low-to-moderate lipophilicity (LogP ≈ **1**).

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT generate a molecule identical to osimertinib.""",

#     "mestranol_similarity":"""Condition for molecule design:
#     Design a drug-like molecule structurally similar to mestranol (SMILES: COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1, canonical: C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C).
#     Preserve the core scaffold and key functional groups.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO MESTRANOL, defined as:
#     - SMILES: COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1
#     - canonical SMILES: C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C""",

#     "median2":"""Your task is to design a SMILES string for a molecule that is simultaneously similar to two reference molecules:
#     Design a drug-like molecule that exhibits high ECFP6 fingerprint similarity to both reference compounds simultaneously:

#     Two reference molecules:
#     - Tadalafil SMILES: O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C
#     - Sildenafil SMILES: CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C

#     Achieve balanced similarity to both.""",

#     "median1":"""Your task is to design a SMILES string for a molecule that satisfies the following condition:
#     Design a drug-like molecule that exhibits high ECFP4 fingerprint similarity to both reference compounds simultaneously:

#     Two reference molecules:
#     - camphor SMILES: CC1(C)C2CCC1(C)C(=O)C2
#     - menthol SMILES: CC(C)C1CCC(C)CC1O)""",

#     "jnk3":"""Design a drug-like molecule with high predicted JNK3 inhibitory activity.
#     Maximize the model-predicted probability of JNK3 inhibition.

#     IMPORTANT:
#     - Design chemically valid, realistic molecules.
#     - Preserve critical features related to JNK3 binding.""",

#     "gsk3b":"""Condition:
#     Design a molecule that achieves high predicted binding affinity to the GSK3B target.

#     IMPORTANT:
#     - GSK3B activity is evaluated by a predictive ML model trained on bioactivity data.
#     - Your goal is to maximize the predicted binding probability (score between 0 and 1).""",

#     "fexofenadine_mpo":"""Condition for molecule design:
#     - Achieve high structural similarity to fexofenadine (SMILES: CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4).
#     - Target a Topological Polar Surface Area (TPSA) around **90**.
#     - Aim for moderate lipophilicity with a LogP value close to **4**.

#     IMPORTANT CONSTRAINT:
#     YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO FEXOFENADINE.""",

#     "drd2":"""Maximize the probability of binding to the DRD2 receptor (Dopamine Receptor D2).

#     IMPORTANT CONSTRAINTS:
#     - Design drug-like molecules.
#     - Maximize the DRD2 binding score as high as possible.
#     - Avoid generating identical structures to provided examples.
#     - Avoid repeating molecules you already generated.""",

#     "opv_pce_pcdtbt":"""Condition for Molecular Design:
#     Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

#     Objective = PCE_PCDTBT - SAscore, where:
#     - PCE_PCDTBT: Power Conversion Efficiency of the molecule when paired with PCDTBT as the donor.
#     - SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

#     Your molecule should:
#     - Achieve high PCE_PCDTBT in both settings.
#     - Have low SAscore (simple, stable, synthetically feasible structure).

#     Desirable features to increase PCE_PCDTBT and decrease SAscore:
#     - Strong Donor-Acceptor (D-A) character for charge separation.
#     - Extended conjugation for charge transport.
#     - Planar structure for π-π stacking.
#     - Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
#     - Avoid excessive rings or rare functional groups that increase synthetic complexity.
#     - Use commonly studied OPV substructures (see below).

#     Helpful Building Blocks:
#     - Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
#     - Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
#     - Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).""",

#     "opv_pce_pcbm":"""Condition for Molecular Design:
#     Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

#     Objective = PCE_PCBM - SAscore, where:
#     - PCE_PCBM: Power Conversion Efficiency of the molecule when paired with PCBM as the acceptor.
#     - SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

#     Your molecule should:
#     - Achieve high PCE_PCBM.
#     - Have low SAscore (simple, stable, synthetically feasible structure).

#     Desirable features to increase PCE_PCBM and decrease SAscore:
#     - Strong Donor-Acceptor (D-A) character for charge separation.
#     - Extended conjugation for charge transport.
#     - Planar structure for π-π stacking.
#     - Alkyl chains (e.g., octyl, hexyl) for solubility and processability.
#     - Avoid excessive rings or rare functional groups that increase synthetic complexity.
#     - Use commonly studied OPV substructures (see below).

#     Helpful Building Blocks:
#     - Donor units: thiophene (C1=CSC=C1), fluorene, triphenylamine.
#     - Acceptor units: benzothiadiazole (C1=CC2=NSN=C2C=C1), diketopyrrolopyrrole (DPP).
#     - Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).
#     """,

#     "emitters":"""Condition for Molecular Design:
#     Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

#     Objective 1
#     - Name: Oscillator strength
#     - Notation: f12
#     - Goal: HIGHER IS BETTER

#     Objective 2
#     - Name: Singlet-triplet energy gap
#     - Notation: ΔE(S1 - T1)
#     - Goal: SMALLER IS BETTER

#     Objective 3
#     - Name: Composite Objective
#     - Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
#     - Goal: HIGHER IS BETTER

#     Your molecule should:
#     - Emit light efficiently (maximize f12),
#     - Minimize the singlet-triplet gap (ΔE(S1 - T1) ≈ 0 eV),
#     - Target excitation energy around 3.2 eV for blue light emission,
#     - Avoid overly complex or synthetically inaccessible motifs (e.g., large rings, rare atoms),
#     - Be stable and realistically synthesizable (implicitly guided by structure).

#     Helpful Design Principles:
#     - Planar conjugated systems increase f12 and stabilize excited states,
#     - Rigid aromatic rings and π-bridges promote high emission and reduce vibrational loss,
#     - Small ΔE(S1-T1) enhances TADF via reverse intersystem crossing (RISC),
#     - Electron-donating and withdrawing groups can tune excitation properties.

#     Example building blocks (SMILES):
#     - Electron donors: triphenylamine C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC=CC=C3, carbazole C1=CC=C2C(=C1)C3=CC=CC=C3N2
#     - Electron acceptors: benzothiadiazole C1=CC2=NSN=C2C=C1, triazine C1=CN=NN=C1
#     - π-spacers: thiophene C1=CSC=C1"""

# }

# Task-specific functional group requirements
# TASK_FUNCTIONAL_GROUPS = {
#     "albuterol_similarity": ["hydroxyl", "amine", "aromatic_ring"],
#     "valsartan_smarts": ["tetrazole", "amide", "aromatic_ring"],
#     "logp_optimization": ["aromatic_ring", "hydrophobic_group"],
#     "qed_optimization": ["aromatic_ring", "hydrogen_bond_donor", "hydrogen_bond_acceptor"]
# }

TASK_CONDITIONS = get_task_to_condition_dict()


def get_task_specific_prompt(task_name: str, prompt_type: str, **kwargs) -> str:
    """Get task-specific prompt based on task name and prompt type"""
    if task_name not in TASK_CONDITIONS:
        raise ValueError(f"Unknown task: {task_name}")

    # Get base template
    if prompt_type == "scientist":
        return get_base_scientist_prompt(
            task_condition=TASK_CONDITIONS[task_name], **kwargs
        )
    elif prompt_type == "scientist_with_review":
        return get_base_scientist_prompt_with_review(
            task_condition=TASK_CONDITIONS[task_name], **kwargs
        )
    elif prompt_type == "reviewer":
        return get_base_reviewer_prompt(
            task_condition=TASK_CONDITIONS[task_name], **kwargs
        )
    elif prompt_type == "scientist_with_double_checker":
        return get_base_scientist_prompt_with_double_checker_review(
            task_condition=TASK_CONDITIONS[task_name], **kwargs
        )
    elif prompt_type == "double_checker":
        return get_base_double_checker_prompt(
            task_condition=TASK_CONDITIONS[task_name], **kwargs
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


# def get_task_functional_groups(task_name: str) -> list:
#     """Get required functional groups for a specific task"""
#     if task_name not in TASK_FUNCTIONAL_GROUPS:
#         raise ValueError(f"Unknown task: {task_name}")
#     return TASK_FUNCTIONAL_GROUPS[task_name]
