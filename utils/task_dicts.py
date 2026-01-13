import prompts.task_prompts
import prompts.task_prompts.opv_pce_pcbm
from utils.metrics import *
import utils.utils

def get_task_to_condition_dict():
    return {
    "albuterol_similarity":"""Condition for molecule design:
Design a drug-like molecule structurally similar to albuterol (SMILES:  CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O). 
Preserve the core scaffold and key functional groups.
  
IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL, defined as:  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O""",

    "isomers_c7h8n2o2": """Condition for molecular design:
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.

HARD CONSTRAINT (MUST follow exactly):  
Generate the valid molecule SMILES that exactly includes:
- 7 Carbon (C) atoms  
- 8 Hydrogen (H) atoms  
- 2 Nitrogen (N) atoms  
- 2 Oxygen (O) atoms  
YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

Constraints:
- No missing or extra atoms are allowed.
- You must not directly copy example molecules.
- Avoid repeating previously generated SMILES.""",

"isomers_c9h10n2o2pf2cl": """Your task is to design a SMILES string for a molecule that satisfies the following condition:
Create an isomer of molecular formula **C9H10N2O2PF2Cl**.

HARD CONSTRAINT (MUST follow exactly): 
Generate the valid molecule SMILES that exactly includes:
  - 9 Carbon (C) atoms
  - 10 Hydrogen (H) atoms
  - 2 Nitrogen (N) atoms
  - 2 Oxygen (O) atoms
  - 1 Phosphorus (P) atom
  - 2 Fluorine (F) atoms
  - 1 Chlorine (Cl) atom
YOU MUST GENERATE A MOLECULE SMILES THAT MATCHES THIS FORMULA EXACTLY.

Constraints:
- No missing or extra atoms are allowed.
- You must not directly copy example molecules.
- Avoid repeating previously generated SMILES.""",
"amlodipine_mpo": """Design Objectives:
1. Structural similarity to amlodipine (SMILES: Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC):
   - Aim for a high similarity score based on ECFP4 fingerprints.

2. Ring count constraint:
   - The molecule should have approximately 3 rings, which reflects the topology of amlodipine.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO AMLODIPINE: Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC.""",

"celecoxib_rediscovery": """Condition for molecule design:
Design a drug-like molecule structurally similar to celecoxib (SMILES: CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F. ,
Preserve the core scaffold and important pharmacophores.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO CELECOXIB, defined as:  
- SMILES: CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F""",

"deco_hop":"""Design a drug-like molecule that preserves the **fixed core scaffold** while modifying peripheral decorations to explore chemical diversity.

Chemical Constraints:
- Preserve the scaffold matching SMARTS: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12]
- Avoid the forbidden SMARTS patterns:
  - CS([#6])(=O)=O
  - [#7]-c1ccc2ncsc2c1
- Maintain moderate similarity to the reference pharmacophore:
  - SMILES: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C (similarity capped at 0.85)

IMPORTANT CONSTRAINTS:
- DO NOT modify the preserved scaffold.
- You are encouraged to creatively modify the peripheral decorations (side groups).
- DO NOT repeat molecules already generated.
""",

"zaleplon_mpo":"""Design Objectives:
Design a SMILES string for a molecule that satisfies the following conditions: 
- The molecule must have **high structural similarity** to zaleplon (SMILES: O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1). A high Tanimoto similarity score (based on atom-pair fingerprints) indicates success. 
- Match the molecular formula **C19H17N3O2** exactly (correct atom counts).(Exactly 19 Carbon (C) atoms, 17 Hydrogen (H) atoms, 3 Nitrogen (N) atoms, and 2 Oxygen (O) atoms.)

   

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ZALEPLON, defined as:  
- SMILES: O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1""",

"valsartan_smarts":"""Condition for molecule design:
Maximize the valsartan_SMARTS score.
A high valsartan_SMARTS score means:
- Your molecule MUST contain the specific SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Your molecule must have:
  - A logP (lipophilicity) similar to ~2.0
  - A TPSA (Topological Polar Surface Area) around ~95
  - A Bertz complexity close to ~800""",

"troglitazon_rediscovery":"""Condition for molecule design:  
Design a drug-like molecule structurally similar to troglitazone (SMILES: Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O. ,
Preserve the core scaffold and important pharmacophores.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TROGLITAZONE, defined as:  
- SMILES: Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O""",

"thiothixene_rediscovery":"""Condition for molecule design:    
Design a drug-like molecule structurally similar to thiothixene (SMILES: CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1. 
Preserve the core scaffold and important pharmacophores.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO THIOTHIXENE, defined as:  
- SMILES: CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1""",

"sitagliptin_mpo":"""Conditions: 

Conditions: 
- Create a structurally similar (i.e., high Tanimoto similarity score) to sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F):
- Match the **molecular formula** C16H15F6N5O (no missing or extra atoms).
- Design a molecule with:
  - logP similar to sitagliptin: {round(sitagliptin_logP, 4)}
  - TPSA (polar surface area) similar to sitagliptin: {round(sitagliptin_TPSA, 4)}.""",

"scaffold_hop":"""Design a drug-like molecule that **removes the original scaffold** while **preserving critical decorations**.

Chemical Constraints:
- REMOVE scaffold SMARTS: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
- PRESERVE decoration SMARTS: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1
- Maintain pharmacophore similarity with SMILES: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C (similarity capped at 0.75).

IMPORTANT CONSTRAINTS:
- REMOVE the core scaffold but PRESERVE key decorations.
- Modify the scaffold creatively to maintain drug-likeness.
- DO NOT repeat molecules already generated.""",

"ranolazine_mpo":"""Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- High structural Tanimoto similarity to ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2).
- Achieve a Topological Polar Surface Area (TPSA) around 95.
- Maintain a lipophilicity (LogP) around 7.
- Include approximately 1 fluorine atom.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to ranolazine.""",

"qed":"""Condition for molecule design:
Maximize the QED (Quantitative Estimation of Drug-likeness) score of the molecule.

IMPORTANT CONSTRAINTS:
- QED score must be as high as possible (close to 1).
- Avoid simply copying example molecules.
- You must NOT generate molecules that are unrealistic or synthetically infeasible.""",

"perindopril_mpo":"""Condition for molecule design: 
Your task is to give feedbacks to scientist LLM to design a SMILES string for a molecule that satisfies the following conditions:
Design Objectives:
1. Structural similarity to perindopril:
   - The molecule should be structurally similar based on ECFP4 (circular fingerprint) similarity.

2. Aromatic ring count:
   - The molecule should contain approximately 2 aromatic rings, which balances molecular complexity and desired scaffold features.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC).""",

"osimertinib_mpo":"""Condition for molecule design:
1. Structural similarity to osimertinib:
   - Functional-Class Fingerprint (FCFP4-style) similarity should be moderate (target ≤ 0.8).
   - Extended-Connectivity Fingerprint (ECFP6-style) similarity should be high (target ≈ 0.85).

2. Physicochemical properties:
   - logP should be low, ideally around 1.
   - TPSA (topological polar surface area) should be near 100.

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to osimertinib (SMILES: COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34).""",

"mestranol_similarity":"""Condition for molecule design:  
Design a drug-like molecule structurally similar to mestranol (SMILES: COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1. 
Preserve the core scaffold and key functional groups.
  
IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO MESTRANOL, defined as:  
- SMILES: COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1  """,

"median2":"""Your task is to design a SMILES string for a molecule that is simultaneously similar to two reference molecules:
Design a drug-like molecule that exhibits high ECFP6 fingerprint similarity to both reference compounds simultaneously:

Two reference molecules:    
- Tadalafil SMILES: O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C
- Sildenafil SMILES: CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C

Achieve balanced similarity to both.""",

"median1":"""Your task is to design a SMILES string for a molecule that satisfies the following condition:
Design a drug-like molecule that exhibits high ECFP4 fingerprint similarity to both reference compounds simultaneously:

Two reference molecules:
- camphor SMILES: CC1(C)C2CCC1(C)C(=O)C2
- menthol SMILES: CC(C)C1CCC(C)CC1O)""",

"jnk3":"""Design a drug-like molecule with high predicted JNK3 inhibitory activity.
Maximize the model-predicted probability of JNK3 inhibition.

IMPORTANT:
- Design chemically valid, realistic molecules.
- Preserve critical features related to JNK3 binding.""",

"gsk3b":"""Condition:    
Design a molecule that achieves high predicted binding affinity to the GSK3B target.

IMPORTANT:
- GSK3B activity is evaluated by a predictive ML model trained on bioactivity data.
- Your goal is to maximize the predicted binding probability (score between 0 and 1).""",

"fexofenadine_mpo":"""Condition for molecule design:
- Achieve high structural similarity to fexofenadine (SMILES: CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4).
Design Objectives:
1. Structural similarity to fexofenadine:
   - Moderate atom-pair similarity is encouraged.
   - Similarity scores above 0.8 are not rewarded.

2. Physicochemical properties:
   - TPSA should be close to 90.
   - logP should be close to 4.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO FEXOFENADINE.""",

"drd2":"""Maximize the probability of binding to the DRD2 receptor (Dopamine Receptor D2).

IMPORTANT CONSTRAINTS:
- Design drug-like molecules.
- Maximize the DRD2 binding score as high as possible.
- Avoid generating identical structures to provided examples.
- Avoid repeating molecules you already generated.""",

"opv_pce_pcdtbt":"""Condition for Molecular Design:
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
- Side chains: linear or branched alkyl chains (e.g., CCCCOCC, CCCCCCCCC).""",

"opv_pce_pcbm":"""Condition for Molecular Design:
Design a molecule suitable for use as an organic photovoltaic (OPV) material, with the goal of maximizing the following composite objective:

Objective = PCE_PCBM - SAscore, where:
- PCE_PCBM: Power Conversion Efficiency of the molecule when paired with PCBM as the acceptor.
- SAscore: Synthetic Accessibility score (penalizes difficult-to-synthesize molecules).

Your molecule should:
- Achieve high PCE_PCBM.
- Have low SAscore (simple, stable, synthetically feasible structure).

 Desirable features to increase PCE_PCBM and decrease SAscore:
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
""",

"emitters":"""Condition for Molecular Design:
Achieve the following three objectives to achieve a light-emitting molecule with high quantum efficiency and blue-light emission capability:

Objective 1
- Name: Oscillator strength 
- Notation: f12
- Goal: HIGHER IS BETTER

Objective 2
- Name: Singlet-triplet energy gap
- Notation: ΔE(S1 - T1)
- Goal: SMALLER IS BETTER

Objective 3
- Name: Composite Objective
- Notation: +f12 - ΔE(S1 - T1) - |ΔE(S0 - S1) - 3.2 eV|
- Goal: HIGHER IS BETTER

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
- π-spacers: thiophene C1=CSC=C1"""

}

# Mapping for retrieval_node datasets
def get_task_to_dataset_path_dict():
    return {
    "albuterol_similarity": "/home/anonymous/mt_mol/dataset/250k_top_100/albuterol_similarity_score.json",
    "isomers_c7h8n2o2": "/home/anonymous/mt_mol/dataset/250k_top_100/isomers_c7h8n2o2_score.json",
    "isomers_c9h10n2o2pf2cl": "/home/anonymous/mt_mol/dataset/250k_top_100/isomers_c9h10n2o2pf2cl_score.json",
    "amlodipine_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/amlodipine_mpo_score.json",
    "celecoxib_rediscovery": "/home/anonymous/mt_mol/dataset/250k_top_100/celecoxib_rediscovery_score.json",
    "deco_hop": "/home/anonymous/mt_mol/dataset/250k_top_100/deco_hop_score.json",
    "drd2": "/home/anonymous/mt_mol/dataset/250k_top_100/drd2_score.json",
    "fexofenadine_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/fexofenadine_mpo_score.json",
    "gsk3b": "/home/anonymous/mt_mol/dataset/250k_top_100/gsk3b_score.json",
    "jnk3": "/home/anonymous/mt_mol/dataset/250k_top_100/jnk3_score.json",
    "median1": "/home/anonymous/mt_mol/dataset/250k_top_100/median1_score.json",
    "median2": "/home/anonymous/mt_mol/dataset/250k_top_100/median2_score.json",
    "mestranol_similarity": "/home/anonymous/mt_mol/dataset/250k_top_100/mestranol_similarity_score.json",
    "osimertinib_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/osimertinib_mpo_score.json",
    "perindopril_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/perindopril_mpo_score.json",
    "qed": "/home/anonymous/mt_mol/dataset/250k_top_100/qed_score.json",
    "ranolazine_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/ranolazine_mpo_score.json",
    "scaffold_hop": "/home/anonymous/mt_mol/dataset/250k_top_100/scaffold_hop_score.json",
    "sitagliptin_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/sitagliptin_mpo_score.json",
    "thiothixene_rediscovery": "/home/anonymous/mt_mol/dataset/250k_top_100/thiothixene_rediscovery_score.json",
    "troglitazon_rediscovery": "/home/anonymous/mt_mol/dataset/250k_top_100/troglitazon_rediscovery_score.json",
    "valsartan_smarts": "/home/anonymous/mt_mol/dataset/250k_top_100/valsartan_smarts_score.json",
    "zaleplon_mpo": "/home/anonymous/mt_mol/dataset/250k_top_100/zaleplon_mpo_score.json",
    
}

def get_task_to_score_dict():
    return {
    "albuterol_similarity": get_albuterol_similarity_score,
    "isomers_c7h8n2o2": get_isomers_c7h8n2o2_score,
    "isomers_c9h10n2o2pf2cl": get_isomers_c9h10n2o2pf2cl_score,
    "amlodipine_mpo": get_amlodipine_mpo_score,
    "celecoxib_rediscovery": get_celecoxib_rediscovery_score,
    "deco_hop": get_deco_hop_score,
    "drd2": get_drd2_score,
    "fexofenadine_mpo": get_fexofenadine_mpo_score,
    "gsk3b": get_gsk3b_score,
    "jnk3": get_jnk3_score,
    "median1": get_median1_score,
    "median2": get_median2_score,
    "mestranol_similarity": get_mestranol_similarity_score,
    "osimertinib_mpo": get_osimertinib_mpo_score,
    "perindopril_mpo": get_perindopril_mpo_score,
    "qed":get_qed_score,
    "ranolazine_mpo": get_ranolazine_mpo_score,
    "scaffold_hop": get_scaffold_hop_score,
    "sitagliptin_mpo": get_sitagliptin_mpo_score,
    "thiothixene_rediscovery": get_thiothixene_rediscovery_score,
    "troglitazon_rediscovery": get_troglitazon_rediscovery_score,
    "valsartan_smarts": get_valsartan_smarts_score,
    "zaleplon_mpo": get_zaleplon_mpo_score,
    "opv_pce_pcbm": get_opv_score,
    "opv_pce_pcdtbt": get_opv_score,
    "emitters": get_emitters_score,
}

# Mapping for scientist prompt functions
def get_task_to_scientist_prompt_dict():
    return {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_scientist_prompt,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_scientist_prompt,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_scientist_prompt,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_scientist_prompt,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_scientist_prompt,
    "deco_hop": prompts.task_prompts.deco_hop.get_scientist_prompt,
    "drd2": prompts.task_prompts.drd2.get_scientist_prompt,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_scientist_prompt,
    "gsk3b": prompts.task_prompts.gsk3b.get_scientist_prompt,
    "jnk3": prompts.task_prompts.jnk3.get_scientist_prompt,
    "median1": prompts.task_prompts.median1.get_scientist_prompt,
    "median2": prompts.task_prompts.median2.get_scientist_prompt,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_scientist_prompt,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_scientist_prompt,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_scientist_prompt,
    "qed": prompts.task_prompts.qed.get_scientist_prompt,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_scientist_prompt,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_scientist_prompt,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_scientist_prompt,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_scientist_prompt,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_scientist_prompt,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_scientist_prompt,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_scientist_prompt,
    "opv_pce_pcbm": prompts.task_prompts.opv_pce_pcbm.get_scientist_prompt,
    "opv_pce_pcdtbt": prompts.task_prompts.opv_pce_pcdtbt.get_scientist_prompt,
    "emitters": prompts.task_prompts.emitters.get_scientist_prompt,
}

# Mapping for scientist prompt with reviewer
def get_task_to_scientist_prompt_with_review_dict():
    return {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_scientist_prompt_with_review,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_scientist_prompt_with_review,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_scientist_prompt_with_review,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_scientist_prompt_with_review,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_scientist_prompt_with_review,
    "deco_hop": prompts.task_prompts.deco_hop.get_scientist_prompt_with_review,
    "drd2": prompts.task_prompts.drd2.get_scientist_prompt_with_review,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_scientist_prompt_with_review,
    "gsk3b": prompts.task_prompts.gsk3b.get_scientist_prompt_with_review,
    "jnk3": prompts.task_prompts.jnk3.get_scientist_prompt_with_review,
    "median1": prompts.task_prompts.median1.get_scientist_prompt_with_review,
    "median2": prompts.task_prompts.median2.get_scientist_prompt_with_review,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_scientist_prompt_with_review,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_scientist_prompt_with_review,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_scientist_prompt_with_review,
    "qed": prompts.task_prompts.qed.get_scientist_prompt_with_review,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_scientist_prompt_with_review,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_scientist_prompt_with_review,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_scientist_prompt_with_review,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_scientist_prompt_with_review,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_scientist_prompt_with_review,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_scientist_prompt_with_review,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_scientist_prompt_with_review,
    "opv_pce_pcbm": prompts.task_prompts.opv_pce_pcbm.get_scientist_prompt_with_review,
    "opv_pce_pcdtbt": prompts.task_prompts.opv_pce_pcdtbt.get_scientist_prompt_with_review,
    "emitters": prompts.task_prompts.emitters.get_scientist_prompt_with_review,

}

def get_task_to_reviewer_prompt_dict():
    return {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_reviewer_prompt,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_reviewer_prompt,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_reviewer_prompt,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_reviewer_prompt,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_reviewer_prompt,
    "deco_hop": prompts.task_prompts.deco_hop.get_reviewer_prompt,
    "drd2": prompts.task_prompts.drd2.get_reviewer_prompt,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_reviewer_prompt,
    "gsk3b": prompts.task_prompts.gsk3b.get_reviewer_prompt,
    "jnk3": prompts.task_prompts.jnk3.get_reviewer_prompt,
    "median1": prompts.task_prompts.median1.get_reviewer_prompt,
    "median2": prompts.task_prompts.median2.get_reviewer_prompt,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_reviewer_prompt,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_reviewer_prompt,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_reviewer_prompt,
    "qed": prompts.task_prompts.qed.get_reviewer_prompt,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_reviewer_prompt,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_reviewer_prompt,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_reviewer_prompt,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_reviewer_prompt,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_reviewer_prompt,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_reviewer_prompt,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_reviewer_prompt,
    "opv_pce_pcbm": prompts.task_prompts.opv_pce_pcbm.get_reviewer_prompt,
    "opv_pce_pcdtbt": prompts.task_prompts.opv_pce_pcdtbt.get_reviewer_prompt,
    "emitters": prompts.task_prompts.emitters.get_reviewer_prompt,
}

# Mapping for scientist prompt with double checker

def get_task_to_scientist_prompt_with_double_checker_dict():
    return {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_scientist_prompt_with_double_checker_review,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_scientist_prompt_with_double_checker_review,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_scientist_prompt_with_double_checker_review,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_scientist_prompt_with_double_checker_review,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_scientist_prompt_with_double_checker_review,
    "deco_hop": prompts.task_prompts.deco_hop.get_scientist_prompt_with_double_checker_review,
    "drd2": prompts.task_prompts.drd2.get_scientist_prompt_with_double_checker_review,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_scientist_prompt_with_double_checker_review,
    "gsk3b": prompts.task_prompts.gsk3b.get_scientist_prompt_with_double_checker_review,
    "jnk3": prompts.task_prompts.jnk3.get_scientist_prompt_with_double_checker_review,
    "median1": prompts.task_prompts.median1.get_scientist_prompt_with_double_checker_review,
    "median2": prompts.task_prompts.median2.get_scientist_prompt_with_double_checker_review,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_scientist_prompt_with_double_checker_review,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_scientist_prompt_with_double_checker_review,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_scientist_prompt_with_double_checker_review,
    "qed": prompts.task_prompts.qed.get_scientist_prompt_with_double_checker_review,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_scientist_prompt_with_double_checker_review,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_scientist_prompt_with_double_checker_review,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_scientist_prompt_with_double_checker_review,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_scientist_prompt_with_double_checker_review,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_scientist_prompt_with_double_checker_review,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_scientist_prompt_with_double_checker_review,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_scientist_prompt_with_double_checker_review,
    "opv_pce_pcbm": prompts.task_prompts.opv_pce_pcbm.get_scientist_prompt_with_double_checker_review,
    "opv_pce_pcdtbt": prompts.task_prompts.opv_pce_pcdtbt.get_scientist_prompt_with_double_checker_review,
    "emitters": prompts.task_prompts.emitters.get_scientist_prompt_with_double_checker_review,
}

def get_task_to_double_checker_prompt_dict():
    return {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_double_checker_prompt,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_double_checker_prompt,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_double_checker_prompt,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_double_checker_prompt,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_double_checker_prompt,
    "deco_hop": prompts.task_prompts.deco_hop.get_double_checker_prompt,
    "drd2": prompts.task_prompts.drd2.get_double_checker_prompt,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_double_checker_prompt,
    "gsk3b": prompts.task_prompts.gsk3b.get_double_checker_prompt,
    "jnk3": prompts.task_prompts.jnk3.get_double_checker_prompt,
    "median1": prompts.task_prompts.median1.get_double_checker_prompt,
    "median2": prompts.task_prompts.median2.get_double_checker_prompt,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_double_checker_prompt,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_double_checker_prompt,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_double_checker_prompt,
    "qed": prompts.task_prompts.qed.get_double_checker_prompt,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_double_checker_prompt,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_double_checker_prompt,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_double_checker_prompt,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_double_checker_prompt,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_double_checker_prompt,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_double_checker_prompt,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_double_checker_prompt,
    "opv_pce_pcbm": prompts.task_prompts.opv_pce_pcbm.get_double_checker_prompt,
    "opv_pce_pcdtbt": prompts.task_prompts.opv_pce_pcdtbt.get_double_checker_prompt,
    "emitters": prompts.task_prompts.emitters.get_double_checker_prompt,
}

def get_task_to_functional_group_dict():
    return {
    "albuterol_similarity": utils.utils.describe_albuterol_features,
    "isomers_c7h8n2o2": utils.utils.count_atoms,
    "isomers_c9h10n2o2pf2cl": utils.utils.count_atoms,
    "amlodipine_mpo": utils.utils.describe_albuterol_features,
    "celecoxib_rediscovery": utils.utils.describe_celecoxib_features,
    "deco_hop": utils.utils.describe_deco_hop_features,
    "drd2": utils.utils.describe_drd2_features,
    "fexofenadine_mpo": utils.utils.describe_fexofenadine_features,
    "gsk3b": utils.utils.describe_gsk3b_features,
    "jnk3": utils.utils.describe_jnk3_features,
    "median1": utils.utils.describe_median1_features,
    "median2": utils.utils.describe_median2_features,
    "mestranol_similarity": utils.utils.describe_mestranol_features,
    "osimertinib_mpo": utils.utils.describe_osimertinib_features,
    "perindopril_mpo": utils.utils.describe_albuterol_features,
    "qed": utils.utils.describe_qed_features,
    "ranolazine_mpo": utils.utils.describe_ranolazine_features,
    "scaffold_hop": utils.utils.describe_scaffold_hop_features,
    "sitagliptin_mpo": utils.utils.describe_sitagliptin_features,
    "thiothixene_rediscovery": utils.utils.describe_thiothixene_features,
    "troglitazon_rediscovery": utils.utils.describe_troglitazon_features,
    "valsartan_smarts": utils.utils.describe_valsartan_features,
    "zaleplon_mpo": utils.utils.describe_zaleplon_features,
    
    # TODO: opv, OLED,
}
