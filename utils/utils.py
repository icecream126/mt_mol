from rdkit import Chem, RDLogger
import torch
import glob
import re
import os
import numpy as np
import torch.nn.functional as F
from rdkit.Chem import Draw
import wandb
# from langchain.vectorstores import FAISS
from langchain.schema import Document
from rdkit.Chem import Fragments
import re
from rdkit import Chem
from rdkit.Chem import Fragments
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

from rdkit import Chem
from collections import defaultdict
from rdkit.Chem import inchi
import time
import traceback
from typing import List, Callable, Tuple
from typing import Union
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.rdchem import Mol
from sklearn.cluster import KMeans
from sklearn.model_selection._split import _BaseKFold
from tqdm.auto import tqdm



import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdchem import Mol
from tqdm.auto import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Cluster import Butina
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np
import io
import sys
from io import StringIO
from operator import itemgetter
from typing import Optional, List, Tuple
from itertools import combinations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors3D import NPR1, NPR2
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Fragments
import requests
# from utils.metrics import *
import pandas as pd
from openai import OpenAI

import json
from dataclasses import dataclass
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import inchi
import pubchempy as pcp
import json

import importlib.util
import sys

# Helper functions (unchanged)
def get_conformer_energies(mol: Chem.Mol) -> List[float]:
    return [float(conf.GetProp("Energy")) for conf in mol.GetConformers()]

def gen_conformers(mol: Chem.Mol, num_confs=50):
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        energy_list = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        for energy_tuple, conf in zip(energy_list, mol.GetConformers()):
            converged, energy = energy_tuple
            conf.SetDoubleProp("Energy", energy)
        mol = Chem.RemoveHs(mol)
    except ValueError:
        mol = None
    return mol


def refine_conformers(mol: Chem.Mol, energy_threshold: float = 50, rms_threshold: Optional[float] = 0.5) -> Chem.Mol:
    energy_list = [float(conf.GetProp("Energy")) for conf in mol.GetConformers()]
    energy_array = np.array(energy_list)
    min_energy = min(energy_list)
    energy_array -= min_energy
    energy_remove_idx = np.argwhere(energy_array > energy_threshold)
    for i in energy_remove_idx.flatten()[::-1]:
        mol.RemoveConformer(int(i))
    conf_ids = [x.GetId() for x in mol.GetConformers()]
    if rms_threshold is not None:
        rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(conf_ids, 2)]
        rms_remove_idx = list(set([x[1] for x in rms_list if x[2] < rms_threshold]))
        for i in sorted(rms_remove_idx, reverse=True):
            mol.RemoveConformer(int(i))
    return mol

def update_topk_smiles(topk_smiles, new_entry, k=100):
    """
    Inserts a new (SMILES, score) entry into topk_smiles if the score is high enough.
    Keeps the list sorted in descending order by score and keeps only the top k entries.
    
    Args:
        topk_smiles (list of tuples): List of (SMILES, score), sorted by score descending.
        new_entry (tuple): A new (SMILES, score) entry.
        k (int): Maximum number of entries to keep.

    Returns:
        list of tuples: Updated list of top-k (SMILES, score).
    """
    smiles_set = set(smiles for smiles, _ in topk_smiles)

    # Avoid inserting duplicates
    if new_entry[0] in smiles_set:
        return topk_smiles

    # If new score is better than worst, insert and trim
    if len(topk_smiles) < k or new_entry[1] > topk_smiles[-1][1]:
        topk_smiles.append(new_entry)
        topk_smiles = sorted(topk_smiles, key=lambda x: x[1], reverse=True)
        if len(topk_smiles) > k:
            topk_smiles = topk_smiles[:k]
    return topk_smiles

def run_pubchem_functions(function_names, smiles, utils_path="/home/anonymous/mt_mol/utils/utils.py"):
    """
    Dynamically loads functions from utils.py and executes them with the given SMILES.
    
    Args:
        function_names (list of str): List of function names to call.
        smiles (str): SMILES string input for the functions.
        utils_path (str): Path to the utils.py file containing the functions.
    
    Returns:
        str: Structured output showing the SMILES and each function name with its result.
    """
    import importlib.util
    import sys

    # Load the module dynamically
    module_name = "chem_utils"
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = utils_module
    spec.loader.exec_module(utils_module)

    output = [f"=== SMILES: {smiles} ===\n"]

    for fn_name in function_names:
        if hasattr(utils_module, fn_name):
            func = getattr(utils_module, fn_name)
            try:
                result = func(smiles)
                formatted = f"[{fn_name}]\n{result.strip()}"
            except Exception as e:
                formatted = f"[{fn_name}] - ERROR: {e}"
        else:
            formatted = f"[{fn_name}] - NOT FOUND"
        
        output.append(formatted)

    return "\n\n".join(output)




def smiles2inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchikey = inchi.MolToInchiKey(mol)
    return inchikey

def safe_llm_call(prompt, llm, llm_type, llm_temperature, max_retries=10, sleep_sec=2, tools=None):
    for attempt in range(max_retries):
        try:
            if tools:
                raw_response = llm.chat.completions.create(
                    model=llm_type,
                    messages=prompt,
                    tools=tools,
                    response_format={"type": "json_object"},
                    temperature=llm_temperature,
                )
            else:
                raw_response = llm.chat.completions.create(
                    model=llm_type,
                    messages=prompt,
                    response_format={"type": "json_object"},
                    temperature=llm_temperature,
                )
            # content = raw_response.choices[0].message.content
            # result = json.loads(content)
            return raw_response# , content
        except Exception as e:
            print(f"[Error] LLM call failed on attempt {attempt + 1}: {e}")
            traceback.print_exc()
            time.sleep(sleep_sec)
    print("[Warning] All retries failed. Returning empty output.")
    return {}, ""



def format_topk_smiles(topk_smiles):

    formatted = "\n".join(
        f"({repr(smiles.strip())}, {score:.6f})"
        for smiles, score in topk_smiles
    )
    return formatted

def add_with_limit(s, item, max_len=10000):
    if len(s) < max_len:
        s.add(item)
        # return s
    else:
        print(f"Cannot add '{item}': reached max size ({max_len})")

def format_set_as_text(s):
    if not s:
        return "Currently no history"
    return "\n".join(sorted(s))



def count_atoms(m):
    m = Chem.AddHs(m)

    atomic_count = defaultdict(lambda: 0)
    for atom in m.GetAtoms():
        atomic_count[atom.GetSymbol()] += 1

    # Convert to text format
    text_output = ""
    for atom, count in sorted(atomic_count.items()):
        text_output += f"- {atom}: {count}\n"

    return text_output

def describe_zaleplon_features(mol):
    descriptions = []

    # Benzene rings
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic system).")

    # Aromatic nitrogen heterocycles (important for zaleplon structure)
    aromatic_nitrogen = Fragments.fr_Ar_N(mol)
    if aromatic_nitrogen:
        descriptions.append(f"- {aromatic_nitrogen} aromatic nitrogen atom(s) (indicative of nitrogen-containing heterocycles like pyrazole, pyridine).")

    # Amide groups (C=O-N)
    amide_count = Fragments.fr_amide(mol)
    if amide_count:
        descriptions.append(f"- {amide_count} amide bond(s) (C=O–N linkage).")

    # Nitrile groups (C≡N)
    nitrile_count = Fragments.fr_nitrile(mol)
    if nitrile_count:
        descriptions.append(f"- {nitrile_count} nitrile group(s) (C≡N triple bond).")

    if not descriptions:
        descriptions.append("- No key zaleplon-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_valsartan_features(mol):
    descriptions = []

    # Benzene rings
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic system).")

    # Amide groups
    amide_count = Fragments.fr_amide(mol)
    if amide_count:
        descriptions.append(f"- {amide_count} amide bond(s) (C=O–N linkages).")

    # Tertiary amines
    tertiary_amine = Fragments.fr_NH0(mol)
    if tertiary_amine:
        descriptions.append(f"- {tertiary_amine} tertiary amine group(s) (nitrogen with three carbon attachments).")

    # Carboxylic acids
    carboxylic_acid = Fragments.fr_COO(mol)
    if carboxylic_acid:
        descriptions.append(f"- {carboxylic_acid} carboxylic acid group(s) (COOH).")

    if not descriptions:
        descriptions.append("- No key valsartan-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_troglitazon_features(mol):
    descriptions = []

    # Phenol groups (aromatic OH)
    phenol_count = Fragments.fr_phenol(mol)
    if phenol_count:
        descriptions.append(f"- {phenol_count} phenol group(s) (aromatic hydroxyl).")

    # Benzene rings
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic system).")

    # Alkyl ethers (O-alkyl, such as methoxy groups)
    alkyl_ethers = Fragments.fr_ether(mol)
    if alkyl_ethers:
        descriptions.append(f"- {alkyl_ethers} alkyl ether group(s) (possibly methoxy groups).")

    # Carbonyl groups (for TZD ring - C=O bonds)
    carbonyls = Fragments.fr_C_O(mol)
    if carbonyls:
        descriptions.append(f"- {carbonyls} carbonyl group(s) (C=O, characteristic of thiazolidinedione ring).")

    # Sulfur atoms (for TZD or sulfur-containing rings)
    sulfur_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    if sulfur_count:
        descriptions.append(f"- {sulfur_count} sulfur atom(s) (suggesting sulfur-containing rings like TZD).")

    if not descriptions:
        descriptions.append("- No key troglitazone-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_thiothixene_features(mol):
    descriptions = []

    # Sulfur atom counts (important for thioxanthene core)
    sulfur_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    if sulfur_count:
        descriptions.append(f"- {sulfur_count} sulfur atom(s) (suggesting thioxanthene or sulfur-containing core).")

    # Piperazine ring (2 nitrogen atoms in 6-membered ring)
    piperazine_like = Fragments.fr_NH0(mol)
    if piperazine_like:
        descriptions.append(f"- {piperazine_like} tertiary amine group(s) (related to piperazine presence).")

    # Benzene rings
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic systems typical in thiothixene).")

    # Aromatic amines (aromatic-NH connections)
    aromatic_nh = Fragments.fr_Ar_N(mol)
    if aromatic_nh:
        descriptions.append(f"- {aromatic_nh} aromatic nitrogen(s) (often in antipsychotic scaffolds).")

    if not descriptions:
        descriptions.append("- No key thiothixene-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_sitagliptin_features(mol):
    descriptions = []

    # Fluorine atoms
    fluorine_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")
    if fluorine_count:
        descriptions.append(f"- {fluorine_count} fluorine atom(s) (important for fluorinated drug-like properties).")

    # Amide groups
    amide_count = Fragments.fr_amide(mol)
    if amide_count:
        descriptions.append(f"- {amide_count} amide group(s) (key feature in drug scaffolds).")

    # Tertiary amines
    tertiary_amine = Fragments.fr_NH0(mol)
    if tertiary_amine:
        descriptions.append(f"- {tertiary_amine} tertiary amine group(s) (neutral nitrogen atoms).")


    # Aliphatic hydroxyl groups (for TPSA)
    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s) (polar substituents).")

    # Benzene rings (sometimes present)
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic character).")

    if not descriptions:
        descriptions.append("- No key sitagliptin-like fragments found.")

    res = "\n".join(descriptions)
    return res



def describe_scaffold_hop_features(mol):
    descriptions = []

    # Focus on substituents (side chains), not core scaffold
    aromatic_oh = Fragments.fr_Ar_OH(mol)
    if aromatic_oh:
        descriptions.append(f"- {aromatic_oh} aromatic hydroxyl group(s) (important substituent).")

    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s) (common polar side chain).")

    amides = Fragments.fr_amide(mol)
    if amides:
        descriptions.append(f"- {amides} amide group(s) (common bioisostere side chain).")

    ethers = Fragments.fr_ether(mol)
    if ethers:
        descriptions.append(f"- {ethers} ether group(s) (polar linker substituent).")

    halogens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ["F", "Cl", "Br", "I"])
    if halogens:
        descriptions.append(f"- {halogens} halogen atom(s) (substituent tuning reactivity or lipophilicity).")

    if not descriptions:
        descriptions.append("- No key substituent fragments found (possible pure scaffold structure).")

    res = "\n".join(descriptions)
    return res


def describe_ranolazine_features(mol):
    descriptions = []

    # TPSA-relevant polar groups
    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s) (enhances TPSA).")

    amides = Fragments.fr_amide(mol)
    if amides:
        descriptions.append(f"- {amides} amide bond(s) (polar, improves TPSA).")

    ethers = Fragments.fr_ether(mol)
    if ethers:
        descriptions.append(f"- {ethers} ether group(s) (common polar linkage).")

    # Hydrophobicity
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (hydrophobic character).")


    # Fluorine atoms
    fluorine_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")
    if fluorine_count:
        descriptions.append(f"- {fluorine_count} fluorine atom(s) detected (fluorine tuning of properties).")

    if not descriptions:
        descriptions.append("- No key ranolazine-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_qed_features(mol):
    descriptions = []

    aromatic_rings = Fragments.fr_benzene(mol)
    if aromatic_rings:
        descriptions.append(f"- {aromatic_rings} benzene ring(s) (good for drug-likeness).")

    aliphatic_hydroxy = Fragments.fr_Al_OH(mol)
    if aliphatic_hydroxy:
        descriptions.append(f"- {aliphatic_hydroxy} aliphatic hydroxyl group(s) (enhances solubility).")

    amides = Fragments.fr_amide(mol)
    if amides:
        descriptions.append(f"- {amides} amide bond(s) (common in bioactive compounds).")

    ethers = Fragments.fr_ether(mol)
    if ethers:
        descriptions.append(f"- {ethers} ether group(s) (common linker motifs in drugs).")


    if not descriptions:
        descriptions.append("- No key drug-likeness promoting fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_perindopril_features(mol):
    descriptions = []

    carboxylic_acid = Fragments.fr_COO(mol)
    if carboxylic_acid:
        descriptions.append(f"- {carboxylic_acid} carboxylic acid group(s) (important for ACE inhibition).")

    amides = Fragments.fr_amide(mol)
    if amides:
        descriptions.append(f"- {amides} amide bond(s) (peptide-like linkages).")

    secondary_amines = Fragments.fr_NH1(mol)
    if secondary_amines:
        descriptions.append(f"- {secondary_amines} secondary amine group(s) (N–H).")

    aromatic_rings = Fragments.fr_benzene(mol)
    if aromatic_rings:
        descriptions.append(f"- {aromatic_rings} benzene ring(s) detected (although perindopril is primarily aliphatic).")

    aliphatic_hydroxy = Fragments.fr_Al_OH(mol)
    if aliphatic_hydroxy:
        descriptions.append(f"- {aliphatic_hydroxy} aliphatic hydroxyl group(s) (–OH on non-aromatic carbon).")

    if not descriptions:
        descriptions.append("- No key perindopril-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_osimertinib_features(mol):
    descriptions = []

    aniline = Fragments.fr_Ar_N(mol)
    if aniline:
        descriptions.append(f"- {aniline} aniline-type aromatic amine group(s) detected (important for osimertinib's activity).")

    acrylamide_like = Fragments.fr_amide(mol)
    if acrylamide_like:
        descriptions.append(f"- {acrylamide_like} amide group(s) (likely part of acrylamide-like warheads).")

    methoxy_groups = Fragments.fr_ether(mol)
    if methoxy_groups:
        descriptions.append(f"- {methoxy_groups} ether group(s) (–O–), typically methoxy groups present in osimertinib structure.")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic systems).")

    if not descriptions:
        descriptions.append("- No key osimertinib-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_mestranol_features(mol):
    descriptions = []

    phenol_count = Fragments.fr_phenol(mol)
    if phenol_count:
        descriptions.append(f"- {phenol_count} phenol group(s) (aromatic hydroxyls on an aromatic ring).")

    ether_groups = Fragments.fr_ether(mol)
    if ether_groups:
        descriptions.append(f"- {ether_groups} ether linkage(s) (–O– group, e.g., methoxy group).")

    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s) (e.g., alcohol group at 17-position).")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic ring presence).")

    if not descriptions:
        descriptions.append("- No key mestranol-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_median2_features(mol):
    descriptions = []

    carbonyl_groups = Fragments.fr_C_O(mol)
    if carbonyl_groups:
        descriptions.append(f"- {carbonyl_groups} carbonyl group(s) (C=O, as in amide or ketone).")

    tertiary_amines = Fragments.fr_NH0(mol)
    if tertiary_amines:
        descriptions.append(f"- {tertiary_amines} tertiary amine group(s) (no hydrogen attached).")

    secondary_amines = Fragments.fr_NH1(mol)
    if secondary_amines:
        descriptions.append(f"- {secondary_amines} secondary amine group(s) (one hydrogen attached).")

    ether_linkages = Fragments.fr_ether(mol)
    if ether_linkages:
        descriptions.append(f"- {ether_linkages} ether linkage(s) (–O– group).")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic system).")

    if not descriptions:
        descriptions.append("- No key tadalafil/sildenafil-like features found.")

    res = "\n".join(descriptions)
    return res

def describe_median1_features(mol):
    descriptions = []

    hydroxyl_groups = Fragments.fr_Al_OH(mol)
    if hydroxyl_groups:
        descriptions.append(f"- {hydroxyl_groups} aliphatic hydroxyl group(s) (e.g., like in menthol).")

    ketone_groups = Fragments.fr_C_O(mol)
    if ketone_groups:
        descriptions.append(f"- {ketone_groups} carbonyl group(s) (e.g., like in camphor).")

    if not descriptions:
        descriptions.append("- No key camphor/menthol-like features found.")

    res = "\n".join(descriptions)
    return res

def describe_jnk3_features(mol):
    descriptions = []

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic system for π-π stacking with JNK3 binding pocket).")

    hbond_donors = Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) + Fragments.fr_Al_OH(mol)
    if hbond_donors:
        descriptions.append(f"- {hbond_donors} potential hydrogen bond donor group(s) (amines or hydroxyls).")

    hbond_acceptors = Fragments.fr_C_O(mol) + Fragments.fr_ether(mol) + Fragments.fr_amide(mol)
    if hbond_acceptors:
        descriptions.append(f"- {hbond_acceptors} potential hydrogen bond acceptor group(s) (carbonyls, ethers, amides).")

    amide_groups = Fragments.fr_amide(mol)
    if amide_groups:
        descriptions.append(f"- {amide_groups} amide group(s) (important for hinge binding interactions).")

    nitrogen_heterocycles = Fragments.fr_pyridine(mol) + Fragments.fr_imidazole(mol)
    if nitrogen_heterocycles:
        descriptions.append(f"- {nitrogen_heterocycles} nitrogen-containing aromatic ring(s) (pyridine, imidazole, etc.).")

    if not descriptions:
        descriptions.append("- No key JNK3-inhibitor-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_gsk3b_features(mol):
    descriptions = []

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic system for π-stacking).")

    hbond_donors = Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) + Fragments.fr_Al_OH(mol)
    if hbond_donors:
        descriptions.append(f"- {hbond_donors} potential H-bond donor group(s) (amines, hydroxyls).")

    hbond_acceptors = Fragments.fr_C_O(mol) + Fragments.fr_ether(mol) + Fragments.fr_amide(mol)
    if hbond_acceptors:
        descriptions.append(f"- {hbond_acceptors} potential H-bond acceptor group(s) (carbonyls, ethers, amides).")

    amides = Fragments.fr_amide(mol)
    if amides:
        descriptions.append(f"- {amides} amide group(s) (important for hinge binding to GSK3B).")

    nitrogen_heterocycles = Fragments.fr_pyridine(mol) + Fragments.fr_imidazole(mol)
    if nitrogen_heterocycles:
        descriptions.append(f"- {nitrogen_heterocycles} nitrogen-containing aromatic ring(s) (e.g., pyridine, imidazole).")

    if not descriptions:
        descriptions.append("- No key GSK3B inhibitor-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_fexofenadine_features(mol):
    descriptions = []

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (aromatic systems).")

    carboxylic_acid = Fragments.fr_COO(mol)
    if carboxylic_acid:
        descriptions.append(f"- {carboxylic_acid} carboxylic acid group(s) (–COOH).")

    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s) (non-aromatic alcohols).")

    if not descriptions:
        descriptions.append("- No key fexofenadine-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_drd2_features(mol):
    descriptions = []

    aromatic_rings = Fragments.fr_benzene(mol)
    if aromatic_rings:
        descriptions.append(f"- {aromatic_rings} benzene ring(s) (aromatic system).")

    primary_amine = Fragments.fr_NH2(mol)
    if primary_amine:
        descriptions.append(f"- {primary_amine} primary amine group(s) (–NH₂).")

    secondary_amine = Fragments.fr_NH1(mol)
    if secondary_amine:
        descriptions.append(f"- {secondary_amine} secondary amine group(s) (–NH–).")

    tertiary_amine = Fragments.fr_NH0(mol)
    if tertiary_amine:
        descriptions.append(f"- {tertiary_amine} tertiary amine group(s) (–N–).")

    if not descriptions:
        descriptions.append("- No key DRD2-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_deco_hop_features(mol):
    descriptions = []

    sulfonamide_count = Fragments.fr_sulfonamd(mol)
    if sulfonamide_count:
        descriptions.append(f"- {sulfonamide_count} sulfonamide group(s) (–SO₂NH₂).")

    amide_count = Fragments.fr_amide(mol)
    if amide_count:
        descriptions.append(f"- {amide_count} amide group(s) (–C(=O)–NH–).")

    aromatic_nitrogen = Fragments.fr_Ar_N(mol)
    if aromatic_nitrogen:
        descriptions.append(f"- {aromatic_nitrogen} aromatic nitrogen atom(s) in the ring.")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s).")

    if not descriptions:
        descriptions.append("- No key deco_hop-like fragments found.")

    res = "\n".join(descriptions)
    return res


def describe_amlodipine_features(mol):
    descriptions = []

    ester_count = Fragments.fr_ester(mol)
    if ester_count:
        descriptions.append(f"- {ester_count} ester group(s) (–COO– linkage).")

    carbonyl_count = Fragments.fr_COO(mol)
    if carbonyl_count:
        descriptions.append(f"- {carbonyl_count} carbonyl-containing group(s) (e.g., ester C=O).")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s).")

    aliphatic_amines = Fragments.fr_NH0(mol)
    if aliphatic_amines:
        descriptions.append(f"- {aliphatic_amines} tertiary amine group(s) (no N–H bond).")

    if not descriptions:
        descriptions.append("- No key amlodipine-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_celecoxib_features(mol):
    descriptions = []

    # Detect sulfonamide group (-SO2NH-)
    sulfonamide_count = Fragments.fr_sulfonamd(mol)
    if sulfonamide_count:
        descriptions.append(f"- {sulfonamide_count} sulfonamide group(s) (important for celecoxib's bioactivity).")

    # Detect pyrazole ring (5-membered ring with 2 nitrogens)
    pyrazole_smarts = Chem.MolFromSmarts("n1nccc1")  # Simple pyrazole core
    pyrazole_matches = mol.GetSubstructMatches(pyrazole_smarts)
    if pyrazole_matches:
        descriptions.append(f"- {len(pyrazole_matches)} pyrazole ring(s) (5-membered N-heterocyclic rings).")

    # Detect benzene rings
    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s) (providing hydrophobicity).")

    # Detect aryl-sulfonamide linkage via SMARTS: aromatic C-S(=O)(=O)-N
    aromatic_sulfonamide_smarts = Chem.MolFromSmarts("cS(=O)(=O)N")
    aromatic_sulfonamide_matches = mol.GetSubstructMatches(aromatic_sulfonamide_smarts)
    if aromatic_sulfonamide_matches:
        descriptions.append(f"- {len(aromatic_sulfonamide_matches)} aryl-sulfonamide linkage(s) (aromatic ring connected to sulfonamide group).")

    # Detect para-substitution pattern
    para_disubstitution = Fragments.fr_para_substituted_benzene(mol) if hasattr(Fragments, 'fr_para_substituted_benzene') else 0
    if para_disubstitution:
        descriptions.append(f"- {para_disubstitution} para-disubstituted benzene ring(s) (common in celecoxib).")

    if not descriptions:
        descriptions.append("- No key celecoxib-like fragments found.")

    res = "\n".join(descriptions)
    return res

def describe_albuterol_features(mol):
    descriptions = []

    phenol_count = Fragments.fr_phenol(mol)
    if phenol_count:
        descriptions.append(f"- {phenol_count} phenol group(s) (aromatic hydroxyl).")

    aromatic_oh = Fragments.fr_Ar_OH(mol)
    if aromatic_oh:
        descriptions.append(f"- {aromatic_oh} aromatic hydroxyl group(s).")

    secondary_amine = Fragments.fr_NH1(mol)
    if secondary_amine:
        descriptions.append(f"- {secondary_amine} secondary amine group(s).")

    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s), possibly benzylic alcohol.")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s).")

    aryl_methyl = Fragments.fr_aryl_methyl(mol)
    if aryl_methyl:
        descriptions.append(f"- {aryl_methyl} aryl methyl group(s), which may relate to ring substituents.")

    if not descriptions:
        descriptions.append("- No key albuterol-like fragments found.")

    res = "\n".join(descriptions)
    return res

import re

def get_reviewer_think_dict(response: str):
    response = response.strip()
    step_thinking = {}

    # Find all step matches
    step_matches = list(re.finditer(r"(Step\s*\d+)\s*Feedback:\s*", response))

    for i in range(len(step_matches)):
        step_key = step_matches[i].group(1).lower().replace(" ", "")  # e.g., step1
        start = step_matches[i].end()

        if i + 1 < len(step_matches):
            end = step_matches[i + 1].start()
        else:
            end = len(response)

        step_content = response[start:end].strip()
        step_thinking[step_key] = step_content

    return step_thinking


def get_scientist_think_dict(response: str):
    response = response.strip()
    step_thinking = {}

    # Find all step headers
    step_matches = list(re.finditer(r"(Step\d+):\s*(.+?)\n", response))

    # Extract each step section
    for i in range(len(step_matches)):
        step_key = step_matches[i].group(1).lower()  # e.g., step1
        start = step_matches[i].start()

        if i + 1 < len(step_matches):
            end = step_matches[i + 1].start()
        else:
            end = len(response)

        step_text = response[start:end].strip()
        step_thinking[step_key] = step_text

    # Extract SMILES string from "Final Output" section
    smiles_match = re.search(r"Final Output:\s*SMILES:\s*([^\s]+)", response)
    if smiles_match:
        step_thinking["smiles""smiles"] = smiles_match.group(1)

    return step_thinking


import torch
from typing import List

def compute_auc_topk_online_torch(score_list: List[float], k: int = 10) -> float:
    """
    Compute the AUC of top-k average scores vs number of oracle calls using torch.

    Args:
        score_list: List of scores from oracle evaluations (one per call)
        topk: Number of top scores to average

    Returns:
        AUC value normalized to [0, 1]
    """
    if len(score_list) == 0:
        return 0.0

    scores = torch.tensor(score_list)
    topk_curve = []

    for i in range(1, len(scores) + 1):
        current_topk = torch.topk(scores[:i], min(k, i)).values
        avg_topk = current_topk.mean().item()
        topk_curve.append(avg_topk)

    curve_tensor = torch.tensor(topk_curve)
    auc = torch.trapz(curve_tensor, dx=1.0) / (len(score_list) * 1.0)

    # Normalize AUC to [0, 1] range using min-max (assuming scores ∈ [0, 1])
    return float(auc)



def get_pretty_description_str(description_list):
    # Parse into registry
    TOOL_REGISTRY = {}

    for line in description_list:
        try:
            name_match = re.match(r"^(.*?):", line)
            input_match = re.search(r"\(input:\s*(.*?)\s*,\s*output:", line)
            output_match = re.search(r"output:\s*(.*?)\)", line)
            desc_end = re.search(r"\(input:", line)

            if name_match:
                name = name_match.group(1).strip()
                description = line[len(name_match.group(0)):desc_end.start()].strip() if desc_end else line
                input_type = input_match.group(1).strip() if input_match else "Unknown"
                output_type = output_match.group(1).strip() if output_match else "Unknown"

                TOOL_REGISTRY[name] = {
                    "description": description,
                    "input_type": input_type,
                    "output_type": output_type
                }
        except Exception as e:
            print(f"Error parsing: {line} - {e}")

    # Convert to tool_list_str
    tool_list_str = "\n".join([
        f"{name}: {meta['description']} (input: {meta['input_type']}, output: {meta['output_type']})"
        for name, meta in TOOL_REGISTRY.items()
    ])

    return tool_list_str


# Step 1: Parse the tool file
def load_tool_descriptions(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tool_blocks = re.split(r"(?=Function: )", raw_text.strip())
    documents = []

    for block in tool_blocks:
        name_match = re.search(r"Function:\s*(.*)", block)
        desc_match = re.search(r"Description:\s*(.*)", block)
        input_match = re.search(r"Input Type\s*:\s*(.*)", block)
        output_match = re.search(r"Output Type\s*:\s*(.*)", block)

        if name_match:
            name = name_match.group(1).strip()
            desc = desc_match.group(1).strip() if desc_match else ""
            input_type = input_match.group(1).strip() if input_match else ""
            output_type = output_match.group(1).strip() if output_match else ""

            full_text = f"{name}: {desc} (input: {input_type}, output: {output_type})"
            documents.append(Document(page_content=full_text, metadata={"tool_name": name}))
    return documents

# Step 2: Build vectorstore
def build_vectorstore(documents):
    # embedding_model = OpenAIEmbeddings()  # or DeepSeekEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(documents, embedding_model)

# Step 3: Retrieve top-K relevant tools
def retrieve_top_k_tools(vectorstore, query: str, k=20):
    top_docs = vectorstore.similarity_search(query, k=k)
    top_descriptions = [doc.page_content for doc in top_docs]
    top_tool_names = [doc.metadata["tool_name"] for doc in top_docs]
    return top_descriptions, top_tool_names


def get_pretty_topk_string(topk_dict, max_smiles_length, property_name):
    res = ""
    for item in topk_dict:
        res += f"SMILES: {item['smiles']}  | {property_name}: {float(item[property_name]):.5f}\n"

    return res

def get_scientist_output_dict(scientist_output):
    step_pattern = re.compile(r"Step (\d+):.*?\n(.*?)(?=\nStep \d+:|\nFinal proposed SMILES:|\Z)", re.DOTALL)
    steps = {f"step{m.group(1)}": m.group(2).strip() for m in step_pattern.finditer(scientist_output)}

    smiles_match = re.search(r"Final proposed SMILES:\s*\n?([^\s<]+)", scientist_output)
    steps["smiles"] = smiles_match.group(1).strip() if smiles_match else None
    return steps

def get_reviewer_output_dict(reviewer_output):
    # Extract feedback for each step using a non-greedy match until the next "Step X:" or end of string
    # Use regex to extract feedback for each step
    pattern = r"Step (\d+): (.*?)(?=(?:\n)?Step \d+:|\Z)"
    matches = re.findall(pattern, reviewer_output, re.DOTALL)

    # Build feedback dictionary
    feedback_dict = {f"step{step}": feedback.strip().replace("\n", " ") for step, feedback in matches}
    return feedback_dict


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)
    except:
        return None   


    if len(smiles) == 0:
        return None

    return smiles

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_files_in_dir(dir, specs=None):
    if specs is None:
        return natural_sort(glob.glob(os.path.join(dir,"*")))
    else:
        return natural_sort(glob.glob(os.path.join(dir,specs)))
    

# Metrics
def get_rmse(target, predicted):
    return torch.square(torch.subtract(target, predicted)).mean().item()

def get_mae(target, predicted):
    return torch.abs(torch.subtract(target, predicted)).item()

# log results
def save_results(logger, log_dir, iteration, smiles, molweight, diff, property_unit, property_name, min_diff):
    smiles_file = os.path.join(log_dir, f"{iteration}_smiles.txt")
    molweight_file = os.path.join(log_dir, f"{iteration}_{property_name}.txt")
    image_file = os.path.join(log_dir, f"{iteration}_structure.jpg")
    
    # Save SMILES string
    with open(smiles_file, "w") as f:
        f.write(smiles)
    
    # Save molweight
    with open(molweight_file, "w") as f:
        f.write(f"{molweight} {property_unit}")
    
    # Generate molecule image
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(image_file)
    
    # Log results to wandb\
    logger.log({
        "iteration": iteration,
        "smiles": smiles.strip(),
        property_name: molweight,
        f"{property_name}_diff": diff,
        "min_diff": min_diff,
        "structure_image": wandb.Image(image_file)
    })