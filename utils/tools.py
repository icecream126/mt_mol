import itertools
from operator import itemgetter
from typing import List, Tuple

import pandas as pd
import pystow
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from tqdm.auto import tqdm

from typing import List, Callable, Tuple
from typing import Union

import numpy as np
import pandas as pd
from rdkit import Chem



from rdkit.Chem import Descriptors

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit import Chem, DataStructs, RDLogger

import numpy as np

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
# from utils.utils import gen_conformers, refine_conformers, get_conformer_energies
# import utils.utils
# from utils.metrics import *
import pandas as pd


import json
from dataclasses import dataclass
from pathlib import Path

from rdkit.Chem import inchi
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, inchi

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

def get_rdkit_complexity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(Descriptors.BertzCT(mol))

def get_rdkit_number_of_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(mol.GetNumAtoms())

def get_rdkit_number_of_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(mol.GetNumBonds())

def get_rdkit_rotatable_bond_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(Lipinski.NumRotatableBonds(mol))

def get_rdkit_h_bond_donor_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(Lipinski.NumHDonors(mol))

def get_rdkit_h_bond_acceptor_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(Lipinski.NumHAcceptors(mol))

def get_rdkit_molecular_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(rdMolDescriptors.CalcMolFormula(mol))

def get_rdkit_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(Chem.MolToSmiles(mol, canonical=True))

def get_rdkit_inchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return str(inchi.MolToInchi(mol))

def get_center(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    mol = gen_conformers(mol, num_confs=1)
    if mol is None or mol.GetNumConformers() == 0:
        return "[Error] No conformers found"
    center = np.array(ComputeCentroid(mol.GetConformer(0)))
    return f"Geometric center: {center.round(3).tolist()}"


def get_shape_moments(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    mol = gen_conformers(mol, num_confs=1)
    if mol is None or mol.GetNumConformers() == 0:
        return "[Error] No conformers found"
    return f"NPR1: {NPR1(mol):.4f}, NPR2: {NPR2(mol):.4f}"


def refine_conformers(smi: str, energy_threshold: float = 50.0, rms_threshold: Optional[float] = 0.5) -> str:
    mol = Chem.MolFromSmiles(smi)
    mol = gen_conformers(mol, num_confs=10)
    if mol is None or mol.GetNumConformers() == 0:
        return "[Error] Failed to generate conformers"
    before = mol.GetNumConformers()
    mol = refine_conformers(mol, energy_threshold, rms_threshold)
    after = mol.GetNumConformers()
    return f"Refined conformers: from {before} to {after} conformers."


def get_conformer_energies(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    mol = gen_conformers(mol, num_confs=10)
    if mol is None or mol.GetNumConformers() == 0:
        return "[Error] No conformers to retrieve energies from"
    energies = get_conformer_energies(mol)
    return f"Conformer energies: {[round(e, 4) for e in energies]}"



class RingSystemFinder:

    def __init__(self):
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#16R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            bnd.SetBoolProp("protected", True)

    @staticmethod
    def cleave_linker_bonds(mol):
        frag_bond_list = []
        for bnd in mol.GetBonds():
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())
        if frag_bond_list:
            frag_mol = Chem.FragmentOnBonds(mol, frag_bond_list)
            Chem.SanitizeMol(frag_mol)
            return frag_mol
        return mol

    def cleanup_fragments(self, mol, keep_dummy=False):
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        if keep_dummy:
                            atm.SetProp("atomLabel", "R")
                        else:
                            atm.SetAtomicNum(1)
                        atm.SetIsotope(0)
                frag = Chem.RemoveAllHs(frag)
                frag = self.fix_bond_stereo(frag)
                ring_system_list.append(frag)
        return ring_system_list

    @staticmethod
    def fix_bond_stereo(mol):
        for bnd in mol.GetBonds():
            if bnd.GetBondType() == Chem.BondType.DOUBLE:
                if bnd.GetBeginAtom().GetDegree() == 1 or bnd.GetEndAtom().GetDegree() == 1:
                    bnd.SetStereo(Chem.BondStereo.STEREONONE)
        return mol


    def find_ring_systems(self, mol, keep_dummy=False, as_mols=False):
        self.tag_bonds_to_preserve(mol)
        frag_mol = self.cleave_linker_bonds(mol)
        output_list = self.cleanup_fragments(frag_mol, keep_dummy=keep_dummy)
        if not as_mols:
            output_list = [Chem.MolToSmiles(x) for x in output_list]
        return output_list
    
class RingSystemLookup:
    def __init__(self, ring_file=None, ignore_stereo=False):
        ring_csv_name = "chembl_ring_systems.csv"
        if ignore_stereo:
            ring_csv_name = ring_csv_name.replace(".csv", "_no_stereo.csv")
        if ring_file is None:
            # url = f'https://github.com/PatWalters/useful_rdkit_utils/tree/master/data/{ring_csv_name}'

            self.rule_path = '/home/anonymous/mt_mol/dataset/chembl_ring_systems.csv'
        else:
            self.rule_path = ring_file
        self.ignore_stereo = ignore_stereo
        print('self.rule_path: ', self.rule_path)
        self.ring_df = pd.read_csv(self.rule_path)
        self.ring_dict = dict(self.ring_df[["InChI", "Count"]].values)
        self.ring_system_finder = RingSystemFinder()
        self.enumerator = rdMolStandardize.TautomerEnumerator()

    def process_smiles(self, smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "[Error] Invalid SMILES"
        if self.ignore_stereo:
            Chem.RemoveStereochemistry(mol)
        ring_systems = self.ring_system_finder.find_ring_systems(mol, as_mols=True)
        results = []
        for ring in ring_systems:
            smiles = Chem.MolToSmiles(ring)
            inchi = Chem.MolToInchiKey(ring)
            count = self.ring_dict.get(inchi, 0)
            results.append((smiles, count))
        if not results:
            return "[Info] No known ring systems found."
        return "\n".join([f"{smi}\tCount: {cnt}" for smi, cnt in results])

def get_min_ring_frequency(smi: str) -> str:
    ring_lookup = RingSystemLookup(ignore_stereo=False)
    ring_list = ring_lookup.process_smiles(smi)
    if not ring_list or isinstance(ring_list, str):
        return "[Info] No ring systems found."
    ring_data = ring_lookup.process_smiles(smi)
    if not ring_data:
        return "[Info] No ring data available."
    ring_data.sort(key=itemgetter(1))
    ring, count = ring_data[0]
    return f"Min frequency ring: {ring}, Count: {count}"


def remove_stereo_from_smiles(smi_in: str) -> str:
    mol = Chem.MolFromSmiles(smi_in)
    if mol is None:
        return "[Error] Invalid SMILES"
    Chem.RemoveStereochemistry(mol)
    smi_out = Chem.MolToSmiles(mol)
    inchi_key = Chem.MolToInchiKey(mol)
    return f"SMILES without stereo: {smi_out}\nInChI Key: {inchi_key}"


def get_spiro_atoms(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "[Error] Invalid SMILES"
    info = mol.GetRingInfo()
    ring_sets = [set(x) for x in info.AtomRings()]
    spiro_atoms = []
    for i, j in itertools.combinations(ring_sets, 2):
        inter = i & j
        if len(inter) == 1:
            spiro_atoms += list(inter)
    return f"Spiro atom indices: {sorted(set(spiro_atoms))}" if spiro_atoms else "[Info] No spiro atoms."


def max_ring_size(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "[Error] Invalid SMILES"
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    return f"Max ring size: {max(map(len, atom_rings)) if atom_rings else 0}"


def ring_stats(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "[Error] Invalid SMILES"
    max_size = max_ring_size(smi).split(": ")[-1]
    num_rings = CalcNumRings(mol)
    return f"Number of rings: {num_rings}\nMax ring size: {max_size}"


def smi2mol_with_errors(smi: str) -> str:
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromSmiles(smi)
    err = sio.getvalue()
    sys.stderr = sys.__stderr__
    if mol is None:
        return f"[Error] Invalid SMILES: {smi}\n{err.strip()}"
    return f"[Success] Valid molecule.\nWarnings/Errors: {err.strip()}"


def count_fragments(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "[Error] Invalid SMILES"
    frag_count = len(Chem.GetMolFrags(mol, asMols=True))
    return f"Number of fragments: {frag_count}"


def get_largest_fragment(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "[Error] Invalid SMILES"
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    if not frag_list:
        return "No fragments found"
    frag_mw_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_mw_list.sort(key=itemgetter(0), reverse=True)
    largest = frag_mw_list[0][1]
    return f"Largest fragment SMILES: {Chem.MolToSmiles(largest)}"


# ==== Fragments ====
#TODO: Change the below functions' input to be smiles
def fr_al_coo(smiles): return Fragments.fr_Al_COO(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_al_oh(smiles): return Fragments.fr_Al_OH(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_al_oh_notert(smiles): return Fragments.fr_Al_OH_noTert(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_arn(smiles): return Fragments.fr_ArN(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ar_coo(smiles): return Fragments.fr_Ar_COO(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ar_n(smiles): return Fragments.fr_Ar_N(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ar_nh(smiles): return Fragments.fr_Ar_NH(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ar_oh(smiles): return Fragments.fr_Ar_OH(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_coo(smiles): return Fragments.fr_COO(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_coo2(smiles): return Fragments.fr_COO2(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_c_o(smiles): return Fragments.fr_C_O(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_c_o_nocoo(smiles): return Fragments.fr_C_O_noCOO(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_c_s(smiles): return Fragments.fr_C_S(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_hoccn(smiles): return Fragments.fr_HOCCN(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_imine(smiles): return Fragments.fr_Imine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nh0(smiles): return Fragments.fr_NH0(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nh1(smiles): return Fragments.fr_NH1(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nh2(smiles): return Fragments.fr_NH2(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_n_o(smiles): return Fragments.fr_N_O(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ndealk1(smiles): return Fragments.fr_Ndealkylation1(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ndealk2(smiles): return Fragments.fr_Ndealkylation2(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nhpyrrole(smiles): return Fragments.fr_Nhpyrrole(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_sh(smiles): return Fragments.fr_SH(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_aldehyde(smiles): return Fragments.fr_aldehyde(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_alkyl_carbamate(smiles): return Fragments.fr_alkyl_carbamate(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_alkyl_halide(smiles): return Fragments.fr_alkyl_halide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_allylic_oxid(smiles): return Fragments.fr_allylic_oxid(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_amide(smiles): return Fragments.fr_amide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_amidine(smiles): return Fragments.fr_amidine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_aniline(smiles): return Fragments.fr_aniline(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_aryl_methyl(smiles): return Fragments.fr_aryl_methyl(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_azide(smiles): return Fragments.fr_azide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_azo(smiles): return Fragments.fr_azo(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_barbitur(smiles): return Fragments.fr_barbitur(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_benzene(smiles): return Fragments.fr_benzene(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_benzodiazepine(smiles): return Fragments.fr_benzodiazepine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_bicyclic(smiles): return Fragments.fr_bicyclic(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_diazo(smiles): return Fragments.fr_diazo(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_dihydropyridine(smiles): return Fragments.fr_dihydropyridine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_epoxide(smiles): return Fragments.fr_epoxide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ester(smiles): return Fragments.fr_ester(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ether(smiles): return Fragments.fr_ether(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_furan(smiles): return Fragments.fr_furan(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_guanido(smiles): return Fragments.fr_guanido(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_halogen(smiles): return Fragments.fr_halogen(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_hdrzine(smiles): return Fragments.fr_hdrzine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_hdrzone(smiles): return Fragments.fr_hdrzone(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_imidazole(smiles): return Fragments.fr_imidazole(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_imide(smiles): return Fragments.fr_imide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_isocyan(smiles): return Fragments.fr_isocyan(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_isothiocyan(smiles): return Fragments.fr_isothiocyan(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ketone(smiles): return Fragments.fr_ketone(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_ketone_topliss(smiles): return Fragments.fr_ketone_Topliss(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_lactam(smiles): return Fragments.fr_lactam(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_lactone(smiles): return Fragments.fr_lactone(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_methoxy(smiles): return Fragments.fr_methoxy(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_morpholine(smiles): return Fragments.fr_morpholine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nitrile(smiles): return Fragments.fr_nitrile(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nitro(smiles): return Fragments.fr_nitro(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nitro_arom(smiles): return Fragments.fr_nitro_arom(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nitro_arom_nonortho(smiles): return Fragments.fr_nitro_arom_nonortho(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_nitroso(smiles): return Fragments.fr_nitroso(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_oxazole(smiles): return Fragments.fr_oxazole(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_oxime(smiles): return Fragments.fr_oxime(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_para_hydroxylation(smiles): return Fragments.fr_para_hydroxylation(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_phenol(smiles): return Fragments.fr_phenol(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_phenol_noorthohbond(smiles): return Fragments.fr_phenol_noOrthoHbond(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_phos_acid(smiles): return Fragments.fr_phos_acid(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_phos_ester(smiles): return Fragments.fr_phos_ester(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_piperdine(smiles): return Fragments.fr_piperdine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_piperzine(smiles): return Fragments.fr_piperzine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_priamide(smiles): return Fragments.fr_priamide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_prisulfonamd(smiles): return Fragments.fr_prisulfonamd(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_pyridine(smiles): return Fragments.fr_pyridine(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_quatn(smiles): return Fragments.fr_quatN(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_sulfide(smiles): return Fragments.fr_sulfide(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_sulfonamd(smiles): return Fragments.fr_sulfonamd(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_sulfone(smiles): return Fragments.fr_sulfone(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_term_acetylene(smiles): return Fragments.fr_term_acetylene(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_tetrazole(smiles): return Fragments.fr_tetrazole(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_thiazole(smiles): return Fragments.fr_thiazole(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_thiocyan(smiles): return Fragments.fr_thiocyan(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_thiophene(smiles): return Fragments.fr_thiophene(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_unbrch_alkane(smiles): return Fragments.fr_unbrch_alkane(Chem.MolFromSmiles(smiles), countUnique=True)
def fr_urea(smiles): return Fragments.fr_urea(Chem.MolFromSmiles(smiles), countUnique=True)


# ==== rdMolDescriptors ====
def bcut2d(smiles):
    return rdMolDescriptors.BCUT2D(Chem.MolFromSmiles(smiles))

def calcautocorr2d(smiles, CustomAtomProperty="GasteigerCharges"):
    return rdMolDescriptors.CalcAUTOCORR2D(Chem.MolFromSmiles(smiles), CustomAtomProperty)

def calcchi0n(smiles, force=None):
    return rdMolDescriptors.CalcChi0n(Chem.MolFromSmiles(smiles), force)

def calcchi0v(smiles, force=None):
    return rdMolDescriptors.CalcChi0v(Chem.MolFromSmiles(smiles), force)

def calcchi1n(smiles, force=None):
    return rdMolDescriptors.CalcChi1n(Chem.MolFromSmiles(smiles), force)

def calcchi1v(smiles, force=None):
    return rdMolDescriptors.CalcChi1v(Chem.MolFromSmiles(smiles), force)

def calcchi2n(smiles, force=None):
    return rdMolDescriptors.CalcChi2n(Chem.MolFromSmiles(smiles), force)

def calcchi2v(smiles, force=None):
    return rdMolDescriptors.CalcChi2v(Chem.MolFromSmiles(smiles), force)

def calcchi3n(smiles, force=None):
    return rdMolDescriptors.CalcChi3n(Chem.MolFromSmiles(smiles), force)

def calcchi3v(smiles, force=None):
    return rdMolDescriptors.CalcChi3v(Chem.MolFromSmiles(smiles), force)

def calcchi4n(smiles, force=None):
    return rdMolDescriptors.CalcChi4n(Chem.MolFromSmiles(smiles), force)

def calcchi4v(smiles, force=None):
    return rdMolDescriptors.CalcChi4v(Chem.MolFromSmiles(smiles), force)

def calccrippendescriptors(smiles, includeHs=None, force=None):
    return rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(smiles), includeHs, force)

def calcexactmolwt(smiles, onlyHeavy=None):
    return rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smiles), onlyHeavy)

def calcfractioncsp3(smiles):
    return rdMolDescriptors.CalcFractionCSP3(Chem.MolFromSmiles(smiles))

def calckappa1(smiles):
    return rdMolDescriptors.CalcKappa1(Chem.MolFromSmiles(smiles))

def calckappa2(smiles):
    return rdMolDescriptors.CalcKappa2(Chem.MolFromSmiles(smiles))

def calckappa3(smiles):
    return rdMolDescriptors.CalcKappa3(Chem.MolFromSmiles(smiles))

def calclabuteasa(smiles, includeHs=None, force=None):
    return rdMolDescriptors.CalcLabuteASA(Chem.MolFromSmiles(smiles), includeHs, force)

def calcmolformula(smiles, separateIsotopes=None, abbreviateHIsotopes=None):
    return rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles), separateIsotopes, abbreviateHIsotopes)

def calcnumaliphaticcarbocycles(smiles):
    return rdMolDescriptors.CalcNumAliphaticCarbocycles(Chem.MolFromSmiles(smiles))

def calcnumaliphaticheterocycles(smiles):
    return rdMolDescriptors.CalcNumAliphaticHeterocycles(Chem.MolFromSmiles(smiles))

def calcnumaliphaticrings(smiles):
    return rdMolDescriptors.CalcNumAliphaticRings(Chem.MolFromSmiles(smiles))

def calcnumamidebonds(smiles):
    return rdMolDescriptors.CalcNumAmideBonds(Chem.MolFromSmiles(smiles))

def calcnumaromaticcarbocycles(smiles):
    return rdMolDescriptors.CalcNumAromaticCarbocycles(Chem.MolFromSmiles(smiles))

def calcnumaromaticheterocycles(smiles):
    return rdMolDescriptors.CalcNumAromaticHeterocycles(Chem.MolFromSmiles(smiles))

def calcnumaromaticrings(smiles):
    return rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(smiles))

def calcnumatomstereocenters(smiles):
    return rdMolDescriptors.CalcNumAtomStereoCenters(Chem.MolFromSmiles(smiles))

def calcnumatoms(smiles):
    return rdMolDescriptors.CalcNumAtoms(Chem.MolFromSmiles(smiles))

def calcnumhba(smiles):
    return rdMolDescriptors.CalcNumHBA(Chem.MolFromSmiles(smiles))

def calcnumhbd(smiles):
    return rdMolDescriptors.CalcNumHBD(Chem.MolFromSmiles(smiles))

def calcnumheavyatoms(smiles):
    return rdMolDescriptors.CalcNumHeavyAtoms(Chem.MolFromSmiles(smiles))

def calcnumheteroatoms(smiles):
    return rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(smiles))

def calcnumheterocycles(smiles):
    return rdMolDescriptors.CalcNumHeterocycles(Chem.MolFromSmiles(smiles))

def calcnumlipinskihba(smiles):
    return rdMolDescriptors.CalcNumLipinskiHBA(Chem.MolFromSmiles(smiles))

def calcnumlipinskihbd(smiles):
    return rdMolDescriptors.CalcNumLipinskiHBD(Chem.MolFromSmiles(smiles))

def calcnumrings(smiles):
    return rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smiles))

def calcnumrotatablebonds(smiles, strict=None):
    return rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(smiles), strict)

def calcnumsaturatedcarbocycles(smiles):
    return rdMolDescriptors.CalcNumSaturatedCarbocycles(Chem.MolFromSmiles(smiles))

def calcnumsaturatedheterocycles(smiles):
    return rdMolDescriptors.CalcNumSaturatedHeterocycles(Chem.MolFromSmiles(smiles))

def calcnumsaturatedrings(smiles):
    return rdMolDescriptors.CalcNumSaturatedRings(Chem.MolFromSmiles(smiles))

def calcnumunspecifiedatomstereocenters(smiles):
    return rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(Chem.MolFromSmiles(smiles))

def calcoxidationnumbers(smiles):
    return rdMolDescriptors.CalcOxidationNumbers(Chem.MolFromSmiles(smiles))

def calcpbf(smiles, confId=None):
    return rdMolDescriptors.CalcPBF(Chem.MolFromSmiles(smiles), confId)

def calcphi(smiles):
    return rdMolDescriptors.CalcPhi(Chem.MolFromSmiles(smiles))

def getconnectivityinvariants(smiles, includeRingMembership=None):
    return rdMolDescriptors.GetConnectivityInvariants(Chem.MolFromSmiles(smiles), includeRingMembership)

def getfeatureinvariants(smiles):
    return rdMolDescriptors.GetFeatureInvariants(Chem.MolFromSmiles(smiles))

def gethashedatompairfingerprint(
    smiles,
    nBits=1024,
    minLength=1,
    maxLength=30,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False,
    use2D=True,
    confId=-1
):
    return list(rdMolDescriptors.GetHashedAtomPairFingerprint(
        Chem.MolFromSmiles(smiles), nBits, minLength, maxLength,
        fromAtoms, ignoreAtoms, atomInvariants,
        includeChirality, use2D, confId
    ))

def gethashedatompairfingerprintasbitvect(
    smiles,
    nBits=1024,
    minLength=1,
    maxLength=30,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    nBitsPerEntry=4,
    includeChirality=False,
    use2D=True,
    confId=-1
):
    return list(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), nBits, minLength, maxLength,
        fromAtoms, ignoreAtoms, atomInvariants,
        nBitsPerEntry, includeChirality, use2D, confId
    ))

def gethashedmorganfingerprint(
    smiles,
    radius=2,
    nBits=1024,
    invariants=None,
    fromAtoms=None,
    useChirality=False,
    useBondTypes=True,
    useFeatures=False,
    bitInfo=None,
    includeRedundantEnvironments=False
):
    return list(rdMolDescriptors.GetHashedMorganFingerprint(
        Chem.MolFromSmiles(smiles), radius, nBits, invariants, fromAtoms,
        useChirality, useBondTypes, useFeatures,
        bitInfo, includeRedundantEnvironments
    ))

def gethashedtopologicaltorsionfingerprint(
    smiles,
    nBits=1024,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False
):
    return list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(
        Chem.MolFromSmiles(smiles), nBits, targetSize, fromAtoms,
        ignoreAtoms, atomInvariants, includeChirality
    ))

def gethashedtopologicaltorsionfingerprintasbitvect(
    smiles,
    nBits=1024,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    nBitsPerEntry=4,
    includeChirality=False
):
    return list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), nBits, targetSize, fromAtoms,
        ignoreAtoms, atomInvariants,
        nBitsPerEntry, includeChirality
    ))

def getmaccskeysfingerprint(smiles):
    fp=rdMolDescriptors.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smiles))
    return list(fp.GetOnBits())

def getmorganfingerprint(smiles, radius=2):
    fp = rdMolDescriptors.GetMorganFingerprint(Chem.MolFromSmiles(smiles), radius=2)
    return fp.GetNonzeroElements()

def getmorganfingerprintasbitvect(
    smiles,
    radius=2,
    nBits=1024,
    invariants=None,
    fromAtoms=None,
    useChirality=False,
    useBondTypes=True,
    useFeatures=False,
    bitInfo=None,
    includeRedundantEnvironments=False
):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nBits)
    return list(fp.GetOnBits())# , nBits, invariants, fromAtoms, useChirality, useBondTypes, useFeatures, bitInfo, includeRedundantEnvironments))

def gettopologicaltorsionfingerprint(
    smiles,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False
):
    fp = rdMolDescriptors.GetTopologicalTorsionFingerprint(Chem.MolFromSmiles(smiles), targetSize=4)
    return fp.GetNonzeroElements()

def mqns_(smiles, force=None):
    return rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smiles), force)

def peoe_vsa_(smiles, bins=None, force=None):
    return rdMolDescriptors.PEOE_VSA_(Chem.MolFromSmiles(smiles), bins, force)

def smr_vsa_(smiles, bins=None, force=None):
    return rdMolDescriptors.SMR_VSA_(Chem.MolFromSmiles(smiles), bins, force)

def slogp_vsa_(smiles, bins=None, force=None):
    return rdMolDescriptors.SlogP_VSA_(Chem.MolFromSmiles(smiles), bins, force)
if __name__=="__main__":

    import inspect
    import sys

    # Create test mol
    # mol = Chem.MolFromSmiles("CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
    mol = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"

    print(f"calcnumaromaticrings: {calcnumaromaticrings(mol)}")
    print(f"calcnumhba: {calcnumhba(mol)}")
    print(f"calcnumrotatablebonds: {calcnumrotatablebonds(mol)}")

    print()
    print(f"fr_COO: {fr_coo(mol)}")
    print(f"fr_benzene: {fr_benzene(mol)}")
    print(f"fr_NH0: {fr_nh0(mol)}")

    print()
    print(f"get_molecular_formula: {get_rdkit_molecular_formula(mol)}")
    print(f"get_inchi: {get_rdkit_inchi(mol)}")
    print(f"get_canonical_smiles: {get_rdkit_canonical_smiles(mol)}")

    print()
    print(f"bcut_2d: {bcut2d(mol)}")
    print(f"calc_autocorr_2d: {calcautocorr2d(mol)}")
    print(f"calc_chi_1v: {calcchi1v(mol)}")

    print()
    print(f"calc_labute_asa: {calclabuteasa(mol)}")
    print(f"calc_crippen_descriptors: {calccrippendescriptors(mol)}")
    print(f"get_connectivity_invariants: {getconnectivityinvariants(mol)}")


