# https://tdcommons.ai/functions/oracles/
from guacamol.standard_benchmarks import isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, similarity, amlodipine_rings, decoration_hop
from tdc import Oracle
import sys

def get_isomers_c7h8n2o2_score(smiles: str):
    oracle = Oracle(name = 'Isomers_C7H8N2O2')
    return oracle(smiles)

def get_isomers_c9h10n2o2pf2cl_score(smiles: str):
    oracle = Oracle(name = 'Isomers_C9H10N2O2PF2Cl')
    return oracle(smiles)


def get_albuterol_similarity_score(smiles: str):
    oracle = Oracle(name = 'Albuterol_Similarity')
    return oracle(smiles)

def get_celecoxib_rediscovery_score(smiles: str):
    oracle = Oracle(name = 'Celecoxib_Rediscovery')
    return oracle(smiles)

def get_amlodipine_mpo_score(smiles: str):
    oracle = Oracle(name = 'Amlodipine_MPO')
    return oracle(smiles)

def get_deco_hop_score(smiles: str):
    oracle = Oracle(name = 'Deco Hop')
    return oracle(smiles)


def get_drd2_score(smiles: str):
    oracle = Oracle(name="DRD2")
    return oracle(smiles)


def get_fexofenadine_mpo_score(smiles: str):
    oracle = Oracle(name="Fexofenadine_MPO")
    return oracle(smiles)


def get_gsk3b_score(smiles: str):
    oracle = Oracle(name="GSK3B")
    return oracle(smiles)


def get_jnk3_score(smiles: str):
    oracle = Oracle(name="JNK3")
    return oracle(smiles)


def get_median1_score(smiles: str):
    oracle = Oracle(name="Median 1")
    return oracle(smiles)


def get_median2_score(smiles: str):
    oracle = Oracle(name="Median 2")
    return oracle(smiles)


def get_mestranol_similarity_score(smiles: str):
    oracle = Oracle(name="Mestranol_Similarity")
    return oracle(smiles)


def get_osimertinib_mpo_score(smiles: str):
    oracle = Oracle(name="Osimertinib_MPO")
    return oracle(smiles)


def get_perindopril_mpo_score(smiles: str):
    oracle = Oracle(name="Perindopril_MPO")
    return oracle(smiles)


def get_qed_score(smiles: str):
    oracle = Oracle(name="QED")
    return oracle(smiles)


def get_ranolazine_mpo_score(smiles: str):
    oracle = Oracle(name="Ranolazine_MPO")
    return oracle(smiles)


def get_scaffold_hop_score(smiles: str):
    oracle = Oracle(name="Scaffold Hop")
    return oracle(smiles)


def get_sitagliptin_mpo_score(smiles: str):
    oracle = Oracle(name="Sitagliptin_MPO")
    return oracle(smiles)


def get_thiothixene_rediscovery_score(smiles: str):
    oracle = Oracle(name="Thiothixene_Rediscovery")
    return oracle(smiles)


def get_troglitazon_rediscovery_score(smiles: str):
    oracle = Oracle(name="Troglitazone_Rediscovery")
    return oracle(smiles)


def get_valsartan_smarts_score(smiles: str):
    oracle = Oracle(name="Valsartan_SMARTS")
    return oracle(smiles)


def get_zaleplon_mpo_score(smiles: str):
    oracle = Oracle(name="Zaleplon_MPO")
    return oracle(smiles)
