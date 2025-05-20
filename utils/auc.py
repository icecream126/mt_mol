import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum_auc = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1]))
    buffer_max_idx = ordered_results[-1][1][1] # last components' [1][1]: score
    
    for idx in range(freq_log, min(buffer_max_idx, max_oracle_calls), freq_log):
        temp_result = [item for item in ordered_results if item[1][1] <= idx]
        if len(temp_result) == 0:
            continue
        top_n_now = np.mean(
            [item[1][0] for item in sorted(temp_result, key=lambda kv: kv[1][0], reverse=True)[:top_n]]
        )
        sum_auc += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    
    final_result = sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    top_n_now = np.mean([item[1][0] for item in final_result])
    sum_auc += (buffer_max_idx - called) * (top_n_now + prev) / 2

    if finish and buffer_max_idx < max_oracle_calls:
        sum_auc += (max_oracle_calls - buffer_max_idx) * top_n_now
    
    return sum_auc / max_oracle_calls

# ---------------------------
# Function to process the text file
# ---------------------------
def compute_topk_auc(smiles_scores, top_k=5, max_oracle_calls=1000, freq_log=1, finish=False, buffer_max_idx=1000):
    # smiles_scores = []
    
    print("number of generated smiles", len(smiles_scores))
    # Step 2: Remove invalid or duplicate SMILES
    seen = set()
    mol_buffer = {}
    for i, (smi, score) in enumerate(smiles_scores):
        mol = Chem.MolFromSmiles(smi)
        if mol:  # valid SMILES
            canonical_smi = Chem.MolToSmiles(mol)  # Canonicalize
            if canonical_smi not in seen:
                # Store as {canonical_smi: [score, generation index]}
                mol_buffer[canonical_smi] = [score, i + 1]
                seen.add(canonical_smi)
            else:
                continue  # duplicated canonical SMILES
        else:
            continue  # invalid SMILES

    # Step 3: Compute AUC
    auc = top_auc(mol_buffer, top_k, finish=finish, freq_log=freq_log, max_oracle_calls=max_oracle_calls)
    print(f"Top-{top_k} AUC Score: {auc:.4f}")
    return auc, mol_buffer

if __name__ == "__main__":
    # Example usage
    smiles_scores = [
        ("CCO", 0.9),
        ("CCN", 0.8),
        ("CCC", 0.7),
        ("CCO", 0.6),  # Duplicate
        ("CNO", 0.5),
        ("CNC", 0.4),
        ("CNO", 0.3),  # Duplicate
        ("CCO", 0.2),  # Duplicate
        ("CCN", 0.1),  # Duplicate
    ]
    
    finish= len(smiles_scores)>=1000
    top_k_auc, mol_buffer = compute_topk_auc(smiles_scores, top_k=10, max_oracle_calls=1000, freq_log=1, finish=finish, buffer_max_idx=len(smiles_scores))