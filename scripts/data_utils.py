import copy
import glob
import json
import os
import enrichment_utils

def load_raw_advbench():
    # Read from advbench.json file
    with open('/workspace/refusal_direction/dataset/processed/advbench.json', 'r') as file:
        return json.load(file)

def load_advbench():
    advbench_data = load_raw_advbench()

    # Advbench instructions don't include punctuation marks for some reason.
    punctuated_advbench_data = copy.deepcopy(advbench_data)
    for item in punctuated_advbench_data:
        item['instruction'] += '.'
    return punctuated_advbench_data

def load_alpaca():
    # Read from advbench.json file
    with open('/workspace/refusal_direction/dataset/processed/alpaca.json', 'r') as file:
        return json.load(file)

def load_frac_active(sae_name, sae_id, suffix, which_tokens):
    if which_tokens not in ('last', 'all'):
        raise ValueError(f"which_tokens={which_tokens} is invalid, must be 'last' or 'all")
    pattern = f'../data/sae_frac_active_chat/{sae_name}/{sae_id}_{suffix}_*_{which_tokens}.pt'
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files found matching glob {pattern}")
    if len(files) > 1:
        raise ValueError(f"Too many files found matching glob {pattern}: {files}")    
    file, = files
    base, length, which = os.path.basename(file).rsplit('_', 2)
    return enrichment_utils.load_tensor(file), len(length)
