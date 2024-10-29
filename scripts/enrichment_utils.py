# TODO(ebreck) Split these up into meaningful files.

import gc
import json
import os
import sys
from jaxtyping import Float
from torch import Tensor
import torch as t
from torch import Tensor, nn

from jaxtyping import Float, Int
import einops
import numpy as np
from scipy.stats import chi2_contingency, chi2
import torch as t
import requests
from sae_lens import (
    SAE,
    HookedSAETransformer,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"

project_script_path = os.path.abspath('../scripts')
if project_script_path not in sys.path: sys.path.append(project_script_path)

import model_utils

def load_tensor(filename):
    if device == "mps":
        tensor = t.load(filename, map_location="cpu")
        tensor.to(device, dtype=t.float32)
    else:
        tensor = t.load(filename)
    return tensor


def chi_squared_test(column):
    """
    Perform a chi-squared test.

    Parameters:
    column : list, numpy array, or pytorch tensor
        The observed counts of: [active_harmful, active_harmless, inactive_harmful, inactive_harmless]

    Returns:
    chi2_stat : float
        The chi-squared statistic.
    p_value : float
        The p-value of the test.
    dof : int
        Degrees of freedom.
    expected_freq : numpy array
        The expected frequencies table.
    """
    # print(column)
    table = einops.rearrange(column, "(b1 b2) -> b1 b2 ", b1=2)

    if (table[0].sum() == 0) | (table[1].sum() == 0):
        return 1        
    chi2_stat, p_value, dof, expected_freq = chi2_contingency(table)

    return p_value

def chi_square_test_latents(
    frac_active_harmful: Float[Tensor, 'd_sae'], 
    len_harmful: int, 
    frac_active_harmless: Float[Tensor, 'd_sae'], 
    len_harmless: int
) -> Float[Tensor, 'd_sae']:
    """
    Performs a chi-square test over the SAE latents.

    Parameters:
    - frac_active_harmful (float): Fraction of harmful prompts that the SAE latents are active.
    - len_harmful (int): Total number of harmful prompts.
    - frac_active_harmless (float): Fraction of harmless prompts that the SAE latents are active.
    - len_harmless (int): Total number of harmless prompts.

    Returns:
    - Float[Tensor, 'd_sae']: The result of the chi-squared test.
    """

    # Calculate inactive latents for both groups
    inactive_harmful = (1 - frac_active_harmful) * len_harmful
    inactive_harmless = (1 - frac_active_harmless) * len_harmless

    # Stack the activity data into a PyTorch tensor
    sae_latent_activity = t.stack([
        frac_active_harmful * len_harmful, 
        frac_active_harmless * len_harmless, 
        inactive_harmful, 
        inactive_harmless])

    # Apply chi-squared test along the specified axis
    result = np.apply_along_axis(chi_squared_test, axis=0, arr=sae_latent_activity.numpy())

    return t.from_numpy(result)


def get_second_min(x):
    min_value = t.min(x)
    mask = x != min_value
    second_min_value = t.min(x[mask])

    return second_min_value

def clear_memory_after(f):
    def func(*args, **kwargs):
        result = f(*args, **kwargs)
        gc.collect()
        t.cuda.empty_cache()
        return result
    return func

@clear_memory_after
def get_activations(prompts, sae_name, sae_id):
    t.set_grad_enabled(False)
    gemma2: HookedSAETransformer = HookedSAETransformer.from_pretrained("gemma-2-2b-it", device=device)
    gemma2_sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_name,
        sae_id=sae_id,
        device=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    all_sae_acts_post = []

    for prompt in tqdm(prompts):
        prompt = model_utils.get_chat_template(prompt, tokenizer)
        # Get top activations on final token
        _, cache = gemma2.run_with_cache_with_saes(
            prompt,
            saes=[gemma2_sae],
            stop_at_layer=gemma2_sae.cfg.hook_layer + 1,
        )
        sae_acts_post = cache[f"{gemma2_sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]
        all_sae_acts_post.append(sae_acts_post)

    return t.stack(all_sae_acts_post)

def get_frac_active(sae_activations):
    return (sae_activations > 0).sum(dim=(0)) / sae_activations.shape[0]

def smooth_frac_active(frac_active):
    # Smooth the vector to avoid zeroes by arbitrarily adding half the second
    # smallest value.  This allows dividing by the vector without errors.
    # This is arbitrary; find a better way...
    return t.where(frac_active == 0, get_second_min(frac_active)/2, frac_active)

def get_relative_activation(unsmoothed_frac_active_interest, unsmoothed_frac_active_baseline):
    # Rough statistic: a latent is "interesting" if its activation is big compared to
    # the activation in some baseline.
    return smooth_frac_active(unsmoothed_frac_active_interest)/smooth_frac_active(unsmoothed_frac_active_baseline)

# See https://www.neuronpedia.org/api-doc for instructions on how to get a Neuronpedia API key
# Save it in this file.
with open('/workspace/neuronpedia-api', 'r') as f:
    neuronpedia_headers = {"X-Api-Key": f.read().strip()}

EXPLANATION_CACHE_PATH = os.path.abspath('../data/explanation_cache.json')

# Cache calls to fetch explanations from Neuronpedia to make re-runs quick, and since
# some latents appear for multiple sets of harms.
# Note: these are auto-interpretation, so take with a grain of salt, but they have some value
# for quickly getting a sense of something.
try:
    if EXPLANATION_CACHE:
        print("Cache EXPLANATION_CACHE already exists, not overwriting it to avoid repeated API calls!")
except NameError:
    with open(EXPLANATION_CACHE_PATH) as f:
        EXPLANATION_CACHE = json.load(f)

def fetch_explanations(path):
    global EXPLANATION_CACHE
    if path in EXPLANATION_CACHE:
        return EXPLANATION_CACHE[path]
    else:
        response = requests.get(f"https://www.neuronpedia.org/api/feature/{path}", headers=neuronpedia_headers)
        explanations = response.json().get('explanations', [])
        explanation = explanations[0].get('description', "(unknown)") if explanations else "(unknown)"
        EXPLANATION_CACHE[path] = explanation
        # Every 10 new explanations, save them out to the cache file.
        if len(EXPLANATION_CACHE) % 10 == 0:
            with open(EXPLANATION_CACHE_PATH, 'wt') as f:
                json.dump(EXPLANATION_CACHE, f, indent=2)
        return explanation