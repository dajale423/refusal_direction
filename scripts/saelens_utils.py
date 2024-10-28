from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
import einops
import torch as t
import sys

sys.path.append('../')
from scripts.tensor_utils import get_projection

def get_sae_activation(
    model, 
    sae,
    prompt,
    latent_idx = None,
    token_position = -1):

    # Get activations on final token
    _, cache = model.run_with_cache_with_saes(
        prompt,
        saes=[sae],
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, token_position, :]

    if latent_idx is None:
        return sae_acts_post
    return sae_acts_post[latent_idx].item()
    

def get_refusal_layer_resid_activations(model, 
                                        sae, 
                                        prompt, 
                                        latent_idx,
                                        refusal_direction,
                                        refusal_layer = 15):
    hook_name_refusal = f'blocks.{refusal_layer}.hook_resid_pre'
    
    activation = get_sae_activation(model, sae, prompt, None, None)
    activation_per_pos = activation.squeeze()[:, latent_idx]
    max_activation = activation_per_pos.max().item()
    last_token_activation = activation_per_pos[-1].item()
    
    sae.use_error_term = True
    _, original_cache = model.run_with_cache_with_saes(
        prompt,
        saes=[sae],
        stop_at_layer=refusal_layer + 1,
    )
    
    original_activation = original_cache[hook_name_refusal]
    activation_shape = original_activation.shape
    original_projection = get_projection(refusal_direction, original_activation)
    original_projection_last_token = original_projection[:, -1].item()

    return max_activation, last_token_activation, original_projection_last_token, activation_shape
    