from nnsight import LanguageModel

import gc
import itertools
import math
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import torch as t
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from tabulate import tabulate
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from scripts.tensor_utils import get_projection
device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"

def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float = None,
    steering_coefficient_ctx: Float[Tensor, "pos"] = None, # used to steer differently for different positions
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    # print(activations.shape)
    if steering_coefficient is None:
        perturbation_vector = einops.einsum(steering_coefficient_ctx, sae.W_dec[latent_idx], "ctx, n_dim -> ctx n_dim")
        return activations + perturbation_vector
    else:
        return activations + steering_coefficient * sae.W_dec[latent_idx]


def steering_hook_manual(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    direction: Float[Tensor, "d_in"],
    steering_coefficient: float = None,
    steering_coefficient_ctx: Float[Tensor, "pos"] = None, # used to steer differently for different positions
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    ## normalize the vector
    norm = t.norm(direction, p=2)
    direction = direction/norm
    
    if steering_coefficient is None:
        if activations.shape[1] == 1:
            # print("shape of 1")
            return activations
        perturbation_vector = einops.einsum(steering_coefficient_ctx, direction, "ctx, n_dim -> ctx n_dim")
        return activations + perturbation_vector
    else:
        return activations + steering_coefficient * direction

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)

def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = None,
    steering_coefficient_ctx: Float[Tensor, "pos"] = None, # used to steer differently for different positions
    max_new_tokens: int = 50,
    prepend_bos: bool = True
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    model.reset_hooks()
    
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient=steering_coefficient,
        steering_coefficient_ctx = steering_coefficient_ctx
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, prepend_bos = prepend_bos, **GENERATE_KWARGS)

    return output

def generate_with_steering_manual(
    model: HookedSAETransformer,
    direction: Float[Tensor, "d_in"],
    prompt: str,
    steering_coefficient: float = None,
    steering_coefficient_ctx: Float[Tensor, "pos"] = None, # used to steer differently for different positions
    max_new_tokens: int = 50,
    hook_name: str = 'blocks.15.hook_resid_pre',
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    
    _steering_hook = partial(
        steering_hook_manual,
        direction=direction,
        steering_coefficient=steering_coefficient,
        steering_coefficient_ctx = steering_coefficient_ctx
    )

    with model.hooks(fwd_hooks=[(hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output

## steer along SAE latent by some coefficient
def get_projection_for_coefficient(model, sae, prompt, latent_idx, refusal_direction, resid_pre_shape, steering_coefficient = None, steering_coefficient_ctx = None, refusal_layer = 15, prepend_bos = True):

    hook_name = f'blocks.{refusal_layer}.hook_resid_pre'
    perturbed_final_resid_pre_store = t.zeros(resid_pre_shape, device=device)

    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient= steering_coefficient,
        steering_coefficient_ctx= steering_coefficient_ctx,
    )
    def get_activation_perturbed(
        activation, hook
    ):
        '''
        Get the activation
        '''
        perturbed_final_resid_pre_store[:, :] = activation[:, :].detach()

    intervention_logits = model.run_with_hooks(
        prompt,
        fwd_hooks=[(sae.cfg.hook_name, _steering_hook),
                   (hook_name, get_activation_perturbed)],
        stop_at_layer=refusal_layer + 1,
        prepend_bos = prepend_bos
    )
    
    steered_activation = perturbed_final_resid_pre_store.clone()
    
    return get_projection(refusal_direction, steered_activation)

def add_along_latent(model, sae, prompt, coefficient, latent_idx, refusal_direction, activation_shape, refusal_layer = 15):
        
    new_projection_last_token_add = get_projection_for_coefficient(model, sae, prompt, latent_idx, 
                                                                                refusal_direction, refusal_layer, 
                                                                                activation_shape, coefficient, None)
    
    return new_projection_last_token_add[:, -1].item()


def get_gradient(model, prompt, layer, refusal_direction, refusal_layer = 15, prepend_bos = False):
    model.reset_hooks()

    backward_cache = {}
    def backward_hook(gradient, hook):
        backward_cache[hook.name] = gradient.detach()

    model.add_hook(f'blocks.{layer}.hook_resid_post', backward_hook, dir="bwd")
    hook_name_refusal = f'blocks.{refusal_layer}.hook_resid_pre'
    
    def metric_hook(activations, hook):
        projection = get_projection(refusal_direction, activations)[0,-1]
        projection.backward()
    
    model.add_hook(hook_name_refusal, metric_hook, dir="fwd")
    
    _, full_cache = model.run_with_cache(
        prompt,
        stop_at_layer=refusal_layer + 1,
        prepend_bos = prepend_bos
    )

    return backward_cache[f'blocks.{layer}.hook_resid_post']

    