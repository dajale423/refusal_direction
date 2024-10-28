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

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)

def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = None,
    steering_coefficient_ctx: Float[Tensor, "pos"] = None, # used to steer differently for different positions
    max_new_tokens: int = 50,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient=steering_coefficient,
        steering_coefficient_ctx = steering_coefficient_ctx
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output

def get_projection(direction, activation):
    direction_norm = t.linalg.vector_norm(direction)
    return einops.einsum(direction, activation.double(), "n_dim, batch ctx n_dim -> batch ctx")  / direction_norm

## steer along SAE latent by some coefficient
def get_projection_for_coefficient(model, sae, prompt, latent_idx, refusal_direction, refusal_layer, resid_pre_shape, steering_coefficient = None, steering_coefficient_ctx = None):

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
    )
    
    steered_activation = perturbed_final_resid_pre_store.clone()
    
    return get_projection(refusal_direction, steered_activation)

    