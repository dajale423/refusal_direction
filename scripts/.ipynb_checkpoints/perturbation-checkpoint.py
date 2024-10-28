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


def get_sae_activation(
    model, 
    sae,
    prompt,
    latent_idx,
    token_position = -1):

    # Get activations on final token
    _, cache = model.run_with_cache_with_saes(
        prompt,
        saes=[sae],
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, token_position, :]

    return sae_acts_post[latent_idx].item()

def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)

def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = 1.0,
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
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output

def get_projection(direction, activation):
    direction_norm = t.linalg.vector_norm(direction)
    return einops.einsum(direction, activation.double(), "n_dim, batch ctx n_dim -> batch ctx")  / direction_norm
    