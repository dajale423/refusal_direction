import gc
import einops
import torch as t
from rich import print as rprint

import enrichment_utils

refusal_direction = enrichment_utils.load_tensor("../pipeline/runs/gemma-2-2b-it/direction.pt")
refusal_layer = 15
refusal_hook_name = f"blocks.{refusal_layer}.hook_resid_post"


def get_refusal_projection(direction, activation):
    direction_norm = t.linalg.vector_norm(direction)
    return einops.einsum(direction, activation.double(), "n_dim, batch ctx n_dim -> batch ctx")  / direction_norm


def get_projection(prompt, model, token_position=-1):
    _, cache = model.run_with_cache(prompt)
    projection = get_refusal_projection(refusal_direction, cache[refusal_hook_name])
    if token_position is not None:
        return projection[0,token_position].item()
    else:
        # 1: because we skip the projection onto the BOS (beginning-of-sentence) token
        # which always seems to have a large value
        return list(projection[0, 1:])

hook_name_refusal = f'blocks.{refusal_layer}.hook_resid_pre'

def metric_hook(activations, hook):
    projection = get_refusal_projection(refusal_direction, activations)[0, -1]
    projection.backward()


def get_refusal_projection_gradients(prompt, model, layer):
    hook_name_latents = f'blocks.{layer}.hook_resid_post'
    model.reset_hooks()

    backward_cache = {}
    def backward_hook(gradient, hook):
        backward_cache[hook.name] = gradient.detach()

    model.add_hook(hook_name_latents, backward_hook, dir="bwd")
    model.add_hook(hook_name_refusal, metric_hook, dir="fwd")
    model.run_with_cache(prompt, stop_at_layer=refusal_layer + 1)

    # Remove batch dimension, there's just the one prompt anyway.
    return backward_cache[hook_name_latents][0, :, :] # tokens x latents

def explain_gradient_latents(prompt, gradients, model, sae):
    num_tokens = gradients.shape[0]
    str_toks = model.to_str_tokens(prompt)
    for position in range(num_tokens):
        rprint(f"\n***** position {position}/{num_tokens} " + "".join([f"[b u green]{str_tok}[/]" if i == position else str_tok for i, str_tok in enumerate(str_toks)]).strip())
        topk = (sae.W_dec @ gradients[position, :]).topk(k=5)
        for latent, v in zip(topk.indices, topk.values):
            if not sae.startswith('sae_gemma-2-2b'):
                raise ValueError(f"SAE {sae.get_name()} not supported for Neuronpedia links")
            path = f"gemma-2-2b/{sae.cfg.hook_layer}-gemmascope-res-16k/{latent}"        
            explanation = enrichment_utils.fetch_explanations(path)
            print(f"{latent.item():5} {v.item():.4f} {explanation}")