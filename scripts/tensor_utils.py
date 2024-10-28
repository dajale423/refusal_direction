import torch as t
import einops

def load_tensor(filename):
    device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
    if device == "mps":
        tensor = t.load(filename, map_location="cpu")
        tensor.to(device, dtype=t.float32)
    else:
        tensor = t.load(filename)
    return tensor

def get_second_min(x):
    min_value = t.min(x)
    mask = x != min_value
    second_min_value = t.min(x[mask])

    return second_min_value

def get_projection(direction, activation):
    direction_norm = t.linalg.vector_norm(direction)
    return einops.einsum(direction, activation.double(), "n_dim, batch ctx n_dim -> batch ctx")  / direction_norm
