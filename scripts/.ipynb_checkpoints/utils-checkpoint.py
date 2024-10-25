import torch as t
import einops
import numpy as np
from scipy.stats import chi2_contingency, chi2

def load_tensor(filename):
    if device == "mps":
        tensor = t.load(filename, map_location="cpu")
        tensor.to(device, dtype=t.float32)
    else:
        tensor = t.load(filename)
    return tensor

from scipy.stats import chi2_contingency, chi2

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