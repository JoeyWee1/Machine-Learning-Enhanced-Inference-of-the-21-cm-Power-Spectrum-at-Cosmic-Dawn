import corner
import dynesty
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.mcmc.ensemble import ln_likelihood_emulator

def make_prior_transform(domains: dict):
    """
    Return a function mapping u ~ Uniform[0,1]^5 to physical parameters
    under log-uniform priors (matching helpers/sampling.py).

    Parameter order: [L40, fesc10, eps, h, fnoise]

    Prior support is widened to [0.5*lo, 1.5*hi] per domain key,
    matching _build_priors() in sampling.py.
    """
    bounds = [
        (0.5 * domains['L40_xray'][0], 1.5 * domains['L40_xray'][1]),
        (0.5 * domains['fesc10'][0],   1.5 * domains['fesc10'][1]),
        (0.5 * domains['epsilon'][0],  1.5 * domains['epsilon'][1]),
        (0.5 * domains['h'][0],        1.5 * domains['h'][1]),
        (1e-3, 1e1),   # fnoise
    ]
    log_lo = np.array([np.log(lo) for lo, _ in bounds])
    log_hi = np.array([np.log(hi) for _, hi in bounds])

    def prior_transform(u):
        return np.exp(log_lo + u * (log_hi - log_lo))

    return prior_transform



def make_emulator_ln_likelihood(model, p_obs: np.ndarray, processed: dict):
    """
    Return a log-likelihood function compatible with dynesty.

    dynesty evaluates one point at a time; the NN forward pass is a
    single vector call per evaluation.
    """
    def log_likelihood(theta):
        L40, fesc10, eps, h, fnoise = theta

        with torch.no_grad():
            # Model input order: [L40, fesc10, eps, h]
            params_raw = np.array([[L40, fesc10, eps, h]])
            params_standardised = processed["params_scaler"].transform(params_raw)
            params_tensor = torch.tensor(params_standardised, dtype=torch.float32)
            pred_scaled = model(params_tensor).cpu().numpy().reshape(1, -1)

        pred_raw = processed["weight_scaler"].inverse_transform(pred_scaled)
        p_model  = processed["pca"].inverse_transform(pred_raw)[0]
        if processed.get("log_power", False):
            p_model = np.exp(p_model)

        return ln_likelihood_emulator(fnoise, p_model, p_obs)

    return log_likelihood

def build_sampler(
    model,
    processed: dict,
    p_obs: np.ndarray,
    domains: dict,
    nlive: int = 1500,
    print_progress: bool = True,
):
    """
    Run dynesty DynamicNestedSampler and return the results object.

    Parameters
    ----------
    model : nn.Module
        Trained emulator in eval() mode.
    processed : dict
        Preprocessing artefacts (weight_scaler, pca, log_power).
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum.
    domains : dict
        Feature domains with keys 'epsilon', 'L40_xray', 'fesc10', 'h'.
    nlive : int
        Number of live points. Higher = more accurate but slower.
        1500 is a reasonable default; 3000 for publication quality.
    print_progress : bool
        Whether to print dynesty progress.

    Returns
    -------
    dynesty.results.Results
        Contains samples (physical space), weights, and log-evidence.
    """
    model.eval()

    prior_transform = make_prior_transform(domains)
    log_likelihood  = make_emulator_ln_likelihood(model, p_obs, processed)

    sampler = dynesty.DynamicNestedSampler(
        log_likelihood,
        prior_transform,
        ndim=5,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
    )
    sampler.run_nested(print_progress=print_progress)
    return sampler.results





