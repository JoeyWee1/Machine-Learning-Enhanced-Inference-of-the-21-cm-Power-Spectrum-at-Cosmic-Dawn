import emcee as mc
import numpy as np
from scipy.stats import loguniform
import torch

# Priors

def _build_priors(unscaled_feature_domains: dict) -> dict:
    """
    Construct loguniform prior objects once from feature domains.

    Call this once before sampling and pass the result to ln_uniform_prior
    and ln_post_vec so that prior objects are not re-created on every
    posterior evaluation.

    Parameters
    ----------
    unscaled_feature_domains : dict
        Dictionary mapping 'epsilon', 'L40_xray', 'fesc10', 'h' to
        [min, max] physical bounds.

    Returns
    -------
    dict
        loguniform objects keyed by 'epsilon', 'L40_xray', 'fesc10',
        'h', and 'fnoise'.
    """
    def make(key):
        lo, hi = unscaled_feature_domains[key]
        return loguniform(a=0.5 * lo, b=1.5 * hi)

    return {
        'epsilon':  make('epsilon'),
        'L40_xray': make('L40_xray'),
        'fesc10':   make('fesc10'),
        'h':        make('h'),
        'fnoise':   loguniform(a=1e-3, b=1e1),
    }

def ln_uniform_prior(L40, fesc10, eps, h, fnoise, priors):
    """
    Compute the joint log-prior probability using pre-built prior objects.

    Parameters
    ----------
    L40, fesc10, eps, h, fnoise : float
        Parameter values.
    priors : dict
        Pre-built loguniform objects keyed by 'L40_xray', 'fesc10',
        'epsilon', 'h', 'fnoise'. Build once with _build_priors().

    Returns
    -------
    float
        Sum of log-prior probabilities. -inf if any value is outside
        its prior support.
    """
    return (
        priors['h'].logpdf(h)
        + priors['epsilon'].logpdf(eps)
        + priors['L40_xray'].logpdf(L40)
        + priors['fesc10'].logpdf(fesc10)
        + priors['fnoise'].logpdf(fnoise)
    )

# Emulator likelihood and log-post

def ln_likelihood_emulator(fnoise, p_model, p_obs):
    """
    Compute log-likelihood for an emulated p_model given a pre-evaluated model spectrum.

    Assumes a Gaussian likelihood in the limit of many modes per k bin.

    Parameters
    ----------
    fnoise : float
        Noise fraction parameter such that P_noise(k) = fnoise * P_model(k).
    p_model : np.ndarray of shape (54,)
        Model power spectrum, already evaluated by the emulator.
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum.

    Returns
    -------
    float
        Log-likelihood value.
    """
    var = 2 * (p_model ** 2) * ((1 - fnoise) ** 2) / 100
    delta = p_obs - p_model
    return -0.5 * np.sum(delta ** 2 / var) - 0.5 * np.sum(np.log(2 * np.pi * var))

def ln_post_emulator(thetas, model, p_obs, processed, priors):
    """
    Vectorized log-posterior over a batch of walker positions.

    Used with EnsembleSampler(vectorize=True). emcee passes all
    (n_walkers // 2) positions at once, so the NN forward pass is a
    single batched matrix multiply instead of n_walkers // 2 separate
    calls.

    Parameters
    ----------
    thetas : np.ndarray of shape (n, 5)
        Walker positions, columns [L40, fesc10, eps, h, fnoise].
    model : nn.Module
        Trained emulator, in eval() mode.
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum.
    processed : dict
        Preprocessing artefacts (params_scaler, weight_scaler, pca, log_power).
    priors : dict
        Pre-built loguniform objects from _build_priors().

    Returns
    -------
    np.ndarray of shape (n,)
        Log-posterior values. -inf for positions outside prior support.
    """
    n = len(thetas)
    results = np.full(n, -np.inf)
    log_priors = np.zeros(n)
    valid = np.ones(n, dtype=bool)

    # Prior check for all walkers — no NN calls yet
    for i, (L40, fesc10, eps, h, fnoise) in enumerate(thetas):
        lp = ln_uniform_prior(L40, fesc10, eps, h, fnoise, priors)
        if not np.isfinite(lp):
            valid[i] = False
        else:
            log_priors[i] = lp

    if not np.any(valid):
        return results

    # Single batched forward pass for all valid walkers
    valid_thetas = thetas[valid]
    # Model input order: [L40, fesc10, eps, h] — columns [0, 1, 2, 3] of theta
    params_batch = valid_thetas[:, :4]
    params_batch_scaled = processed["params_scaler"].transform(params_batch)

    with torch.no_grad():
        params_tensor = torch.tensor(params_batch_scaled, dtype=torch.float32)
        pred_scaled = model(params_tensor).cpu().numpy()

    pred_raw = processed["weight_scaler"].inverse_transform(pred_scaled)
    pred_spectra = processed["pca"].inverse_transform(pred_raw)
    if processed.get("log_power", False):
        pred_spectra = np.exp(pred_spectra)

    # Likelihood for each valid walker
    valid_indices = np.where(valid)[0]
    for j, i in enumerate(valid_indices):
        results[i] = ln_likelihood_emulator(thetas[i, 4], pred_spectra[j], p_obs) + log_priors[i]

    return results

def ln_post_log_emulator(phis, model, p_obs, processed, priors): #wrap function that calculates vectors of lnPosteriors to accept log-space positions
    """
    Vectorized log-posterior over a batch of walker positions in log-parameter space.

    Wraps `ln_post_emulator` to accept positions in log space, applying the
    log-Jacobian correction for the change of variables φ = log(θ). This allows
    emcee to sample in log space, which is natural for parameters with loguniform
    priors and improves proposal efficiency when parameters span several orders
    of magnitude.

    The Jacobian of the transformation θ = exp(φ) contributes a log-determinant
    of sum_i(φ_i) to the log-posterior::

        ln p(φ) = ln p(θ) + sum_i φ_i

    Parameters
    ----------
    phis : np.ndarray of shape (n, 5)
        Walker positions in log-parameter space, columns
        [log(L40), log(fesc10), log(eps), log(h), log(fnoise)].
    model : nn.Module
        Trained emulator, in eval() mode.
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum.
    processed : dict
        Preprocessing artefacts (weight_scaler, pca, log_power).
    priors : dict
        Pre-built loguniform objects from _build_priors().

    Returns
    -------
    np.ndarray of shape (n,)
        Log-posterior values in log-parameter space, with Jacobian applied.
        -inf for positions whose exponentiated values fall outside prior support.
    """
    thetas = np.exp(phis)                      
    log_probs = ln_post_emulator(thetas, model, p_obs, processed, priors)
    jacobian = phis.sum(axis=1)              
    finite = np.isfinite(log_probs)
    log_probs[finite] += jacobian[finite]
    return log_probs  

# NRE likelihood and log-post
def ln_post_nre(thetas, model, p_obs, nre_data, priors): # vectorised version
    n = len(thetas) # thetas is shape (n, 5)
    results = np.full(n, -np.inf)
    log_priors = np.zeros(n)
    valid = np.ones(n, dtype=bool)

    for i, (L40, fesc10, eps, h, fnoise) in enumerate(thetas):
        lnPi = ln_uniform_prior(L40=L40, fesc10=fesc10, eps=eps,
                                h=h, fnoise=fnoise, priors=priors)
        if not np.isfinite(lnPi):
            valid[i] = False
        else:
            log_priors[i] = lnPi

    if not np.any(valid):
        return results

    valid_thetas = thetas[valid]                                          # (m, 5)
    log_p_obs_tiled = np.tile(np.log(p_obs), (len(valid_thetas), 1))    # (m, 54)
    x_obs_raw = np.concatenate([log_p_obs_tiled, valid_thetas], axis=1) # (m, 59)
    x_obs = torch.tensor(nre_data["scaler"].transform(x_obs_raw), dtype=torch.float32)

    with torch.no_grad():
        lnr = model(x_obs).squeeze(1).cpu().numpy()                      # (m,)

    results[np.where(valid)[0]] = lnr + log_priors[valid]
    return results

def ln_post_log_nre(phis, model, p_obs, nre_data, priors):
    thetas = np.exp(phis)
    log_probs = ln_post_nre(thetas, model, p_obs, nre_data, priors)
    jacobian = phis.sum(axis=1)
    finite = np.isfinite(log_probs)
    log_probs[finite] += jacobian[finite]
    return log_probs

# Generate chain

def generate_chain(
        ln_post_log: callable,
        n_walkers: int = 32,
        steps: int = 10000,
        discard: int = 0,
        tf: int = 2,
        unscaled_feature_domains: dict = None,
        model=None,
        p_obs: np.ndarray = None,
        processed: dict = None,
) -> dict:

    # Build prior objects once — shared across all n_walkers * steps evaluations
    priors = _build_priors(unscaled_feature_domains)

    rng = np.random.default_rng(seed=1701)
    initial_pos = np.log(np.column_stack([
        priors[key].rvs(size=n_walkers, random_state=rng)
        for key in ('L40_xray', 'fesc10', 'epsilon', 'h', 'fnoise')
    ]))

    # Set eval mode once before sampling — not per-call inside ln_post_log
    model.eval()

    # vectorize=True: emcee passes all n_walkers//2 positions at once,
    # enabling the single batched forward pass in ln_post_log
    sampler = mc.EnsembleSampler(
        n_walkers,
        ndim=5,
        log_prob_fn=ln_post_log,
        args=[model, p_obs, processed, priors],
        vectorize=True,
    )

    sampler.run_mcmc(initial_pos, steps, progress=True)

    mean_frac = sampler.acceptance_fraction.mean()
    try:
        taus = sampler.get_autocorr_time()
        mean_tau = float(np.mean(taus))
        tau = int(max(taus))
        print(f"Mean autocorrelation time: {mean_tau:.2f} steps")
    except Exception:
        print("Warning: chain too short to estimate autocorrelation time. Using tau=1 (no thinning).")
        taus = None
        mean_tau = None
        tau = 1

    print(f"Mean acceptance fraction: {mean_frac:.3f}")

    thinned_samples = sampler.get_chain(discard=discard, thin=tf * tau, flat=False)
    unthinned_samples = sampler.get_chain(discard=discard, flat=False)

    return {
        "sampler": sampler,
        "mean_frac": mean_frac,
        "taus": taus,
        "mean_tau": mean_tau,
        "tau": tau,
        "thinned_samples": thinned_samples,
        "unthinned_samples": unthinned_samples,
    }