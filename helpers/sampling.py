import emcee as mc
import numpy as np
from scipy.stats import loguniform
import torch


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


def ln_uniform_prior(h, eps, L40, fesc10, fnoise, priors):
    """
    Compute the joint log-prior probability using pre-built prior objects.

    Parameters
    ----------
    h, eps, L40, fesc10, fnoise : float
        Parameter values.
    priors : dict
        Pre-built loguniform objects keyed by 'h', 'epsilon', 'L40_xray',
        'fesc10', 'fnoise'. Build once with _build_priors().

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


def ln_likelihood(fnoise, p_model, p_obs):
    """
    Compute log-likelihood given a pre-evaluated model spectrum.

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


def ln_post(theta, model, p_obs, processed, priors):
    """
    Compute the log-posterior for a single walker position.

    Parameters
    ----------
    theta : array-like of shape (5,)
        Parameter vector [eps, L40, fesc10, h, fnoise].
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
    float
        Log-posterior. -inf if outside prior support.
    """
    eps, L40, fesc10, h, fnoise = theta

    lnPi = ln_uniform_prior(h=h, eps=eps, L40=L40, fesc10=fesc10,
                            fnoise=fnoise, priors=priors)
    if not np.isfinite(lnPi):
        return -np.inf

    with torch.no_grad():
        params_tensor = torch.tensor([L40, fesc10, eps, h], dtype=torch.float32)
        pred_scaled = model(params_tensor).cpu().numpy().reshape(1, -1)

    pred_raw = processed["weight_scaler"].inverse_transform(pred_scaled)
    p_model = processed["pca"].inverse_transform(pred_raw)[0]
    if processed.get("log_power", False):
        p_model = np.exp(p_model)

    return ln_likelihood(fnoise, p_model, p_obs) + lnPi


def ln_post_vec(thetas, model, p_obs, processed, priors):
    """
    Vectorized log-posterior over a batch of walker positions.

    Used with EnsembleSampler(vectorize=True). emcee passes all
    (n_walkers // 2) positions at once, so the NN forward pass is a
    single batched matrix multiply instead of n_walkers // 2 separate
    calls.

    Parameters
    ----------
    thetas : np.ndarray of shape (n, 5)
        Walker positions, columns [eps, L40, fesc10, h, fnoise].
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
        Log-posterior values. -inf for positions outside prior support.
    """
    n = len(thetas)
    results = np.full(n, -np.inf)
    log_priors = np.zeros(n)
    valid = np.ones(n, dtype=bool)

    # Prior check for all walkers — no NN calls yet
    for i, (eps, L40, fesc10, h, fnoise) in enumerate(thetas):
        lp = ln_uniform_prior(h, eps, L40, fesc10, fnoise, priors)
        if not np.isfinite(lp):
            valid[i] = False
        else:
            log_priors[i] = lp

    if not np.any(valid):
        return results

    # Single batched forward pass for all valid walkers
    valid_thetas = thetas[valid]
    # Model input order: [L40, fesc10, eps, h] — columns [1, 2, 0, 3] of theta
    params_batch = valid_thetas[:, [1, 2, 0, 3]]

    with torch.no_grad():
        params_tensor = torch.tensor(params_batch, dtype=torch.float32)
        pred_scaled = model(params_tensor).cpu().numpy()

    pred_raw = processed["weight_scaler"].inverse_transform(pred_scaled)
    pred_spectra = processed["pca"].inverse_transform(pred_raw)
    if processed.get("log_power", False):
        pred_spectra = np.exp(pred_spectra)

    # Likelihood for each valid walker
    valid_indices = np.where(valid)[0]
    for j, i in enumerate(valid_indices):
        results[i] = ln_likelihood(thetas[i, 4], pred_spectra[j], p_obs) + log_priors[i]

    return results

def ln_post_vec_log(phis, model, p_obs, processed, priors): #wrap function that calculates vectors of lnPosteriors to accept log-space positions
    thetas = np.exp(phis)                      
    log_probs = ln_post_vec(thetas, model, p_obs, processed, priors)
    jacobian = phis.sum(axis=1)              
    finite = np.isfinite(log_probs)
    log_probs[finite] += jacobian[finite]
    return log_probs  


def generate_chain(
        n_walkers: int = 32,
        steps: int = 10000,
        discard: int = 1000,
        tf: int = 2,
        unscaled_feature_domains: dict = None,
        model=None,
        p_obs: np.ndarray = None,
        processed: dict = None,
) -> tuple:
    """
    Runs an emcee ensemble MCMC sampler to draw posterior samples for the
    21-cm power spectrum parameters.

    Parameters
    ----------
    n_walkers : int, optional
        Number of ensemble walkers. Must be even and >= 2 * ndim. Default 32.
    steps : int, optional
        Number of MCMC steps per walker. Default 10000.
    discard : int, optional
        Number of initial steps to discard as burn-in. Default 1000.
    tf : int, optional
        Thinning factor multiplier (thin = tf * tau). Default 2.
    unscaled_feature_domains : dict
        Dictionary mapping parameter names to [min, max] physical bounds.
        Expected keys: 'epsilon', 'L40_xray', 'fesc10', 'h'.
    model : nn.Module
        Trained neural network emulator for the 21-cm power spectrum.
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum at each k mode.
    processed : dict
        Preprocessing artefacts from `preprocess()`.

    Returns
    -------
    dict with keys:
        "sampler" : emcee.EnsembleSampler
            The sampler object after running, containing the full chain.
        "mean_frac" : float
            Mean acceptance fraction across all walkers. Healthy range ~0.2-0.5.
        "taus" : np.ndarray of shape (5,)
            Autocorrelation time estimate for each parameter.
        "mean_tau" : float
            Mean autocorrelation time across all parameters.
        "tau" : int
            Maximum autocorrelation time, used for thinning.
        "samples" : np.ndarray of shape (n_samples, 5)
            Flattened posterior samples after burn-in and thinning,
            columns [eps, L40, fesc10, h, fnoise].
    """
    # Build prior objects once — shared across all n_walkers * steps evaluations
    priors = _build_priors(unscaled_feature_domains)

    rng = np.random.default_rng(seed=1701)
    initial_pos = np.log(np.column_stack([
        priors[key].rvs(size=n_walkers, random_state=rng)
        for key in ('epsilon', 'L40_xray', 'fesc10', 'h', 'fnoise')
    ]))

    # Set eval mode once before sampling — not per-call inside ln_post_vec
    model.eval()

    # vectorize=True: emcee passes all n_walkers//2 positions at once,
    # enabling the single batched forward pass in ln_post_vec
    sampler = mc.EnsembleSampler(
        n_walkers,
        ndim=5,
        log_prob_fn=ln_post_vec_log,
        args=[model, p_obs, processed, priors],
        vectorize=True,
    )

    sampler.run_mcmc(initial_pos, steps, progress=True)

    mean_frac = sampler.acceptance_fraction.mean()
    taus = sampler.get_autocorr_time()
    mean_tau = np.mean(taus)
    tau = int(max(taus))

    print(f"Mean acceptance fraction: {mean_frac:.3f}")
    print(f"Mean autocorrelation time: {mean_tau:.2f} steps")

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
