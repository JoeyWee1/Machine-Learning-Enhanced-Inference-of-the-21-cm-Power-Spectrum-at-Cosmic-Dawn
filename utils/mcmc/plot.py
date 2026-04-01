import numpy as np
import corner
import matplotlib.pyplot as plt
from dynesty.utils import resample_equal

def plot_emcee_corner(unthinned_chain: np.ndarray, diagnostic: dict, df: int = 10) -> None:
    """
    Plot a corner plot of the posterior samples from an emcee chain.

    Discards burn-in samples based on the autocorrelation time before
    flattening the chain and plotting marginal and joint posterior distributions
    for the 5 model parameters.

    Parameters
    ----------
    unthinned_chain : ndarray of shape (n_steps, n_walkers, n_params)
        Raw emcee chain as returned by sampler.get_chain().
    diagnostic : dict
        Sampling diagnostics. Required keys:
        - 'tau' : float, estimated autocorrelation time used to determine burn-in.
    df : int, optional
        Burn-in is set to df * tau steps. Default 10.

    Returns
    -------
    None
        Displays the corner plot inline.

    Notes
    -----
    Only the first 5 parameters are plotted: L40_xray, fesc10, epsilon, h, fnoise.
    Quantiles shown are 16th, 50th, and 84th percentiles (i.e. median ± 1σ).
    """
    tau = diagnostic["tau"]
    discard = int(df * tau)   # Safe number to discard
    flat = unthinned_chain[discard:, :, :5].reshape(-1, 5)
    
    labels = [
        r"$L_{40}^{\text{X-ray}}$",
        r"$f_{\text{esc}}^{10}$",
        r"$\epsilon$",
        r"$h$",
        r"$f_{\text{noise}}$",
    ]

    plt.figure(figsize=(8, 8), dpi=150)
    corner.corner(
        flat,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )
    plt.show()

def plot_nested_corner(results) -> np.ndarray:
    """
    Plot a corner plot from dynesty nested sampling results.

    Reweights and resamples the nested sampling chain into equal-weight
    posterior samples before plotting marginal and joint distributions
    for the 5 model parameters.

    Parameters
    ----------
    results : dynesty.results.Results
        Output of sampler.results after running dynesty nested sampling.
        Required attributes:
        - 'logwt'    : ndarray of shape (n_samples,), log importance weights.
        - 'logz'     : ndarray of shape (n_samples,), running log evidence estimate.
        - 'samples'  : ndarray of shape (n_samples, n_params), raw samples.

    Returns
    -------
    ndarray of shape (n_resampled, 5)
        Equal-weight posterior samples after resampling.

    Notes
    -----
    Quantiles shown are 16th, 50th, and 84th percentiles (median ± 1σ).
    """
    labels = [
        r"$L_{40}^{\text{X-ray}}$",
        r"$f_{\text{esc}}^{10}$",
        r"$\epsilon$",
        r"$h$",
        r"$f_{\text{noise}}$",
    ]
    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)   # (n_samples, 5)

    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )

    plt.show()
    return samples