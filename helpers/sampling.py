import emcee as mc
import numpy as np
from scipy.stats import loguniform
from helpers.evaluate_model import predict_spectrum


def ln_uniform_prior(h, eps, L40, fesc10, fnoise, unscaled_feature_domains):
    """
    Compute the joint log-prior probability under independent log-uniform priors.

    Each parameter is assigned a log-uniform (reciprocal) prior with support
    [0.5 * domain_min, 1.5 * domain_max], giving equal prior weight to each
    order of magnitude within a broadened version of the known parameter domain.

    The joint log-prior is the sum of the individual log-priors, which follows
    from the assumption of independence between parameters (product of marginals).

    Parameters
    ----------
    h : float
        Hydrogen fraction (or analogous parameter).
    eps : float
        Efficiency parameter epsilon.
    L40 : float
        X-ray luminosity at 40 eV (L40_xray).
    fesc10 : float
        Escape fraction at 10 eV.
    fnoise : float
        Noise fraction parameter. Lives between 1e-3 and 1e1. NUISANCE parameter.
    unscaled_feature_domains : dict
        Dictionary mapping parameter names to [min, max] bounds in unscaled
        (physical) units. Expected keys: 'h', 'epsilon', 'L40_xray', 'fesc10'.

    Returns
    -------
    float
        Sum of log-prior probabilities: log p(h) + log p(eps) + log p(L40) + log p(fesc10) + log p(fnoise).
        Returns -inf if any parameter falls outside its prior support.
    """
    def make_prior(domain, key):
        lo, hi = unscaled_feature_domains[key]
        return loguniform(a=0.5 * lo, b=1.5 * hi)

    priors = {
        'h':        (make_prior(unscaled_feature_domains, 'h'),        h),
        'epsilon':  (make_prior(unscaled_feature_domains, 'epsilon'),  eps),
        'L40_xray': (make_prior(unscaled_feature_domains, 'L40_xray'), L40),
        'fesc10':   (make_prior(unscaled_feature_domains, 'fesc10'),   fesc10),
    }

    return sum(prior.logpdf(val) for prior, val in priors.values()) + loguniform(a=1e-3, b=1e1).logpdf(fnoise)


def ln_likelihood(eps, L40, fesc10, h, fnoise, model, p_obs, processed):
    """
    Computes the log-likelihood of the observed 21-cm power spectrum given
    model parameters, assuming a Gaussian likelihood in the limit of many
    modes per k bin (central limit theorem).

    Parameters
    ----------
    eps : float
        Star formation efficiency in galaxies of 10 solar masses (epsilon_*).
    L40 : float
        X-ray luminosity of early galaxies (L^X-ray_40).
    fesc10 : float
        Escape fraction of ionising photons in galaxies of 10 solar masses.
    h : float
        Hubble expansion rate parameter.
    fnoise : float
        Noise fraction parameter in [1e-3, 1], such that
        P_noise(k) = fnoise * P_model(k).
    model : nn.Module
        Trained neural network emulator for the 21-cm power spectrum.
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum at each k mode.
    processed : dict
        Preprocessing artefacts from `preprocess()`, passed directly to
        `predict_spectrum`.

    Returns
    -------
    float
        Log-likelihood ln L evaluated at the given parameters, computed as:

            ln L = -0.5 * sum((P_obs - P_model)^2 / sigma^2)
                   -0.5 * sum(log(2 * pi * sigma^2))

        where sigma^2(k) = 2 * (P_model(k) * (1 - fnoise))^2 / N_k
        and N_k = 100 modes per k bin is assumed throughout.
    """
    p_model = predict_spectrum(model, params = [L40, fesc10, eps, h], processed=processed)
    var =  2 * (p_model ** 2) * ((1 - fnoise)**2) /100
    delta = p_obs - p_model
    first_term = -0.5 * np.sum((delta**2)/var)
    second_term = -0.5 * np.sum(np.log(2 * np.pi * var))
    lnL = first_term + second_term
    return lnL

def ln_post(theta, model, p_obs, processed, unscaled_feature_domains):
    """
    Computes the log-posterior probability for the 21-cm power spectrum model.

    Combines the log-prior and log-likelihood following Bayes' theorem:
        ln p(theta | D) = ln p(D | theta) + ln p(theta) + const.

    Parameters
    ----------
    theta : array-like of shape (5,)
        Parameter vector in the order [eps, L40, fesc10, h, fnoise]:
        - eps : float
            Star formation efficiency in galaxies of 10 solar masses.
        - L40 : float
            X-ray luminosity of early galaxies.
        - fesc10 : float
            Escape fraction of ionising photons in galaxies of 10 solar masses.
        - h : float
            Hubble expansion rate parameter.
        - fnoise : float
            Noise nuisance parameter, marginalised over during sampling.
    model : nn.Module
        Trained neural network emulator for the 21-cm power spectrum.
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum at each k mode.
    processed : dict
        Preprocessing artefacts from `preprocess()`, passed to
        `predict_spectrum`.
    unscaled_feature_domains : dict
        Dictionary mapping parameter names to [min, max] physical bounds,
        used to construct the log-uniform priors.

    Returns
    -------
    float
        Log-posterior ln p(theta | D). Returns -inf if any parameter
        falls outside its prior support (from ln_uniform_prior).
    """

    if not np.isfinite(lnPi):  # reject early if outside prior support
        return -np.inf
    eps, L40, fesc10, h, fnoise = theta
    lnPi = ln_uniform_prior(h=h, eps=eps,L40=L40,fesc10=fesc10,fnoise=fnoise, unscaled_feature_domains=unscaled_feature_domains)
    lnL = ln_likelihood(eps, L40, fesc10, h, fnoise, model, p_obs, processed)
    return lnL + lnPi

def generate_chain

