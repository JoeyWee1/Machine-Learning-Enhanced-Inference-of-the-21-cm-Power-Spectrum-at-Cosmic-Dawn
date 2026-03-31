import numpy as np
from utils.general import set_seed

def noisify(pmodel, theta4, n_fnoise): # pmodel is just the mean
    n_models = pmodel.shape[0] # shape (n_models ,54)

    # draw fnoise
    fnoise = 10**np.random.uniform(-3,0, size=(n_models, n_fnoise)) # size (n_models, n_fnoise)
    
    # reshape to targe (n_models, n_fnoise, 54)
    pmodel_r = pmodel[:, np.newaxis, :] # (n_models, 1, 54)
    fnoise_r = fnoise[:, :, np.newaxis] # (n_models, n_fnoise, 1)

    # variances
    var = 0.02 * (pmodel_r**2) * ((1-fnoise_r)**2) # (n_models, n_fnoise, 54)

    # draw the gaussian
    pnoisy =  np.random.normal(pmodel_r, np.sqrt(var)) #yields (n_models, n_fnoise, 54)

    
    theta_r = np.repeat(theta4, n_fnoise, axis=0)  # (n_models * n_fnoise, 4)
    fnoise_flat = fnoise.reshape(-1, 1)              # (n_models * n_fnoise, 1)
    pnoisy_flat = pnoisy.reshape(-1, 54)               # (n_models * n_fnoise, 54)

    theta_noisy = np.concatenate([theta_r, fnoise_flat], axis=1) # (n_models * n_fnoise, 5)

    return pnoisy_flat, theta_noisy

def noisify(
    pmodel_train: np.ndarray,
    pmodel_val: np.ndarray,
    pmodel_test: np.ndarray,
    theta4_train: np.ndarray,
    theta4_val: np.ndarray,
    theta4_test: np.ndarray,
    n_fnoise: int,
) -> dict:
    """
    Generate noisy mock observations from noiseless power spectra for all data splits.

    For each simulation in each split, draws n_fnoise independent noise realisations
    by sampling fnoise from a log-uniform distribution and adding Gaussian noise
    scaled by the power spectrum amplitude. Each noisy realisation becomes a
    separate sample with fnoise appended as a 5th parameter.

    Parameters
    ----------
    pmodel_train : ndarray of shape (n_train, 54)
        Noiseless training power spectra.
    pmodel_val : ndarray of shape (n_val, 54)
        Noiseless validation power spectra.
    pmodel_test : ndarray of shape (n_test, 54)
        Noiseless test power spectra.
    theta4_train : ndarray of shape (n_train, 4)
        Training parameters [L40_xray, fesc10, epsilon, h].
    theta4_val : ndarray of shape (n_val, 4)
        Validation parameters [L40_xray, fesc10, epsilon, h].
    theta4_test : ndarray of shape (n_test, 4)
        Test parameters [L40_xray, fesc10, epsilon, h].
    n_fnoise : int
        Number of independent noise realisations to draw per simulation.

    Returns
    -------
    dict
        - 'pnoisy_train' : ndarray of shape (n_train * n_fnoise, 54)
        - 'pnoisy_val'   : ndarray of shape (n_val * n_fnoise, 54)
        - 'pnoisy_test'  : ndarray of shape (n_test * n_fnoise, 54)
        - 'theta5_train'  : ndarray of shape (n_train * n_fnoise, 5)
        - 'theta5_val'    : ndarray of shape (n_val * n_fnoise, 5)
        - 'theta5_test'   : ndarray of shape (n_test * n_fnoise, 5)
        Parameter arrays have fnoise appended as the 5th column:
        [L40_xray, fesc10, epsilon, h, fnoise].

    Notes
    -----
    Noise variance follows: var = 0.02 * p² * (1 - fnoise)²
    fnoise is drawn log-uniformly from [1e-3, 1].
    All scalers and splits are processed independently to prevent data leakage.
    """

    def _noisify_split(pmodel, theta4):
        n_models = pmodel.shape[0] # shape (n_models ,54)

        # draw fnoise
        fnoise = 10**np.random.uniform(-3,0, size=(n_models, n_fnoise)) # size (n_models, n_fnoise)
        
        # reshape to targe (n_models, n_fnoise, 54)
        pmodel_r = pmodel[:, np.newaxis, :] # (n_models, 1, 54)
        fnoise_r = fnoise[:, :, np.newaxis] # (n_models, n_fnoise, 1)

        # variances
        var = 0.02 * (pmodel_r**2) * ((1-fnoise_r)**2) # (n_models, n_fnoise, 54) i hope ;-; 

        # draw the gaussian
        pnoisy =  np.random.normal(pmodel_r, np.sqrt(var)) #yields (n_models, n_fnoise, 54)

        
        theta_r = np.repeat(theta4, n_fnoise, axis=0)  # (n_models * n_fnoise, 4)
        fnoise_flat = fnoise.reshape(-1, 1)              # (n_models * n_fnoise, 1)
        pnoisy_flat = pnoisy.reshape(-1, 54)               # (n_models * n_fnoise, 54)

        theta_noisy = np.concatenate([theta_r, fnoise_flat], axis=1) # (n_models * n_fnoise, 5)

        return pnoisy_flat, theta_noisy

    set_seed()
    pnoisy_train, theta_train = _noisify_split(pmodel_train, theta4_train)
    pnoisy_val,   theta_val   = _noisify_split(pmodel_val,   theta4_val)
    pnoisy_test,  theta_test  = _noisify_split(pmodel_test,  theta4_test)

    return {
        "pnoisy_train": pnoisy_train, "theta5_train": theta_train,
        "pnoisy_val":   pnoisy_val,   "theta5_val":   theta_val,
        "pnoisy_test":  pnoisy_test,  "theta5_test":  theta_test,
    }