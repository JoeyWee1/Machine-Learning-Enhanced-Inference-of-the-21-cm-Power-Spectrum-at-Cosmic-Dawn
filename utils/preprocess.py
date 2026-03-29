import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(data: dict, n_comp: int, log_power: bool = False) -> dict:
    """
    Preprocess raw simulation data for emulator training.

    Standardizes input parameters, compresses power spectra via PCA, and
    standardizes the resulting PCA coefficients. All scalers and PCA are
    fitted on training data only to prevent data leakage.

    Parameters
    ----------
    data : dict
        Raw simulation data with keys:
        - 'raw_params_train/val/test' : ndarray of shape (N, n_params)
        - 'power_train/val/test'      : ndarray of shape (N, n_k)
        - 'k_train/val/test'          : ndarray of shape (n_k,)
    n_comp : int
        Number of PCA components to retain.
    log_power : bool, optional
        If True, applies log to power spectra before PCA. All power values
        must be strictly positive. Default False.

    Returns
    -------
    dict
        Scalers and PCA:
        - 'params_scaler'                     : StandardScaler fitted on training parameters.
        - 'weight_scaler'                     : StandardScaler fitted on training PCA coefficients.
        - 'pca'                               : PCA object fitted on training power spectra.
        - 'evecs'                             : ndarray of shape (n_k, n_comp), PCA eigenvectors.
        - 'explained_variance_ratio'          : ndarray of shape (n_comp,).
        - 'log_power'                         : bool, whether log was applied to power spectra.

        Power spectra (not log transformed):
        - 'power_train/val/test'              : ndarray of shape (N, n_k).
        - 'k_train/val/test'                  : ndarray of shape (n_k,).

        Parameters:
        - 'params_train/val/test_raw'         : ndarray of shape (N, n_params), unscaled.
        - 'params_train/val/test_scaled'      : ndarray of shape (N, n_params), standardized.

        PCA coefficients:
        - 'pca_weights_train/val/test_raw'    : ndarray of shape (N, n_comp), unscaled.
        - 'pca_weights_train/val/test_scaled' : Tensor of shape (N, n_comp), standardized.

    Raises
    ------
    ValueError
        If log_power=True and any power spectrum value is non-positive.

    Notes
    -----
    The preprocessing pipeline is:
        1. (Optional) Apply log to all power spectra.
        2. Fit StandardScaler on training parameters; apply to all splits.
        3. Fit PCA on training power spectra; project all splits onto n_comp components.
        4. Fit StandardScaler on training PCA coefficients; apply to all splits.
    """
    # Extract powers
    power_train_raw = data["power_train"]
    power_val_raw   = data["power_val"]
    power_test_raw  = data["power_test"]


    # Are we working in log power space
    if log_power:
        if np.any(power_train_raw <= 0) or np.any(power_val_raw <= 0) or np.any(power_test_raw <= 0):
            raise ValueError(
                "log_power=True requires all power spectrum values to be strictly positive."
            )
        power_train = np.log(power_train_raw) # Take the logs of the powers
        power_val   = np.log(power_val_raw)
        power_test  = np.log(power_test_raw)
    else:
        power_train = power_train_raw
        power_val   = power_val_raw
        power_test  = power_test_raw

    # Standardize the physical parameters
    params_scaler = StandardScaler().fit(data["raw_params_train"])
    params_train_scaled = params_scaler.transform(data["raw_params_train"])
    params_val_scaled = params_scaler.transform(data["raw_params_val"])
    params_test_scaled = params_scaler.transform(data["raw_params_test"])

    # Fit PCA on training power spectra 
    pca = PCA(n_components=n_comp).fit(power_train)
    evecs  = pca.components_.T # The first n_comp evecs

    # Project all power spectra onto the PCA basis
    pca_weights_train_raw = pca.transform(power_train)
    pca_weights_val_raw = pca.transform(power_val)
    pca_weights_test_raw = pca.transform(power_test)

    # Standardize the PCA coefficients
    weight_scaler = StandardScaler().fit(pca_weights_train_raw)
    pca_weights_train_scaled = torch.tensor(weight_scaler.transform(pca_weights_train_raw), dtype=torch.float32)
    pca_weights_val_scaled = torch.tensor(weight_scaler.transform(pca_weights_val_raw), dtype=torch.float32)
    pca_weights_test_scaled = torch.tensor(weight_scaler.transform(pca_weights_test_raw), dtype=torch.float32)

    return {
        "params_scaler": params_scaler,
        "weight_scaler": weight_scaler,
        "pca": pca,
        "evecs": evecs,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "log_power": log_power,
        "power_train": power_train_raw, # Return the (possibly log) power spectra for later use in plotting
        "power_val": power_val_raw,
        "power_test": power_test_raw, 
        "params_train_raw": data["raw_params_train"],
        "params_val_raw": data["raw_params_val"],
        "params_test_raw": data["raw_params_test"],
        "params_train_scaled": params_train_scaled,
        "params_val_scaled": params_val_scaled,
        "params_test_scaled": params_test_scaled,
        "pca_weights_train_raw": pca_weights_train_raw,
        "pca_weights_val_raw": pca_weights_val_raw,
        "pca_weights_test_raw": pca_weights_test_raw,
        "pca_weights_train_scaled": pca_weights_train_scaled,
        "pca_weights_val_scaled": pca_weights_val_scaled,
        "pca_weights_test_scaled": pca_weights_test_scaled,
        "k_train": data["k_train"],
        "k_val": data["k_val"],
        "k_test": data["k_test"],
    }


    # return {
    #     "params_scaler":           params_scaler,
    #     "weight_scaler":           weight_scaler,
    #     "pca":                     pca,
    #     "W":                       W,
    #     "eig_vals":                pca.explained_variance_,
    #     "explained_variance_ratio": pca.explained_variance_ratio_,
    #     "log_power":               log_power,
    #     "params_train":            params_train,
    #     "params_val":              params_val,
    #     "params_test":             params_test,
    #     "projected_coeffs_train":  projected_coeffs_train,
    #     "projected_coeffs_val":    projected_coeffs_val,
    #     "projected_coeffs_test":   projected_coeffs_test,
    #     "y_train_np":              y_train_np,
    #     "y_val_np":                y_val_np,
    #     "y_test_np":               y_test_np,
    #     "x_train":  torch.tensor(params_train, dtype=torch.float32),
    #     "x_val":    torch.tensor(params_val,   dtype=torch.float32),
    #     "x_test":   torch.tensor(params_test,  dtype=torch.float32),
    #     "y_train":  torch.tensor(y_train_np,   dtype=torch.float32),
    #     "y_val":    torch.tensor(y_val_np,     dtype=torch.float32),
    #     "y_test":   torch.tensor(y_test_np,    dtype=torch.float32),
    # }