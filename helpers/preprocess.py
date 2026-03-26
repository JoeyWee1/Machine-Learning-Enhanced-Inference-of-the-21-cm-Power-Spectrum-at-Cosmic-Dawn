import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(data: dict, n_comp: int) -> dict:
    """
    Preprocesses raw simulation data for neural network training via parameter
    standardization, PCA compression of power spectra, and coefficient standardization.

    Parameters
    ----------
    data : dict
        Dictionary containing raw data splits with the following keys:
        - "raw_params_train/val/test" : array-like of shape (N, n_params)
            Cosmological/simulation parameters for each split.
        - "power_train/val/test" : array-like of shape (N, n_k)
            Power spectra for each split.
        - "k_train/val/test" : array-like
            Wavenumber arrays for each split.
        - "train/val/test_files" : list of str
            Source file paths for each split.
    n_comp : int
        Number of PCA components to retain when compressing the power spectra.

    Returns
    -------
    dict
        Dictionary containing:
        - "params_scaler" : StandardScaler
            Fitted scaler for the input parameters.
        - "weight_scaler" : StandardScaler
            Fitted scaler for the PCA-projected coefficients.
        - "pca" : PCA
            Fitted PCA object with n_comp components.
        - "W" : ndarray of shape (K, n_comp)
            PCA eigenvectors (principal components transposed).
        - "eig_vals" : ndarray of shape (n_comp,)
            Explained variance of each principal component.
        - "params_train/val/test" : ndarray
            Standardized input parameters for each split.
        - "projected_coeffs_train/val/test" : ndarray of shape (N, n_comp)
            Raw PCA-projected power spectrum coefficients for each split.
        - "y_train/val/test_np" : ndarray of shape (N, n_comp)
            Standardized PCA coefficients as NumPy arrays for each split.
        - "x_train/val/test" : torch.Tensor of shape (N, n_params), dtype=float32
            Standardized parameters as PyTorch tensors for each split.
        - "y_train/val/test" : torch.Tensor of shape (N, n_comp), dtype=float32
            Standardized PCA coefficients as PyTorch tensors for each split.

    Notes
    -----
    The preprocessing pipeline is:
        1. Fit a StandardScaler on training parameters and apply to all splits.
        2. Fit PCA on training power spectra; project all splits onto n_comp components.
        3. Fit a StandardScaler on training PCA coefficients and apply to all splits.

    All scalers and the PCA object are fitted exclusively on training data to
    prevent data leakage into the validation and test splits.
    """
    # Standardize the parameters
    params_scaler = StandardScaler().fit(data["raw_params_train"])
    params_train = params_scaler.transform(data["raw_params_train"])
    params_val = params_scaler.transform(data["raw_params_val"])
    params_test = params_scaler.transform(data["raw_params_test"])

    # Perform PCA on the power spectra
    pca = PCA(n_components=n_comp)
    pca.fit(data["power_train"])

    W = pca.components_.T # Eigenvectors (principal components)

    # Project the power spectra onto the PCA components
    projected_coeffs_train = pca.transform(data["power_train"])
    projected_coeffs_val = pca.transform(data["power_val"])
    projected_coeffs_test = pca.transform(data["power_test"])   

    # Standardize the projected coefficients
    weight_scaler = StandardScaler().fit(projected_coeffs_train)
    y_train_np = weight_scaler.transform(projected_coeffs_train)
    y_val_np = weight_scaler.transform(projected_coeffs_val)
    y_test_np = weight_scaler.transform(projected_coeffs_test)

    processed = {
        "params_scaler": params_scaler,
        "weight_scaler": weight_scaler,
        "pca": pca,
        "W": W,
        "eig_vals": pca.explained_variance_,
        "params_train": params_train,
        "params_val": params_val,
        "params_test": params_test,
        "projected_coeffs_train": projected_coeffs_train,
        "projected_coeffs_val": projected_coeffs_val,
        "projected_coeffs_test": projected_coeffs_test,
        "y_train_np": y_train_np,
        "y_val_np": y_val_np,
        "y_test_np": y_test_np,
        "x_train": torch.tensor(params_train, dtype=torch.float32),
        "x_val": torch.tensor(params_val, dtype=torch.float32),
        "x_test": torch.tensor(params_test, dtype=torch.float32),
        "y_train": torch.tensor(y_train_np, dtype=torch.float32),
        "y_val": torch.tensor(y_val_np, dtype=torch.float32),
        "y_test": torch.tensor(y_test_np, dtype=torch.float32),
    }
    return processed
