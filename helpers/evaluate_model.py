import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def evaluate_model(
    model: nn.Module,
    processed: dict,
    raw_data: dict,
    device: str,
) -> dict:
    """
    Evaluates a trained emulator on the test set in both PCA-coefficient space
    and reconstructed power-spectrum space.

    Parameters
    ----------
    model : nn.Module
        Trained emulator, as returned by `train_model`.
    processed : dict
        Preprocessing artefacts from `preprocess()`. Required keys:
        - "x_test" : torch.Tensor of shape (N_test, input_dim)
            Standardized test parameters.
        - "y_test" : torch.Tensor of shape (N_test, n_comp)
            Standardized PCA coefficients for the test set.
        - "weight_scaler" : StandardScaler
            Used to inverse-transform predicted coefficients.
        - "pca" : PCA
            Used to reconstruct power spectra from unscaled coefficients.
    raw_data : dict
        Raw data from the loading stage. Required keys:
        - "power_test" : ndarray of shape (N_test, K)
            Ground-truth power spectra for the test set.
    device : str
        Device the model lives on ("cpu" or "cuda").

    Returns
    -------
    dict
        - "test_loss_normalised_space" : float
            MSE between predicted and true standardized PCA coefficients.
        - "mean_percentage_error" : float
            Mean per-sample mean absolute percentage error (MAPE) across the test set,
            averaged over all k-modes.
        - "p95_percentage_error" : float
            95th-percentile MAPE across samples — a measure of worst-case performance.
        - "pred_y_test" : ndarray of shape (N_test, n_comp)
            Raw model predictions in standardized PCA-coefficient space.
        - "pred_weights_test" : ndarray of shape (N_test, n_comp)
            Predictions inverse-transformed to unscaled PCA-coefficient space.
        - "test_pred_spectra" : ndarray of shape (N_test, K)
            Reconstructed power spectra.
        - "mean_test_error_per_sample" : ndarray of shape (N_test,)
            Per-sample MAPE (%), averaged over k-modes.

    Notes
    -----
    MAPE is computed as::

        100 * mean_k( |P_true - P_pred| / max(|P_true|, 1e-8) )

    The 1e-8 floor prevents division by zero near nodes of the power spectrum.
    """
    model.eval()
    x_test = processed["x_test"].to(device)
    y_test = processed["y_test"].to(device)

    with torch.no_grad():
        pred_y_test = model(x_test).cpu().numpy()

    test_loss = float(
        nn.functional.mse_loss(torch.tensor(pred_y_test), y_test.cpu()).item()
    )

    pred_weights_test = processed["weight_scaler"].inverse_transform(pred_y_test) # undo the standardization of the PCA coefficients to get raw PCA coefficients
    test_pred_spectra = processed["pca"].inverse_transform(pred_weights_test) #  dot product with PCA eigenvectors to get reconstructed spectra in log space if log_power was True, or linear space if log_power was False

    if processed.get("log_power", False):
        test_pred_spectra = np.exp(test_pred_spectra)

    denom = np.maximum(np.abs(raw_data["power_test"]), 1e-8)
    mean_test_error = 100.0 * np.mean(
        np.abs(raw_data["power_test"] - test_pred_spectra) / denom,
        axis=1,
    )

    return {
        "test_loss_normalised_space":  test_loss,
        "mean_percentage_error":       float(np.mean(mean_test_error)),
        "p95_percentage_error":        float(np.quantile(mean_test_error, 0.95)),
        "pred_y_test":                 pred_y_test,
        "pred_weights_test":           pred_weights_test,
        "test_pred_spectra":           test_pred_spectra,
        "mean_test_error_per_sample":  mean_test_error,
    }

def predict_spectrum(
        model: nn.Module,
        params: np.ndarray,
        processed: dict,
) -> np.ndarray:
    """
    Predicts the power spectrum for a single set of input parameters.

    Parameters
    ----------
    model : nn.Module
        Trained emulator, as returned by `train_model`.
    params : np.ndarray of shape (4,)
        Standardized input parameters in the order:
        ["L40_xray", "fesc10", "epsstar", "h_fid"].
    processed : dict
        Preprocessing artefacts from `preprocess()`. Required keys:
        - "weight_scaler" : StandardScaler
            Used to inverse-transform predicted PCA coefficients.
        - "pca" : PCA
            Used to reconstruct the power spectrum from unscaled coefficients
            via a dot product with the principal component eigenvectors.
        - "log_power" : bool, optional
            If True, the reconstructed spectrum is exponentiated to undo
            the log-transform applied during preprocessing.

    Returns
    -------
    np.ndarray of shape (54,)
        Reconstructed power spectrum in linear (not log) space.
    """
    model.eval()
    with torch.no_grad():
        params_tensor = torch.tensor(params, dtype=torch.float32)
        pred_weights_scaled = model(params_tensor).cpu().numpy() 
        # Inverse transform expects (1, n_comp) rn we have (n_comp,)
        if pred_weights_scaled.ndim == 1:
            pred_weights_scaled = pred_weights_scaled.reshape(1, -1) 
    
    pred_weights_raw = processed["weight_scaler"].inverse_transform(pred_weights_scaled) # undo the standardization of the PCA coefficients to get raw PCA coefficients
    pred_spectra = processed["pca"].inverse_transform(pred_weights_raw) #  dot product with PCA eigenvectors to get reconstructed spectra in log space 

    if processed.get("log_power", False):
        pred_spectra = np.exp(pred_spectra)

    return pred_spectra[0]




