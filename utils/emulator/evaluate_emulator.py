import numpy as np
import torch
import torch.nn as nn


def evaluate_emulator(
    model: nn.Module,
    processed: dict,
    device: str,
) -> dict:
    """
    Evaluates a trained emulator on the test set in both PCA-coefficient space
    and reconstructed power-spectrum space.

    Parameters
    ----------
    model : nn.Module
        Trained emulator, as returned by `train_emulator`.
    processed : dict
        Preprocessing artefacts from `utils.preprocess.preprocess()`. Required keys:
        - "params_test_scaled"        : ndarray of shape (N_test, input_dim)
            Standardized test parameters.
        - "pca_weights_test_scaled"   : Tensor of shape (N_test, n_comp)
            Standardized PCA coefficients for the test set.
        - "weight_scaler"             : StandardScaler
            Used to inverse-transform predicted coefficients.
        - "pca"                       : PCA
            Used to reconstruct power spectra from unscaled coefficients.
        - "log_power"                 : bool
            If True, reconstructed spectra are exponentiated.
        - "power_test"                : ndarray of shape (N_test, K)
            Ground-truth power spectra for the test set.
    device : str
        Device the model lives on ("cpu" or "cuda").

    Returns
    -------
    dict
        - "test_loss_normalised_space" : float
            MSE between predicted and true standardized PCA coefficients.
        - "mean_percentage_error" : float
            Mean per-sample MAPE across the test set, averaged over all k-modes.
        - "p95_percentage_error" : float
            95th-percentile MAPE across samples.
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
    """
    model.eval()
    x_test = torch.tensor(processed["params_test_scaled"], dtype=torch.float32).to(device)
    y_test = processed["pca_weights_test_scaled"].to(device)

    with torch.no_grad():
        pred_y_test = model(x_test).cpu().numpy()

    test_loss = float(
        nn.functional.mse_loss(torch.tensor(pred_y_test), y_test.cpu()).item()
    )

    pred_weights_test = processed["weight_scaler"].inverse_transform(pred_y_test)
    test_pred_spectra = processed["pca"].inverse_transform(pred_weights_test)

    if processed.get("log_power", False):
        test_pred_spectra = np.exp(test_pred_spectra)

    denom = np.maximum(np.abs(processed["power_test"]), 1e-8)
    mean_test_error = 100.0 * np.mean(
        np.abs(processed["power_test"] - test_pred_spectra) / denom,
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
