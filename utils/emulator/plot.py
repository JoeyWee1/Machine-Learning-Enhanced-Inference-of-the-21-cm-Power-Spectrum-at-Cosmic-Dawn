import numpy as np
import matplotlib.pyplot as plt

def plot_pca_train_params(processed: dict, n_comp: int) -> dict:
    """
    Plot parameter distributions before and after standardisation and return feature domains.

    For each of the 4 astrophysical parameters, plots overlaid histograms of the
    raw and StandardScaler-normalised training distributions with mean reference
    lines, and computes the min/max domain of each unscaled parameter.

    Parameters
    ----------
    processed : dict
        Output of preprocess(). Required keys:
        - 'params_train_raw'    : ndarray of shape (N, 4), unscaled training parameters.
        - 'params_train_scaled' : ndarray or Tensor of shape (N, 4), standardized parameters.
    n_comp : int
        Number of PCA components (currently unused, reserved for future use).

    Returns
    -------
    dict
        Mapping of parameter name to [min, max] range over the training set. Keys:
        - 'L40_xray'  : [float, float]
        - 'fesc10'    : [float, float]
        - 'epsilon'   : [float, float]
        - 'h'         : [float, float]

    Example
    -------
    >>> domains = plot_pca_train_weights(processed, n_comp=10)
    >>> domains['L40_xray']
    [0.1, 1000.0]
    """
    param_names  = ["L40_xray", "fesc10", "epsilon", "h"]
    param_labels = [
        r"$L_{40}^{\text{X-ray}}$",
        r"$f_{\text{esc}}^{10}$",
        r"$\epsilon$",
        r"$h$",
    ]

    unscaled_ranges = []
    scaled_ranges   = []
    unscaled_means = []
    scaled_means   = []
    unscaled_params = processed['params_train_raw']
    scaled_params   = processed['params_train_scaled']

    # scaled_params may be a torch.Tensor
    if hasattr(scaled_params, 'cpu'):
        scaled_params = scaled_params.cpu().numpy()

    # Finds the range (max, min) and mean of features
    for i in range(0, 4):
        feature_unscaled_weights = unscaled_params[:,i]
        feature_scaled_weights   = scaled_params[:,i]

        unscaled_ranges.append([feature_unscaled_weights.max(), feature_unscaled_weights.min()])
        scaled_ranges.append([feature_scaled_weights.max(), feature_scaled_weights.min()])
        unscaled_means.append(feature_unscaled_weights.mean())
        scaled_means.append(feature_scaled_weights.mean())

    # Make unscllead feature domains
    unscaled_feature_domains = {}

    for i in range(4):
        vals = unscaled_params[:, i]
        unscaled_feature_domains[param_names[i]] = [float(vals.min()), float(vals.max())]
     
     # Plotting muahahaha
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), dpi=150)

    for i in range(4):
        name = param_labels[i]
        ax_u = axes[0, i]  # top row: unscaled
        ax_s = axes[1, i]  # bottom row: scaled

        # Unscaled
        ax_u.hist(unscaled_params[:, i], bins=30, alpha=0.7, color="steelblue")
        ax_u.axvline(unscaled_means[i], color="red", linestyle="--", linewidth=1.5,
                     label=f"μ={unscaled_means[i]:.2f}")
        ax_u.set_title(f"{name}\nUnscaled [{unscaled_ranges[i][1]:.2f}, {unscaled_ranges[i][0]:.2f}]", fontsize=8)
        ax_u.set_ylabel("Count")
        ax_u.legend(fontsize=8)

        # Scaled
        ax_s.hist(scaled_params[:, i], bins=30, alpha=0.7, color="darkorange")
        ax_s.axvline(scaled_means[i], color="red", linestyle="--", linewidth=1.5,
                     label=f"μ={scaled_means[i]:.2f}")
        ax_s.set_title(f"Scaled [{scaled_ranges[i][1]:.2f}, {scaled_ranges[i][0]:.2f}]", fontsize=8)
        ax_s.set_xlabel("Value")
        ax_s.set_ylabel("Count")
        ax_s.legend(fontsize=8)

    fig.suptitle("Parameter distributions before and after StandardScaler", fontsize=13)
    plt.tight_layout()
    plt.show()

    return unscaled_feature_domains

def plot_emulator_test_reconstructions(model_evaluation_data: dict, processed: dict, 
                                       square_side: int = 3, seed: int = 1701) -> None:
    """
    Plot a grid of true vs emulator-predicted power spectra on the test set.

    Randomly samples square_side² test spectra (with index 0 always included)
    and plots each as a log-log true/predicted pair with MAPE in the title.

    Parameters
    ----------
    model_evaluation_data : dict
        Output of evaluate_model(). Required keys:
        - 'test_pred_spectra'           : ndarray of shape (N_test, n_k), predicted spectra.
        - 'mean_test_error_per_sample'  : ndarray of shape (N_test,), MAPE per sample.
    processed : dict
        Output of preprocess(). Required keys:
        - 'power_test' : ndarray of shape (N_test, n_k), true power spectra.
        - 'k_test'     : ndarray of shape (n_k,), wavenumber array.
    square_side : int, optional
        Side length of the subplot grid, producing square_side² panels. Default 3.

    Returns
    -------
    None
        Displays the plot inline.
    """
    N_test = processed["power_test"].shape[0]
    n_samples = square_side * square_side
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(N_test, size=n_samples - 1, replace=False)
    sample_indices = np.concatenate([[0], sample_indices])

    fig, axes = plt.subplots(
        square_side, square_side,
        figsize=(4 * square_side, 4 * square_side),
        dpi=150,
        constrained_layout=True,   # ← prevents overlap
    )
    # Increase vertical space between rows
    fig.get_layout_engine().set(hspace=0.15)

    for i, idx in enumerate(sample_indices):
        ax = axes[i // square_side, i % square_side]
        ax.loglog(processed["k_test"][idx], processed["power_test"][idx],
                  color="steelblue", label="True")
        ax.loglog(processed["k_test"][idx], model_evaluation_data["test_pred_spectra"][idx],
                  '--', color="darkorange", label="Predicted")
        ax.set_xlabel("k-mode index")
        ax.set_ylabel("Power Spectrum")
        ax.set_title(f"Sample {idx} — MAPE: {model_evaluation_data['mean_test_error_per_sample'][idx]:.2f}%",
                     fontsize=9, pad=4)
        ax.legend(fontsize=6)

    # Single legend outside the grid
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("True vs Reconstructed Power Spectra", fontsize=13)
    plt.show()


def plot_mape_distribution(model_evaluation_data: dict) -> None:
    """
    Plot the distribution of per-sample MAPE across the test set.

    Displays a histogram of mean absolute percentage errors with reference
    lines at the mean and 95th percentile.

    Parameters
    ----------
    model_evaluation_data : dict
        Output of evaluate_model(). Required keys:
        - 'mean_test_error_per_sample' : ndarray of shape (N_test,), MAPE per sample.
        - 'mean_percentage_error'      : float, mean MAPE across the test set.
        - 'p95_percentage_error'       : float, 95th percentile MAPE across the test set.

    Returns
    -------
    None
        Displays the plot inline.
    """
    mape = model_evaluation_data["mean_test_error_per_sample"]
    mean = model_evaluation_data['mean_percentage_error']
    p95  = model_evaluation_data['p95_percentage_error']

    plt.figure(dpi=150)
    plt.hist(mape, bins=75)
    plt.axvline(np.mean(mape), label=f'Mean: {mean:.2f}%', ls='--', c='k')
    plt.axvline(np.quantile(mape, 0.95), label=f'95th percentile: {p95:.2f}%', ls=':', c='k')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()