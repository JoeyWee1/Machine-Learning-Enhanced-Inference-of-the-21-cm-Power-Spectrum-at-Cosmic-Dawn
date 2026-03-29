import numpy as np
import matplotlib.pyplot as plt

def plot_pca_train_weights(processed: dict, n_comp: int) -> dict:
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
    param_names = ["L40_xray", "fesc10", "epsilon", "h"]

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
        name = param_names[i]
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
