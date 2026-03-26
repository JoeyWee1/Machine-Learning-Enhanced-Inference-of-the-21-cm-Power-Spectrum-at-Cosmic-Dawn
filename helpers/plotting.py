import numpy as np
import matplotlib.pyplot as plt


def plot_reconstructed_train(
    processed: dict,
    raw_data: dict,
    n_comp: int,
    idx: int = 0,
    plot: bool = False,
) -> float:
    """
    Reconstructs a single training power spectrum from its PCA coefficients
    and optionally plots the components, reconstruction, and fractional residual.

    Parameters
    ----------
    processed : dict
        Preprocessing artefacts from `preprocess()`. Required keys:
        - "W" : ndarray of shape (n_k, n_comp)
            PCA eigenvectors.
        - "projected_coeffs_train" : ndarray of shape (N_train, n_comp)
            Unscaled PCA coefficients for the training set.
        - "pca" : PCA
            Fitted PCA object (used to account for the spectrum mean).
    raw_data : dict
        Required keys:
        - "power_train" : ndarray of shape (N_train, n_k)
            Ground-truth training power spectra.
        - "k_train" : ndarray of shape (n_k,)
            Wavenumber array.
    n_comp : int
        Number of PCA components to use in the reconstruction (must be
        <= the number of components the PCA was fitted with).
    idx : int, optional
        Index of the training sample to reconstruct. Default: 0.
    plot : bool, optional
        If True, produces a three-panel figure showing the PCA component
        contributions, the reconstructed spectrum, and the fractional
        residual. Default: False.

    Returns
    -------
    float
        Mean fractional residual (%) between the original and reconstructed
        power spectrum, averaged over all k-modes.
    """
    k  = raw_data["k_train"][0]          # shape (n_k,)
    power_true = raw_data["power_train"][idx] # shape (n_k,)

    evecs  = processed["W"][:, :n_comp]                        # (n_k, n_comp)
    coeffs = processed["projected_coeffs_train"][idx, :n_comp] # (n_comp,)

    # Reconstruct: project back and add PCA mean
    reconstructed = coeffs @ evecs.T + processed["pca"].mean_  # (n_k,)
    reconstructed = np.exp(reconstructed) if processed["log_power"] else reconstructed

    frac_residual = 100.0 * np.abs(power_true - reconstructed) / np.abs(power_true)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))

        # Panel 1 — individual component contributions
        for i in range(n_comp):
            axes[0].semilogx(k, coeffs[i] * evecs[:, i], label=f"PC {i + 1}")
        axes[0].set_xlabel(r"$k$  [Mpc$^{-1}$]")
        axes[0].set_ylabel("Contribution")
        axes[0].set_title("PCA component contributions")
        axes[0].legend(fontsize=7)

        # Panel 2 — original vs reconstructed spectrum
        axes[1].loglog(k, power_true,    label="Original")
        axes[1].loglog(k, reconstructed, label=f"PCA ({n_comp} components)", ls="--")
        axes[1].set_xlabel(r"$k$  [Mpc$^{-1}$]")
        axes[1].set_ylabel(r"$\Delta^2(k)$  [mK$^2$]")
        axes[1].set_title("Reconstructed spectrum")
        axes[1].legend()

        # Panel 3 — fractional residual
        axes[2].semilogx(k, frac_residual)
        axes[2].set_xlabel(r"$k$  [Mpc$^{-1}$]")
        axes[2].set_ylabel("Fractional residual (%)")
        axes[2].set_title("Fractional residual")

        plt.tight_layout()
        plt.show()

    return float(frac_residual.mean())


def pca_fractional_residual(processed, raw_data, n_comp=10):
    """
    Compute and plot the distribution of mean fractional residuals from PCA
    reconstruction on the training set.

    Loops over every training sample, computes the mean fractional residual
    between the original and PCA-reconstructed power spectrum via
    plot_reconstructed_train, then plots a histogram of the results with
    mean and 95th percentile markers.

    Parameters
    ----------
    processed : dict
        Output of preprocess(), must contain 'params_train'.
    raw_data : dict
        Output of load_splits(), must contain 'power_train'.
    n_comp : int, optional
        Number of PCA components to use for reconstruction. Default is 10.

    Returns
    -------
    list of float
        Mean fractional residuals (%) for each training sample.

    """
    n_samps = len(processed['params_train'])
    residuals = []
    for j in range(0,n_samps):
        residuals.append(plot_reconstructed_train(processed, raw_data, n_comp=n_comp, plot=False, idx=j))

    # Plot histogram

    plt.figure(figsize=(6,6))
    plt.axvline(np.mean(residuals),        color="red",    linestyle="--", label=f"Mean: {np.mean(residuals):.2f}%")
    plt.axvline(np.percentile(residuals, 95), color="orange", linestyle="--", label=f"p95:  {np.percentile(residuals, 95):.2f}%")
    plt.hist(residuals, bins= 30)
    plt.title("Distibution of the mean fractional residuals in training set from PCA reconstruction")
    plt.xlabel("Mean fractional residual")
    plt.ylabel("Count")
    plt.show()

    # Print mean and 95th percentile of the fractional residuals
    print(f"Mean fractional residual: {np.mean(residuals):.2f}%")
    print(f"95th percentile of fractional residuals: {np.percentile(residuals, 95):.2f}%")
    return residuals


def plot_pca_train_weights(processed, raw_data, n_comp):
    """
    Print and plot scaled vs unscaled parameter ranges to motivate standardisation.

    For each of the first n_comp input parameters (e.g. L40_xray, fesc10,
    epsstar, h_fid), computes the value range before and after StandardScaler
    and plots both distributions side by side on the same axes. This illustrates
    how parameters with very different physical units and magnitudes are brought
    to a common scale, which is important for stable neural network training.

    Parameters
    ----------
    processed : dict
        Output of preprocess(), must contain 'x_train' (scaled parameters
        as a torch.Tensor).
    raw_data : dict
        Output of load_splits(), must contain 'raw_params_train'.
    n_comp : int
        Number of input parameter dimensions to plot. Should be <= 4 for
        the default parameter set (L40_xray, fesc10, epsstar, h_fid).

    Returns
    -------
    None
        Prints range statistics and displays a figure with n_comp subplots,
        each showing the unscaled and scaled distributions for one parameter.
    """
    param_names = ["L40_xray", "fesc10", "epsstar", "h_fid"]

    unscaled_ranges = []
    scaled_ranges   = []
    unscaled_means = []
    scaled_means   = []
    unscaled_params = raw_data['raw_params_train']
    scaled_params   = processed['x_train']

    # scaled_params may be a torch.Tensor
    if hasattr(scaled_params, 'cpu'):
        scaled_params = scaled_params.cpu().numpy()

    for i in range(0, 4):
        feature_unscaled_weights = unscaled_params[i, :]
        feature_scaled_weights   = scaled_params[i, :]
        unscaled_ranges.append(feature_unscaled_weights.max() - feature_unscaled_weights.min())
        scaled_ranges.append(feature_scaled_weights.max()   - feature_scaled_weights.min())
        unscaled_means.append(feature_unscaled_weights.mean())
        scaled_means.append(feature_scaled_weights.mean())

    print("Unscaled:")
    for i in range(4):
        vals = unscaled_params[:, i]
        print(f"Feature {i} has range {vals.min():.3e} to {vals.max():.3e} with mean {vals.mean():.3e}")

    

    print("Scaled:")
    for i in range(4):
        vals = scaled_params[:, i]
        print(f"Feature {i} has range {vals.min():.3e} to {vals.max():.3e} with mean {vals.mean():.3e}")


    fig, axes = plt.subplots(1, 4, figsize=(10, 8))
    if n_comp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        name = param_names[i] if i < len(param_names) else f"param_{i}"

        ax.hist(unscaled_params[:, i], bins=30, alpha=0.6,
                color="steelblue", label="Unscaled")
        ax.hist(scaled_params[:, i],   bins=30, alpha=0.6,
                color="darkorange", label="Scaled")

        ax.set_title(f"{name}\nrange: {unscaled_ranges[i]:.3g} → {scaled_ranges[i]:.3g}\nmean: {unscaled_means[i]:.3g} → {scaled_means[i]:.3g}")
        ax.axvline(scaled_means[i], color="black", linestyle="--", linewidth=0.8)


        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("Parameter distributions before and after StandardScaler", fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_reconstructions(
model_evaluation_data: dict,
raw_data: dict,
square_side: int = 5,
)-> None:
    """
    Plots reconstructed power spectra against ground truth for a random subset of test samples.

    Parameters
    ----------
    model_evaluation_data : dict
        Output from `evaluate_model()`, containing reconstructed spectra and true spectra.
    raw_data : dict
        Raw data from the loading stage. Required keys:
        - "power_test" : ndarray of shape (N_test, K)
            Ground-truth power spectra for the test set.
    square_side : int, optional
        Number of random test samples to plot along each side of the square grid (default is 5).
    """
    N_test = raw_data["power_test"].shape[0] # How many test samples
    n_samples = square_side * square_side
    sample_indices = np.random.choice(N_test, size=n_samples-1, replace=False)
    sample_indices = np.concatenate([[0], sample_indices])

    fig, ax = plt.subplots(square_side, square_side, figsize = (3*square_side+3, 3*square_side+3))
    for i, idx in enumerate(sample_indices):
        ax[i//square_side,i%square_side].loglog(raw_data["k_test"][idx],raw_data["power_test"][idx], label="True")
        ax[i//square_side,i%square_side].loglog(raw_data["k_test"][idx],model_evaluation_data["test_pred_spectra"][idx], '--',label="Predicted")
        ax[i//square_side,i%square_side].set_xlabel("k-mode index")
        ax[i//square_side,i%square_side].set_ylabel("Power Spectrum")
        ax[i//square_side,i%square_side].set_title(f"Test Sample {idx} - MAPE: {model_evaluation_data['mean_test_error_per_sample'][idx]:.2f}%")
        ax[i//square_side,i%square_side].grid()
    ax[0,0].legend()
    fig.suptitle("True vs Reconstructed Power Spectra")
    fig.show()

def plot_mape_distribution(model_evaluation_data: dict):
    mape = model_evaluation_data["mean_test_error_per_sample"]
    mean = model_evaluation_data['mean_percentage_error']
    p95 = model_evaluation_data['p95_percentage_error']
    plt.hist(mape, bins=75)
    plt.axvline(np.mean(mape), label=f'Mean: {mean:.2f}', ls='--', c='k')
    plt.axvline(np.quantile(mape, 0.95),label=f'95th percentile: {p95:.2f}', ls=':',  c='k')
    plt.xlabel('Percentage error (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()