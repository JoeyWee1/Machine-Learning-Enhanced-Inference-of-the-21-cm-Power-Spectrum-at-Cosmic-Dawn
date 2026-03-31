import matplotlib.pyplot as plt
import numpy as np
import torch 

def plot_power_spectra(raw_data: dict, idx_min: int = 0, idx_max: int = 8000, interval: int = 100) -> None:
    """
    Plot a subset of 21-cm power spectra on a log-log scale.

    Parameters
    ----------
    raw_data : dict
        Dictionary containing 'k_train' and 'power_train' arrays of shape (n_sims, n_kbins).
    idx_min : int, optional
        Index of first simulation to plot. Default 0.
    idx_max : int, optional
        Index of last simulation to plot (exclusive). Default 8000.
    interval : int, optional
        Step size between plotted simulations e.g. interval=100 plots sims 0, 100, 200, ... Default 100.

    Returns
    -------
    None
        Displays the plot inline.

    Example
    -------
    >>> plot_power_spectra(raw_data, idx_min=0, idx_max=500, interval=50)
    """
    plt.figure(figsize=(5, 5), dpi = 150)
    for i in  range(idx_min, idx_max, interval):
        plt.loglog(raw_data['k_train'][i], raw_data['power_train'][i], label=f'Sim {i+1}')
    plt.xlabel('k-bin')
    plt.ylabel('Power Spectrum')
    plt.title('Examples of 21-cm Power Spectra')
    plt.show()

def evr_stats(processed: dict) -> None:
    """
    Plot and print cumulative explained variance ratio for PCA components.

    Plots cumulative explained variance on a log scale with reference lines
    at 99% and 99.9%, and prints the number of components required to reach
    99%, 99.9%, and 99.99% explained variance.

    Parameters
    ----------
    processed : dict
        Output of preprocess(), must contain:
        - 'evecs'                    : ndarray of shape (n_k, n_comp)
        - 'explained_variance_ratio' : ndarray of shape (n_comp,)

    Returns
    -------
    None
        Displays the plot and prints threshold statistics inline.

    Example
    -------
    >>> evr_stats(processed)
    99.000% threshold explained by first 12 components
    99.900% threshold explained by first 18 components
    99.990% threshold explained by first 24 components
    """
    evecs = processed['evecs']
    explained_variance_ratio = processed['explained_variance_ratio']
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Plot
    plt.figure(figsize=(6,4), dpi=150)
    plt.yscale('log')
    plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.axhline(0.99, ls='--', c='k', label='99%')
    plt.axhline(0.999, ls=':', c='k', label='99.9%')
    plt.xlabel("Modes included")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # State
    thresholds = [0.99, 0.999, 0.9999]
    for threshold in thresholds:
        n = np.searchsorted(cumulative_explained_variance, threshold) + 1
        print(f"{100 * threshold:.3f}% threshold explained by first {n} components")

def plot_reconstructed_train(
    processed: dict,
    n_comp: int,
    idx: int = 0,
    plot: bool = False,
) -> float:
    """
    Reconstruct a single training power spectrum from its PCA coefficients
    and optionally plot the components, reconstruction, and fractional residual.

    Parameters
    ----------
    processed : dict
        Output of preprocess(). Required keys:
        - 'k_train'                : ndarray of shape (n_k,)
        - 'power_train'            : ndarray of shape (N, n_k), possibly log-transformed.
        - 'evecs'                  : ndarray of shape (n_k, n_comp).
        - 'pca_weights_train_raw'  : ndarray of shape (N, n_comp), unscaled PCA coefficients.
        - 'pca'                    : fitted PCA object (used for mean correction).
        - 'log_power'              : bool, whether to exponentiate the reconstruction.
    n_comp : int
        Number of PCA components to use in the reconstruction.
    idx : int, optional
        Index of the training sample to reconstruct. Default 0.
    plot : bool, optional
        If True, produces a three-panel figure showing component contributions,
        the reconstructed spectrum, and the fractional residual. Default False.

    Returns
    -------
    float
        Mean fractional residual (%) between the original and reconstructed
        power spectrum, averaged over all k-modes.

    Example
    -------
    >>> residual = plot_reconstructed_train(processed, n_comp=10, idx=0, plot=True)
    >>> print(f"Mean residual: {residual:.3f}%")
    """
    k  = processed["k_train"][0]          # shape (n_k,)
    power_true = processed["power_train"][idx]  # shape (n_k,)

    evecs  = processed["evecs"][:, :n_comp]                        # (n_k, n_comp)
    coeffs = processed["pca_weights_train_raw"][idx, :n_comp] # (n_comp,)

    # Reconstruct: project back and add PCA mean
    reconstructed = coeffs @ evecs.T + processed["pca"].mean_ # (n_k,)
    reconstructed = np.exp(reconstructed) if processed["log_power"] else reconstructed

    frac_residual = 100.0 * np.abs(torch.tensor(power_true) - reconstructed) / np.abs(power_true)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi = 150)

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

def pca_fractional_residual(processed: dict, n_comp: int = 10) -> list[float]:
    """
    Compute and plot the distribution of PCA reconstruction residuals over the training set.

    Reconstructs every training power spectrum from its PCA coefficients using
    plot_reconstructed_train(), collects the mean fractional residual for each,
    and plots a histogram with mean and 95th percentile reference lines.

    Parameters
    ----------
    processed : dict
        Output of preprocess(). Must contain 'params_train_scaled' to determine
        the number of training samples, plus all keys required by
        plot_reconstructed_train().
    n_comp : int, optional
        Number of PCA components to use in each reconstruction. Default 10.

    Returns
    -------
    list of float
        Mean fractional residual (%) for each training sample.

    Example
    -------
    >>> residuals = pca_fractional_residual(processed, n_comp=10)
    Mean fractional residual: 0.43%
    95th percentile of fractional residuals: 1.12%
    """
    n_samps = len(processed['params_train_raw'])
    residuals = []
    for j in range(0,n_samps):
        residuals.append(plot_reconstructed_train(processed, n_comp=n_comp, plot=False, idx=j))

    # Plot histogram
    plt.figure(figsize=(6,6), dpi = 150)
    plt.axvline(np.mean(residuals), color="red", linestyle="--", label=f"Mean: {np.mean(residuals):.2f}%")
    plt.axvline(np.percentile(residuals, 95), color="orange", linestyle="--", label=f"p95:  {np.percentile(residuals, 95):.2f}%")
    plt.hist(residuals, bins= 30)
    plt.title("Distibution of the mean fractional residuals in training set from PCA reconstruction")
    plt.xlabel("Mean fractional residual")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    return residuals