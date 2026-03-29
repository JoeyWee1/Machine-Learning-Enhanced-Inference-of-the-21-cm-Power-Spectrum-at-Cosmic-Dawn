import matplotlib.pyplot as plt
import numpy as np

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
    plt.figure(figsize=(5, 5))
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
    plt.figure(figsize=(6,4))
    plt.yscale('log')
    plt.plot(cumulative_explained_variance, marker='o')
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