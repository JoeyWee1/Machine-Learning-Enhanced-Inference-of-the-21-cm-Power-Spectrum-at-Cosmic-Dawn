import matplotlib.pyplot as plt
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