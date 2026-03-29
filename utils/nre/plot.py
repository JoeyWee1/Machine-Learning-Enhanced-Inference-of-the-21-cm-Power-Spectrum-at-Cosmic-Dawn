import numpy as np
import matplotlib.pyplot as plt

# def plot_noisy_data(noisy_data: dict, processed: dict, idx: int, n_fnoise: int):
#     k_train = processed["k_train"]
#     pnoisy_train = noisy_data["pnoisy_train"]
#     pmodel_train = processed["power_train"]
    
#     start = idx * n_fnoise

#     plt.figure(figsize=(8,6), dpi=150)
#     for i in range(0, n_fnoise):
#         plt.loglog(k_train[idx], pnoisy_train[start + i], color='red', linestyle='--', alpha=0.3)
#     plt.loglog(k_train[idx], pmodel_train[idx], label = "Ideal")
#     plt.show()

def plot_noisy_data(noisy_data: dict, processed: dict, idx: int, n_fnoise: int) -> None:
    """
    Plot noisy mock observations against the noiseless power spectrum for a single training sample.

    Overlays all n_fnoise noisy realisations of a single simulation in red,
    with the underlying noiseless spectrum in blue for comparison.

    Parameters
    ----------
    noisy_data : dict
        Output of noisify(). Required keys:
        - 'pnoisy_train' : ndarray of shape (n_train * n_fnoise, 54), noisy spectra.
    processed : dict
        Output of preprocess(). Required keys:
        - 'k_train'     : ndarray of shape (n_train, 54), wavenumber arrays.
        - 'power_train' : ndarray of shape (n_train, 54), noiseless power spectra.
    idx : int
        Index of the training simulation to plot.
    n_fnoise : int
        Number of noise realisations per simulation, used to index into pnoisy_train.

    Returns
    -------
    None
        Displays the plot inline.
    """
    k_train      = processed["k_train"]
    pnoisy_train = noisy_data["pnoisy_train"]
    pmodel_train = processed["power_train"]

    start = idx * n_fnoise

    plt.figure(figsize=(8, 6), dpi=150)
    for i in range(n_fnoise):
        plt.loglog(k_train[idx], pnoisy_train[start + i],
                   color='red', linestyle='--', alpha=0.3,
                   label='Noisy realisations' if i == 0 else None)  # label only first
    plt.loglog(k_train[idx], pmodel_train[idx],
               color='steelblue', linewidth=2, label='Noiseless')

    plt.xlabel(r'$k$  [Mpc$^{-1}$]')
    plt.ylabel(r'$\Delta^2(k)$  [mK$^2$]')
    plt.title(f'Noisy Mock Observations — Training Sample {idx} ({n_fnoise} realisations)')
    plt.legend()
    plt.tight_layout()
    plt.show()
