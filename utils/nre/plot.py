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

def plot_noisy_data(noisy_data: dict, processed: dict, idx: int, n_fnoise: int, savefig: str = None) -> None:
    """
    Plot noisy mock observations and their fractional residuals for a single training sample.

    Left panel: fractional residual (pnoisy - pmodel) / pmodel for each fnoise draw.
    Right panel: noisy spectra overlaid on the noiseless spectrum.
    Colours are matched across both panels per realisation.

    Parameters
    ----------
    noisy_data : dict
        Output of noisify(). Required keys:
        - 'pnoisy_train' : ndarray of shape (n_train * n_fnoise, 54), noisy spectra.
        - 'theta5_train' : ndarray of shape (n_train * n_fnoise, 5), parameters with fnoise.
    processed : dict
        Output of preprocess(). Required keys:
        - 'k_train'     : ndarray of shape (n_train, 54), wavenumber arrays.
        - 'power_train' : ndarray of shape (n_train, 54), noiseless power spectra.
    idx : int
        Index of the training simulation to plot.
    n_fnoise : int
        Number of noise realisations per simulation, used to index into pnoisy_train.
    savefig : str, optional
        If provided, saves the figure to this filename instead of displaying it. Default None.

    Returns
    -------
    None
        Displays the plot inline or saves it to a file if savefig is provided.
    """
    k       = processed["k_train"][idx]
    pmodel  = processed["power_train"][idx]
    start   = idx * n_fnoise

    pnoisy  = noisy_data["pnoisy_train"][start : start + n_fnoise]   # (n_fnoise, 54)
    fnoise  = noisy_data["theta5_train"][start : start + n_fnoise, 4]  # (n_fnoise,)

    colours = plt.cm.plasma(np.linspace(0.1, 0.9, n_fnoise))

    fig, axes = plt.subplots( 2, 1, figsize=(4, 6), dpi=150)


    for i in range(n_fnoise):
        # label = rf"$f_{{\text{{noise}}}}={fnoise[i]:.3f}$"
        residual = (pnoisy[i] - pmodel) / pmodel

        axes[0].semilogx(k, residual, color=colours[i], alpha=0.7)
        axes[1].loglog(k, pnoisy[i],  color=colours[i], alpha=0.7, linestyle='--')

    axes[1].loglog(k, pmodel, color='k', linewidth=1.5, label='Noiseless')

    axes[0].axhline(0, color='k', linewidth=1, linestyle=':')
    axes[0].set_xlabel(r'$k$  [Mpc$^{-1}$]')
    axes[0].set_ylabel(r'Fractional Residual')
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel(r'$k$  [Mpc$^{-1}$]')
    axes[1].set_ylabel(r'$\Delta^2(k)$  [mK$^2$]')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=150)

    plt.show()
