import numpy as np
import matplotlib.pyplot as plt
from utils.general import set_seed

def make_nre_datasets(noisy_data: dict, processed: dict, plot: bool = False, savefig: bool = False) -> dict:
    """
    Construct joint/disjoint NRE datasets for train, validation, and test splits.

    For each split, pairs each noisy power spectrum with its true parameters
    (joint, label=1) and with shuffled parameters from another simulation
    (disjoint, label=0). The final dataset is shuffled before returning.

    Parameters
    ----------
    noisy_data : dict
        Output of noisify(). Required keys:
        'pnoisy_train', 'pnoisy_val', 'pnoisy_test' : ndarray of shape (N, 54)
        'theta5_train', 'theta5_val', 'theta5_test'  : ndarray of shape (N, 5)
    processed : dict
        Output of preprocess(). Required key:
        - 'k_train' : ndarray of shape (n_k,), wavenumbers (used for plotting only).
    plot : bool, optional
        If True, plots joint vs disjoint scatter for each parameter on the
        training set at k=0.1 Mpc^-1. Default False.
    savefig : bool, optional
        If True, saves each scatter plot to figs/shuffle_<param>.png. Default False.

    Returns
    -------
    dict with keys 'nre_train', 'nre_val', 'nre_test', each containing:
        - 'pnoisy' : ndarray of shape (2N, 54)
        - 'theta5' : ndarray of shape (2N, 5)
        - 'labels' : ndarray of shape (2N,), 1 for joint, 0 for disjoint
    """
    pnoisy_train = noisy_data["pnoisy_train"]
    pnoisy_val = noisy_data["pnoisy_val"]
    pnoisy_test = noisy_data["pnoisy_test"]

    theta5_train = noisy_data["theta5_train"]
    theta5_val = noisy_data["theta5_val"]
    theta5_test = noisy_data["theta5_test"]
    
    k_train = processed["k_train"]  

    def _make_nre_dataset(pnoisy, theta5, plot=plot, savefig=savefig):
        set_seed()
        pnoisy_j = pnoisy #(n_models, 54)
        theta5_j = theta5 #(n_models, 5)

        # disjoint pais
        shuffled_idx = np.random.permutation(len(pnoisy))
        pnoisy_d = pnoisy #(n_models, 54)
        theta5_d = theta5[shuffled_idx] # (n_models, 5)

        # concat
        pnoisy_nre = np.concatenate([pnoisy_j, pnoisy_d]) #(2*n_models, 54)
        theta5_nre = np.concatenate([theta5_j, theta5_d]) #(2*n_models, 5)
        labels_nre = np.concatenate([np.ones(len(pnoisy)), np.zeros(len(pnoisy))]) #(2*n_models)

        if plot == True:
            idx_k = np.argmin(np.abs(k_train - 0.1))
            pnoisy_slice = pnoisy_nre[:, idx_k]

            labels = [
                r"$L_{40}^{\text{X-ray}}$",
                r"$f_{\text{esc}}^{10}$",
                r"$\epsilon$",
                r"$h$",
                r"$f_{\text{noise}}$",
            ]
            savelabels = ["L40_xray", "fesc10", "epsilon", "h", "fnoise"]

             # Plotting
            for i, label in enumerate(labels):
                theta5_slice = theta5_nre[:, i]
                fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
                ax.scatter(theta5_slice[:len(pnoisy)], pnoisy_slice[:len(pnoisy)], color='blue', label="Joint", s=1, alpha=0.3)
                ax.scatter(theta5_slice[len(pnoisy):], pnoisy_slice[len(pnoisy):], color='red',  label="Disjoint", s=1, alpha=0.3)
                ax.set_xlabel(label)
                ax.set_ylabel(r"$P_{\rm noisy}(k=0.1)$")
                ax.legend()
                fig.tight_layout()
                if savefig:
                    fig.savefig(f"outputs/figs/shuffle_{savelabels[i]}.png", bbox_inches='tight')
                plt.show()
        # shuffle final 
        shuffled_idx = np.random.permutation(2*len(pnoisy))
        pnoisy_nre =pnoisy_nre[shuffled_idx] 
        theta5_nre = theta5_nre[shuffled_idx]
        labels_nre = labels_nre[shuffled_idx]

        nre_dataset={
            "pnoisy":pnoisy_nre,
            "theta5":theta5_nre,
            "labels":labels_nre
        }
        return nre_dataset
    set_seed()
    nre_train = _make_nre_dataset(pnoisy_train, theta5_train, plot=plot, savefig=savefig)
    nre_val = _make_nre_dataset(pnoisy_val, theta5_val, plot=False, savefig=False)
    nre_test = _make_nre_dataset(pnoisy_test, theta5_test, plot=False, savefig=False)

    return {
        "nre_train": nre_train,
        "nre_val": nre_val,
        "nre_test": nre_test,}

    

