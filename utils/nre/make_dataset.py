import numpy as np
import matplotlib.pyplot as plt
from utils.general import set_seed

def make_nre_datasets(noisy_data: dict, processed: dict,plot: bool =False):
    pnoisy_train = noisy_data["pnoisy_train"]
    pnoisy_val = noisy_data["pnoisy_val"]
    pnoisy_test = noisy_data["pnoisy_test"]

    theta5_train = noisy_data["theta5_train"]
    theta5_val = noisy_data["theta5_val"]
    theta5_test = noisy_data["theta5_test"]
    
    k_train = processed["k_train"]  

    def _make_nre_dataset(pnoisy, theta5, plot=plot):
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
            fig, ax = plt.subplots(1,5,figsize=(25,5))
            idx_k = np.argmin(np.abs(k_train - 0.1))
            pnoisy_slice = pnoisy_nre[:, idx_k] #(2*n_models,1)
            
            # each panel is power vs one of our params
            labels = [
                r"$L_{40}^{\text{X-ray}}$",
                r"$f_{\text{esc}}^{10}$",
                r"$\epsilon$",
                r"$h$",
                r"$f_{\text{noise}}$",
            ]
            for i, label in enumerate(labels):
                theta5_slice = theta5_nre[:,i]
                ax[i].scatter(theta5_slice[:len(pnoisy)], pnoisy_slice[:len(pnoisy)], color = 'blue', label = "Joint Distribution", s=1, alpha=0.3)
                ax[i].scatter(theta5_slice[len(pnoisy):], pnoisy_slice[len(pnoisy):], color = 'red', label = "Disjoint Distribution", s=1, alpha=0.3)
                ax[i].legend()
                ax[i].set_title(f"{label} vs $P_{{noisy}}$ at k = 0.1")
                ax[i].set_xlabel(label)
                ax[i].set_ylabel("$P_{{noisy}}$")
            fig.tight_layout()
            fig.show()
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
    nre_train = _make_nre_dataset(pnoisy_train, theta5_train, plot=plot)
    nre_val = _make_nre_dataset(pnoisy_val, theta5_val, plot=False)
    nre_test = _make_nre_dataset(pnoisy_test, theta5_test, plot=False)

    return {
        "nre_train": nre_train,
        "nre_val": nre_val,
        "nre_test": nre_test,}

    

