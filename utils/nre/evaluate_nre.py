import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn


def evaluate_nre(
    model: nn.Module,
    nre_test: dict,
    scaler: StandardScaler,
    savefig: str = None,
) -> dict:
    """
    Evaluate a trained NRE on the test set.

    Computes BCE test loss, classification report, and plots the confusion matrix.

    Parameters
    ----------
    model : nn.Module
        Trained NRE model in eval() mode.
    nre_test : dict
        Test dataset from make_nre_datasets()['nre_test']. Required keys:
        - 'pnoisy' : ndarray of shape (N, 54), noisy power spectra.
        - 'theta5' : ndarray of shape (N, 5), parameters.
        - 'labels' : ndarray of shape (N,), binary labels.
    scaler : StandardScaler
        Scaler fitted on training inputs, returned by train_nre().
    savefig : str, optional
        If provided, saves the confusion matrix figure to this path. Default None.

    Returns
    -------
    dict
        - 'test_loss'  : float, BCE test loss.
        - 'report'     : dict, classification_report output.
        - 'y_pred'     : ndarray of shape (N,), predicted binary labels.
        - 'logits'     : ndarray of shape (N,), raw model outputs (log r).
    """
    pnoisy_test = np.log(nre_test["pnoisy"])
    theta5_test = nre_test["theta5"]
    x_test_raw  = np.concatenate([pnoisy_test, theta5_test], axis=1)
    x_test      = torch.tensor(scaler.transform(x_test_raw), dtype=torch.float32)
    y_test      = nre_test["labels"].astype(int)

    model.eval()
    with torch.no_grad():
        logits = model(x_test).squeeze(-1).cpu().numpy()

    y_pred = (logits > 0).astype(int)

    y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)
    logits_tensor  = torch.tensor(logits,  dtype=torch.float32).reshape(-1, 1)
    test_loss = BCEWithLogitsLoss()(logits_tensor, y_test_tensor).item()

    report = classification_report(y_test, y_pred, target_names=["Disjoint", "Joint"], output_dict=True)

    print(f"Test loss: {test_loss:.6f}")
    print(classification_report(y_test, y_pred, target_names=["Disjoint", "Joint"]))

    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ConfusionMatrixDisplay(cm, display_labels=["Disjoint", "Joint"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(None)
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
    plt.show()

    return {
        "test_loss": test_loss,
        "report":    report,
        "y_pred":    y_pred,
        "logits":    logits,
    }
