import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import reshape
from sklearn.preprocessing import StandardScaler
from utils.general import set_seed

def train_nre(
    model: nn.Module,
    nre_trainset: dict,
    nre_valset: dict,
    epochs: int = 1000,
    batch_size: int = 1024,
    device: str = 'cpu',
    verbose: bool = True,
    plot: bool = True,
    patience: int = 250,
) -> tuple[float, float, int, nn.Module, StandardScaler]:
    """
    Train a Neural Ratio Estimator using binary cross-entropy loss.

    Constructs input features by concatenating log power spectra and parameters,
    fits a StandardScaler on the training set, and trains the model with Adam
    and early stopping. Returns the best model state (lowest validation loss).

    Parameters
    ----------
    model : nn.Module
        NRE model to train. Should output log r (linear final activation).
    nre_trainset : dict
        Training set from make_nre_dataset(). Required keys:
        - 'pnoisy' : ndarray of shape (n_train, 54), noisy power spectra.
        - 'theta5' : ndarray of shape (n_train, 5), parameters.
        - 'labels' : ndarray of shape (n_train,), 1 for joint, 0 for disjoint.
    nre_valset : dict
        Validation set from make_nre_dataset(). Same keys as nre_trainset.
    epochs : int, optional
        Maximum number of training epochs. Default 1000.
    batch_size : int, optional
        Mini-batch size. Default 1024.
    device : str, optional
        Torch device to train on, e.g. 'cpu' or 'cuda'. Default 'cpu'.
    verbose : bool, optional
        If True, prints loss every 100 epochs and on early stopping. Default True.
    plot : bool, optional
        If True, plots training and validation loss curves after training. Default True.
    patience : int, optional
        Number of epochs without validation improvement before early stopping. Default 250.

    Returns
    -------
    best_valid_loss : float
        Best validation loss achieved during training.
    best_train_loss : float
        Training loss at the epoch of best validation loss.
    best_epoch : int
        Epoch index (0-based) at which best validation loss was achieved.
    model : nn.Module
        Model loaded with the best weights.
    x_scaler : StandardScaler
        Scaler fitted on training inputs, required for inference.

    Notes
    -----
    Input features are constructed as [log(pnoisy), theta5], shape (n_samples, 59).
    The scaler is fitted on training data only to prevent data leakage.
    Early stopping restores the best model state before returning.
    """
    pnoisy = np.log(nre_trainset['pnoisy'])
    theta5 = nre_trainset['theta5']
    x_train = torch.tensor(np.concatenate([pnoisy, theta5], axis = 1), dtype=torch.float32) # (n_samples, 59)
    y_train = torch.tensor(nre_trainset['labels'], dtype=torch.float32).reshape(-1, 1) #the 1 and 0 indicating if they are join or disjoint (n_samples,)
    x_scaler = StandardScaler()  # used to scale the inputs (powervk and theta5)
    
    x_scaler.fit(x_train)
    x_train = torch.tensor(x_scaler.transform(x_train), dtype=torch.float32)


    pnoisy = np.log(nre_valset['pnoisy'])
    theta5 = nre_valset['theta5']
    x_val = torch.tensor(np.concatenate([pnoisy, theta5], axis = 1), dtype=torch.float32 )# (n_samples_val, 59)
    y_val = torch.tensor(nre_valset['labels'], dtype=torch.float32).reshape(-1, 1)

    x_val = torch.tensor(x_scaler.transform(x_val), dtype=torch.float32)

    set_seed(1701)
    model = model.to(device)

    # use the BCEWithLogitsLoss from pytorch
    BCE_loss = BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-5)

    best_valid_loss        = float("inf")
    best_train_loss        = None
    best_epoch             = -1
    best_state_dict        = None
    epochs_since_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        # shuffling
        perm = torch.randperm(len(x_train), device=device)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        # loop through batches each epoch
        for start in range(0, len(x_shuffled), batch_size): 
            x_batch = x_shuffled[start : start + batch_size]
            y_batch = y_shuffled[start : start + batch_size]

            optimiser.zero_grad()
            loss = BCE_loss(model(x_batch), y_batch)
            loss.backward()
            optimiser.step()

            total_train_loss += float(loss.item())
            num_batches += 1
        avg_train_loss = total_train_loss / max(num_batches, 1) # across this epoch
        train_losses.append(avg_train_loss)
        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = float(BCE_loss(model(x_val), y_val).item())
            val_losses.append(valid_loss)

        # Bookkeeping
        if valid_loss < best_valid_loss:
                best_valid_loss          = valid_loss
                best_train_loss          = avg_train_loss
                best_epoch               = epoch
                best_state_dict          = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                lr_now = optimiser.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1}/{epochs} | ",
                    f"train={avg_train_loss:.6f} | val={valid_loss:.6f} ",
                    flush=True,
                )
        
        if epochs_since_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch + 1}.",
                        flush=True,
                    )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if plot == True:
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    return best_valid_loss, best_train_loss, best_epoch, model, x_scaler


