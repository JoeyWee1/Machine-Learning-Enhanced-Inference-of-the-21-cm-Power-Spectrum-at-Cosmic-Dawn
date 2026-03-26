import copy
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from helpers.set_seed import set_seed

def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    optimiser: optim.Optimizer,
    processed: dict,
    epochs: int = 1000,
    batch_size: int = 512,
    loss_mode: str = "pca",
    verbose: bool = True,
    trial: optuna.Trial | None = None,
    patience: int = 100,
    device: str = "cpu",
) -> tuple[float, float | None, int, nn.Module]:
    """
    Trains a neural network emulator with early stopping and optional Optuna pruning.

    Loss can be computed either in PCA-coefficient space ("pca") or in reconstructed
    power-spectrum space ("reconstruction"), controlled by `loss_mode`.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    x_train : torch.Tensor of shape (N_train, input_dim)
        Standardized training parameters.
    y_train : torch.Tensor of shape (N_train, n_comp)
        Standardized PCA coefficients for training.
    x_val : torch.Tensor of shape (N_val, input_dim)
        Standardized validation parameters.
    y_val : torch.Tensor of shape (N_val, n_comp)
        Standardized PCA coefficients for validation.
    optimiser : optim.Optimizer
        Configured PyTorch optimiser (e.g. Adam).
    processed : dict
        Preprocessing artefacts from `preprocess()`. Required keys:
        - "weight_scaler" : StandardScaler
            Used to inverse-transform PCA coefficients back to unscaled space.
        - "pca" : PCA
            Used to reconstruct power spectra from unscaled PCA coefficients.
        Only used when loss_mode="reconstruction".
    epochs : int, optional
        Maximum number of training epochs. Default: 1000.
    batch_size : int, optional
        Mini-batch size. Default: 512.
    loss_mode : str, optional
        Which space to compute the MSE loss in:
        - "pca"            : MSE on standardized PCA coefficients (fast).
        - "reconstruction" : MSE on reconstructed power spectra (physically meaningful).
        Default: "pca".
    verbose : bool, optional
        If True, prints loss every 100 epochs and on early stopping. Default: True.
    trial : optuna.Trial or None, optional
        If provided, reports validation loss for Optuna pruning. Default: None.
    patience : int, optional
        Number of epochs without improvement before early stopping. Default: 100.
    device : str, optional
        Device to run training on ("cpu" or "cuda"). Default: "cpu".

    Returns
    -------
    best_valid_loss : float
        Best validation loss achieved (in the space selected by `loss_mode`).
    best_train_loss : float or None
        Training loss at the epoch of best validation loss.
    best_epoch : int
        Zero-indexed epoch at which the best validation loss occurred.
    model : nn.Module
        Model loaded with the best weights found during training.

    Raises
    ------
    ValueError
        If `loss_mode` is not "pca" or "reconstruction".
    optuna.TrialPruned
        If Optuna determines the trial should be pruned.
    """
    if loss_mode not in ("pca", "reconstruction"):
        raise ValueError(f"loss_mode must be 'pca' or 'reconstruction', got '{loss_mode}'")

    set_seed(1701)
    model = model.to(device)

    # Pre-convert reconstruction artefacts to tensors once (only if needed)
    if loss_mode == "reconstruction":
        weight_scaler = processed["weight_scaler"]
        pca = processed["pca"]

        pca_mean  = torch.tensor(weight_scaler.mean_,       dtype=torch.float32, device=device)
        pca_scale = torch.tensor(weight_scaler.scale_,      dtype=torch.float32, device=device)
        W         = torch.tensor(pca.components_.T,         dtype=torch.float32, device=device)
        ps_mean   = torch.tensor(pca.mean_,                 dtype=torch.float32, device=device)

    def compute_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MSE in PCA space, or MSE in reconstructed power-spectrum space."""
        if loss_mode == "pca":
            return nn.functional.mse_loss(y_pred, y_true)

        # Inverse-standardize → unscaled PCA coefficients → power spectra
        coeffs_pred = y_pred * pca_scale + pca_mean
        coeffs_true = y_true * pca_scale + pca_mean
        ps_pred = coeffs_pred @ W.T + ps_mean
        ps_true = coeffs_true @ W.T + ps_mean
        return nn.functional.mse_loss(ps_pred, ps_true)

    best_valid_loss        = float("inf")
    best_train_loss        = None
    best_epoch             = -1
    best_state_dict        = None
    epochs_since_improvement = 0

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        num_batches      = 0

        perm       = torch.randperm(len(x_train), device=device)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for start in range(0, len(x_shuffled), batch_size):
            x_batch = x_shuffled[start : start + batch_size]
            y_batch = y_shuffled[start : start + batch_size]

            optimiser.zero_grad()
            loss = compute_loss(model(x_batch), y_batch)
            loss.backward()
            optimiser.step()

            total_train_loss += float(loss.item())
            num_batches      += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            valid_loss = float(compute_loss(model(x_val), y_val).item())

        # ── Bookkeeping ───────────────────────────────────────────────────────
        if valid_loss < best_valid_loss:
            best_valid_loss          = valid_loss
            best_train_loss          = avg_train_loss
            best_epoch               = epoch
            best_state_dict          = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if trial is not None:
            trial.report(valid_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            lr_now = optimiser.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train={avg_train_loss:.6f} | val={valid_loss:.6f} | "
                f"lr={lr_now:.2e} | mode={loss_mode}",
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

    return best_valid_loss, best_train_loss, best_epoch, model