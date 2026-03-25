from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn, optim


# -----------------------------
# Reproducibility / HPC hygiene
# -----------------------------
def set_seed(seed: int = 1701) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data loading + preprocessing
# -----------------------------
def unpack_simulations(simulations: list[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params_list = []
    power_list = []
    k_list = []

    for sim in simulations:
        astro = sim["astro_params"].item()
        cosmo = sim["cosmo_params"].item()
        params_list.append(
            [
                astro["L40_xray"],
                astro["fesc10"],
                astro["epsstar"],
                cosmo["h_fid"],
            ]
        )
        power_list.append(sim["power"])
        k_list.append(sim["k"])

    return np.asarray(params_list), np.asarray(power_list), np.asarray(k_list)


def load_splits(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    num_files = len(files)
    train_files = files[: int(0.8 * num_files)]
    val_files = files[int(0.8 * num_files) : int(0.9 * num_files)]
    test_files = files[int(0.9 * num_files) :]

    def read_many(file_list: list[Path]) -> list[dict]:
        sims = []
        for f in file_list:
            with np.load(f, allow_pickle=True) as d:
                sims.append(dict(d))
        return sims

    train_sims = read_many(train_files)
    val_sims = read_many(val_files)
    test_sims = read_many(test_files)

    raw_params_train, power_train, k_train = unpack_simulations(train_sims)
    raw_params_val, power_val, k_val = unpack_simulations(val_sims)
    raw_params_test, power_test, k_test = unpack_simulations(test_sims)

    return {
        "raw_params_train": raw_params_train,
        "raw_params_val": raw_params_val,
        "raw_params_test": raw_params_test,
        "power_train": power_train,
        "power_val": power_val,
        "power_test": power_test,
        "k_train": k_train,
        "k_val": k_val,
        "k_test": k_test,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
    }


def preprocess(data: dict, n_comp: int) -> dict:
    params_scaler = StandardScaler().fit(data["raw_params_train"])
    params_train = params_scaler.transform(data["raw_params_train"])
    params_val = params_scaler.transform(data["raw_params_val"])
    params_test = params_scaler.transform(data["raw_params_test"])

    pca = PCA(n_components=n_comp)
    pca.fit(data["power_train"])

    # keep the same convention as in your notebook: W columns are PCA directions
    # W = pca.components_.T
    # projected_coeffs_train = np.real(np.dot(data["power_train"], W))
    # projected_coeffs_val = np.real(np.dot(data["power_val"], W))
    # projected_coeffs_test = np.real(np.dot(data["power_test"], W))

    projected_coeffs_train = pca.transform(data["power_train"])
    projected_coeffs_val = pca.transform(data["power_val"])
    projected_coeffs_test = pca.transform(data["power_test"])   

    weight_scaler = StandardScaler().fit(projected_coeffs_train)
    y_train_np = weight_scaler.transform(projected_coeffs_train)
    y_val_np = weight_scaler.transform(projected_coeffs_val)
    y_test_np = weight_scaler.transform(projected_coeffs_test)

    processed = {
        "params_scaler": params_scaler,
        "weight_scaler": weight_scaler,
        "pca": pca,
        "W": W,
        "eig_vals": pca.explained_variance_,
        "params_train": params_train,
        "params_val": params_val,
        "params_test": params_test,
        "projected_coeffs_train": projected_coeffs_train,
        "projected_coeffs_val": projected_coeffs_val,
        "projected_coeffs_test": projected_coeffs_test,
        "y_train_np": y_train_np,
        "y_val_np": y_val_np,
        "y_test_np": y_test_np,
        "x_train": torch.tensor(params_train, dtype=torch.float32),
        "x_val": torch.tensor(params_val, dtype=torch.float32),
        "x_test": torch.tensor(params_test, dtype=torch.float32),
        "y_train": torch.tensor(y_train_np, dtype=torch.float32),
        "y_val": torch.tensor(y_val_np, dtype=torch.float32),
        "y_test": torch.tensor(y_test_np, dtype=torch.float32),
    }
    return processed


# -----------------------------
# Model
# -----------------------------
class Emulator(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 6, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


# -----------------------------
# Training
# -----------------------------
def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    optimiser: optim.Optimizer,
    epochs: int = 1000,
    batch_size: int = 512,
    verbose: bool = True,
    trial: optuna.Trial | None = None,
    patience: int = 100,
    device: str = "cpu",
) -> tuple[float, float | None, int, nn.Module]:
    set_seed(1701)

    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    best_valid_loss = float("inf")
    best_train_loss = None
    best_epoch = -1
    best_state_dict = None
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        perm = torch.randperm(len(x_train), device=device)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for batch in range(0, len(x_shuffled), batch_size):
            x_batch = x_shuffled[batch : batch + batch_size]
            y_batch = y_shuffled[batch : batch + batch_size]

            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(x_batch), y_batch)
            loss.backward()
            optimiser.step()

            total_train_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)

        model.eval()
        with torch.no_grad():
            valid_loss = float(nn.functional.mse_loss(model(x_val), y_val).item())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_train_loss = avg_train_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
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
                f"train={avg_train_loss:.6f} | val={valid_loss:.6f} | lr={lr_now:.2e}",
                flush=True,
            )

        if epochs_since_improvement >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.",
                    flush=True,
                )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_valid_loss, best_train_loss, best_epoch, model


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(
    model: nn.Module,
    processed: dict,
    raw_data: dict,
    device: str,
) -> dict:
    model.eval()
    x_test = processed["x_test"].to(device)
    y_test = processed["y_test"].to(device)

    with torch.no_grad():
        pred_y_test = model(x_test).cpu().numpy()

    test_loss = float(
        nn.functional.mse_loss(torch.tensor(pred_y_test), y_test.cpu()).item()
    )

    pred_weights_test = processed["weight_scaler"].inverse_transform(pred_y_test)
    # test_pred_spectra = np.dot(pred_weights_test, processed["W"].T)
    test_pred_spectra = processed["pca"].inverse_transform(pred_weights_test)
    # mean_test_error = 100.0 * np.mean(
    #     np.abs(raw_data["power_test"] - test_pred_spectra) / np.abs(raw_data["power_test"]),
    #     axis=1,
    # )
    denom = np.maximum(np.abs(raw_data["power_test"]), 1e-8)
    mean_test_error = 100.0 * np.mean(
    np.abs(raw_data["power_test"] - test_pred_spectra) / denom,
    axis=1,
    )

    return {
        "test_loss_normalised_space": test_loss,
        "mean_percentage_error": float(np.mean(mean_test_error)),
        "p95_percentage_error": float(np.quantile(mean_test_error, 0.95)),
        "pred_y_test": pred_y_test,
        "pred_weights_test": pred_weights_test,
        "test_pred_spectra": test_pred_spectra,
        "mean_test_error_per_sample": mean_test_error,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna tuning for the 21-cm emulator on HPC.")
    parser.add_argument("--data-dir", type=Path, default=Path("simulations"), help="Directory containing .npz simulation files")
    parser.add_argument("--output-dir", type=Path, default=Path("optuna_outputs"), help="Where to save the study, model, and preprocessing artifacts")
    parser.add_argument("--study-name", type=str, default="emulator_optuna")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///optuna_outputs/emulator_optuna.db")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds")
    parser.add_argument("--n-comp", type=int, default=6, help="Number of PCA components")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You passed --device cuda but CUDA is not available.")

    device = "cuda" if args.device == "cuda" else "cpu"
    print(f"Using device: {device}", flush=True)

    raw_data = load_splits(args.data_dir)
    processed = preprocess(raw_data, n_comp=args.n_comp)

    def objective(trial: optuna.Trial) -> float:
        set_seed(args.seed)

        num_layers = trial.suggest_categorical("num_layers", [3, 4, 5, 6, 7, 8])
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        model = Emulator(
            input_dim=processed["x_train"].shape[1],
            output_dim=args.n_comp,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        optimiser = optim.Adam(model.parameters(), lr=lr)

        best_valid_loss, best_train_loss, best_epoch, _ = train_model(
            model=model,
            x_train=processed["x_train"],
            y_train=processed["y_train"],
            x_val=processed["x_val"],
            y_val=processed["y_val"],
            optimiser=optimiser,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=False,
            trial=trial,
            patience=args.patience,
            device=device,
        )

        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("best_train_loss", None if best_train_loss is None else float(best_train_loss))
        return float(best_valid_loss)

    if args.storage is None:
        storage = f"sqlite:///{(args.output_dir / f'{args.study_name}.db').resolve()}"
    else:
        storage = args.storage

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Starting Optuna study '{args.study_name}'", flush=True)
    print(f"Storage: {storage}", flush=True)
    print(f"Data dir: {args.data_dir.resolve()}", flush=True)
    print(f"Output dir: {args.output_dir.resolve()}", flush=True)
    print(f"Trials to run now: {args.n_trials}", flush=True)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    print("Best trial:", flush=True)
    print(f"  value: {study.best_trial.value}", flush=True)
    print(f"  params: {study.best_trial.params}", flush=True)
    print(f"  best_epoch: {study.best_trial.user_attrs.get('best_epoch')}", flush=True)
    print(f"  best_train_loss: {study.best_trial.user_attrs.get('best_train_loss')}", flush=True)

    best_params = study.best_trial.params
    best_model = Emulator(
        input_dim=processed["x_train"].shape[1],
        output_dim=args.n_comp,
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
    )
    best_optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])

    best_valid_loss, best_train_loss, best_epoch, best_model = train_model(
        model=best_model,
        x_train=processed["x_train"],
        y_train=processed["y_train"],
        x_val=processed["x_val"],
        y_val=processed["y_val"],
        optimiser=best_optimizer,
        epochs=max(args.epochs, 10000),
        batch_size=args.batch_size,
        verbose=True,
        trial=None,
        patience=max(args.patience, 1000),
        device=device,
    )

    metrics = evaluate_model(best_model.to(device), processed, raw_data, device=device)
    print(f"Test loss (normalised PCA-weight space): {metrics['test_loss_normalised_space']:.6f}", flush=True)
    print(f"Mean percentage error: {metrics['mean_percentage_error']:.3f}%", flush=True)
    print(f"95th percentile error: {metrics['p95_percentage_error']:.3f}%", flush=True)
    best_model = best_model.to("cpu")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "best_params": best_params,
            "n_comp": args.n_comp,
            "input_dim": processed["x_train"].shape[1],
            "output_dim": args.n_comp,
            "best_valid_loss": best_valid_loss,
            "best_train_loss": best_train_loss,
            "best_epoch": best_epoch,
        },
        args.output_dir / "best_model.pt",
    )

    with open(args.output_dir / "preprocessing.pkl", "wb") as f:
        pickle.dump(
            {
                "params_scaler": processed["params_scaler"],
                "weight_scaler": processed["weight_scaler"],
                "pca": processed["pca"],
                "W": processed["W"],
                "eig_vals": processed["eig_vals"],
            },
            f,
        )

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "study_name": args.study_name,
                "storage": storage,
                "n_trials_total": len(study.trials),
                "best_trial_value": study.best_trial.value,
                "best_params": best_params,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "best_train_loss": best_train_loss,
                "test_loss_normalised_space": metrics["test_loss_normalised_space"],
                "mean_percentage_error": metrics["mean_percentage_error"],
                "p95_percentage_error": metrics["p95_percentage_error"],
                "seed": args.seed,
                "data_dir": str(args.data_dir),
                "train_size": int(len(raw_data["train_files"])),
                "val_size": int(len(raw_data["val_files"])),
                "test_size": int(len(raw_data["test_files"])),
            },
            f,
            indent=2,
        )

    print(f"Saved model to {(args.output_dir / 'best_model.pt').resolve()}", flush=True)
    print(f"Saved preprocessing to {(args.output_dir / 'preprocessing.pkl').resolve()}", flush=True)
    print(f"Saved summary to {(args.output_dir / 'summary.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
