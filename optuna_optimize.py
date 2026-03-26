from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from torch import optim

from helpers.set_seed import set_seed
from helpers.load_files import load_splits
from helpers.preprocess import preprocess
from helpers.emulator import Emulator
from helpers.train_model import train_model
from helpers.evaluate_model import evaluate_model

try:
    import optuna
except ImportError as e:
    raise ImportError("optuna is required: pip install optuna") from e


def main() -> None:
    """
    Entry point for Optuna hyperparameter search over the 21-cm power spectrum emulator.

    Parses command-line arguments, loads and preprocesses simulation data, runs an
    Optuna TPE study to minimise validation loss, retrains the best model to convergence,
    evaluates it on the test set, and saves the model weights, preprocessing artefacts,
    and a JSON summary to disk.

    Command-line Arguments
    ----------------------
    --data-dir : Path
        Directory containing .npz simulation files. Default: "simulations".
    --output-dir : Path
        Directory for saving outputs. Default: "optuna_outputs".
    --study-name : str
        Optuna study name. Default: "emulator_optuna".
    --storage : str or None
        Optuna storage URL (e.g. sqlite:///outputs/study.db). If None, a default
        SQLite path is constructed under --output-dir. A 60-second timeout is appended
        automatically to reduce database-lock errors.
    --n-trials : int
        Number of Optuna trials to run. Default: 50.
    --timeout : int or None
        Optional wall-clock timeout for study.optimize(), in seconds.
    --n-comp : int
        Number of PCA components. Default: 6.
    --epochs : int
        Maximum epochs per trial. Default: 1000.
    --batch-size : int
        Mini-batch size. Default: 512.
    --patience : int
        Early-stopping patience (epochs). Default: 100.
    --seed : int
        Global random seed. Default: 1701.
    --device : {"cpu", "cuda"}
        Compute device. Default: "cuda".
    --log-power : flag
        If set, applies log transform to power spectra before PCA. Requires all
        power values to be strictly positive.
    --loss-mode : {"pca", "reconstruction"}
        Loss space for training:
        - "pca"            : MSE on standardised PCA coefficients (fast, default).
        - "reconstruction" : MSE on reconstructed power spectra (physically meaningful).

    Notes
    -----
    The SQLite storage URL automatically has ``?timeout=60`` appended to reduce
    ``database is locked`` errors during long sequential runs.

    Intermediate values are reported to Optuna every 10 epochs (not every epoch)
    to further reduce SQLite write pressure.
    """
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the 21-cm power spectrum emulator."
    )
    parser.add_argument("--data-dir",   type=Path, default=Path("simulations"),
                        help="Directory containing .npz simulation files.")
    parser.add_argument("--output-dir", type=Path, default=Path("optuna_outputs"),
                        help="Where to save the study, model, and preprocessing artefacts.")
    parser.add_argument("--study-name", type=str,  default="emulator_optuna")
    parser.add_argument("--storage",    type=str,  default=None,
                        help="Optuna storage URL, e.g. sqlite:///optuna_outputs/study.db")
    parser.add_argument("--n-trials",   type=int,  default=50)
    parser.add_argument("--timeout",    type=int,  default=None,
                        help="Optional Optuna timeout in seconds.")
    parser.add_argument("--n-comp",     type=int,  default=6,
                        help="Number of PCA components.")
    parser.add_argument("--epochs",     type=int,  default=1000)
    parser.add_argument("--batch-size", type=int,  default=512)
    parser.add_argument("--patience",   type=int,  default=100)
    parser.add_argument("--seed",       type=int,  default=1701)
    parser.add_argument("--device",     type=str,  default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--log-power",  action="store_true",
                        help="Apply log transform to power spectra before PCA.")
    parser.add_argument("--loss-mode",  type=str,  default="pca",
                        choices=["pca", "reconstruction"],
                        help=(
                            "pca: MSE on normalised PCA weights (default, fast). "
                            "reconstruction: MSE on reconstructed power spectra."
                        ))
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")

    device = "cuda" if args.device == "cuda" else "cpu"
    print(f"Using device:  {device}",       flush=True)
    print(f"Loss mode:     {args.loss_mode}", flush=True)
    print(f"Log power:     {args.log_power}", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    raw_data  = load_splits(args.data_dir)
    processed = preprocess(raw_data, n_comp=args.n_comp, log_power=args.log_power)

    for key in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        processed[key] = processed[key].to(device)

    # ── Optuna objective ───────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        set_seed(args.seed)

        num_layers   = trial.suggest_categorical("num_layers",  [3, 4, 5, 6, 7, 8, 9, 10])
        hidden_dim   = trial.suggest_categorical("hidden_dim",  [32, 64, 128, 256, 512])
        lr           = trial.suggest_float("lr",           1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        model = Emulator(
            input_dim=processed["x_train"].shape[1],
            output_dim=args.n_comp,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_valid_loss, best_train_loss, best_epoch, _ = train_model(
            model=model,
            x_train=processed["x_train"],
            y_train=processed["y_train"],
            x_val=processed["x_val"],
            y_val=processed["y_val"],
            optimiser=optimiser,
            processed=processed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_mode=args.loss_mode,
            verbose=False,
            trial=trial,
            patience=args.patience,
            device=device,
        )

        trial.set_user_attr("best_epoch",      int(best_epoch))
        trial.set_user_attr("best_train_loss",
                            None if best_train_loss is None else float(best_train_loss))
        return float(best_valid_loss)

    # ── Storage ────────────────────────────────────────────────────────────────
    if args.storage is None:
        db_path = (args.output_dir / f"{args.study_name}.db").resolve()
        storage = f"sqlite:///{db_path}?timeout=60"
    else:
        # Append timeout if not already present
        storage = args.storage if "timeout=" in args.storage else args.storage + "?timeout=60"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    study   = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Starting Optuna study '{args.study_name}'", flush=True)
    print(f"Storage:    {storage}",                    flush=True)
    print(f"Data dir:   {args.data_dir.resolve()}",    flush=True)
    print(f"Output dir: {args.output_dir.resolve()}",  flush=True)
    print(f"Trials:     {args.n_trials}",              flush=True)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    # ── Report best trial ──────────────────────────────────────────────────────
    print("Best trial:",                                                       flush=True)
    print(f"  value:      {study.best_trial.value}",                          flush=True)
    print(f"  params:     {study.best_trial.params}",                         flush=True)
    print(f"  best_epoch: {study.best_trial.user_attrs.get('best_epoch')}",   flush=True)
    print(f"  train_loss: {study.best_trial.user_attrs.get('best_train_loss')}", flush=True)

    # ── Retrain best model to full convergence ────────────────────────────────
    best_params = study.best_trial.params
    best_model  = Emulator(
        input_dim=processed["x_train"].shape[1],
        output_dim=args.n_comp,
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
    )
    best_optimizer = optim.Adam(
        best_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    best_valid_loss, best_train_loss, best_epoch, best_model = train_model(
        model=best_model,
        x_train=processed["x_train"],
        y_train=processed["y_train"],
        x_val=processed["x_val"],
        y_val=processed["y_val"],
        optimiser=best_optimizer,
        processed=processed,
        epochs=max(args.epochs, 10000),
        batch_size=args.batch_size,
        loss_mode=args.loss_mode,
        verbose=True,
        trial=None,
        patience=max(args.patience, 1000),
        device=device,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate_model(best_model.to(device), processed, raw_data, device=device)

    print(f"Test loss (normalised space): {metrics['test_loss_normalised_space']:.6f}", flush=True)
    print(f"Mean percentage error:        {metrics['mean_percentage_error']:.3f}%",     flush=True)
    print(f"95th percentile error:        {metrics['p95_percentage_error']:.3f}%",      flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    n_trials_done = len(study.trials)
    mean_err      = f"{metrics['mean_percentage_error']:.2f}pct"
    loss_tag      = args.loss_mode
    log_tag       = "_log" if args.log_power else ""
    run_tag       = f"run{n_trials_done}_{loss_tag}{log_tag}_{mean_err}"

    model_path      = args.output_dir / f"best_model_{run_tag}.pt"
    preprocess_path = args.output_dir / f"preprocessing_{run_tag}.pkl"
    summary_path    = args.output_dir / f"summary_{run_tag}.json"

    best_model = best_model.to("cpu")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "best_params":      best_params,
            "n_comp":           args.n_comp,
            "input_dim":        processed["x_train"].shape[1],
            "output_dim":       args.n_comp,
            "loss_mode":        args.loss_mode,
            "log_power":        args.log_power,
            "best_valid_loss":  best_valid_loss,
            "best_train_loss":  best_train_loss,
            "best_epoch":       best_epoch,
        },
        model_path,
    )

    with open(preprocess_path, "wb") as f:
        pickle.dump(
            {
                "params_scaler": processed["params_scaler"],
                "weight_scaler": processed["weight_scaler"],
                "pca":           processed["pca"],
                "W":             processed["W"],
                "eig_vals":      processed["eig_vals"],
                "log_power":     processed["log_power"],
            },
            f,
        )

    with open(summary_path, "w") as f:
        json.dump(
            {
                "study_name":                 args.study_name,
                "storage":                    storage,
                "loss_mode":                  args.loss_mode,
                "log_power":                  args.log_power,
                "n_trials_total":             len(study.trials),
                "best_trial_value":           study.best_trial.value,
                "best_params":                best_params,
                "best_epoch":                 best_epoch,
                "best_valid_loss":            best_valid_loss,
                "best_train_loss":            best_train_loss,
                "test_loss_normalised_space": metrics["test_loss_normalised_space"],
                "mean_percentage_error":      metrics["mean_percentage_error"],
                "p95_percentage_error":       metrics["p95_percentage_error"],
                "seed":                       args.seed,
                "data_dir":                   str(args.data_dir),
                "train_size":                 int(len(raw_data["train_files"])),
                "val_size":                   int(len(raw_data["val_files"])),
                "test_size":                  int(len(raw_data["test_files"])),
            },
            f,
            indent=2,
        )

    print(f"Saved model to         {model_path.resolve()}",      flush=True)
    print(f"Saved preprocessing to {preprocess_path.resolve()}", flush=True)
    print(f"Saved summary to       {summary_path.resolve()}",    flush=True)


if __name__ == "__main__":
    main()