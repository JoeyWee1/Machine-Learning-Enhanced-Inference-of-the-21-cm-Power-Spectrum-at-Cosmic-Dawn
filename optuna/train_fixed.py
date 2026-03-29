# train_fixed.py
"""
Train and evaluate the emulator with externally specified hyperparameters.

Usage examples:
    python train_fixed.py --num-layers 6 --hidden-dim 512 --lr 8.24e-4 --weight-decay 2.43e-4
    python train_fixed.py --from-study   # loads best params from existing Optuna study
    python train_fixed.py --params-json '{"num_layers": 6, "hidden_dim": 512, "lr": 0.000824, "weight_decay": 0.000243}'
"""

import argparse
import json
import pickle
from pathlib import Path

import optuna
import torch
from torch import optim

from optuna.optuna_optimize_emulator import (
    Emulator,
    evaluate_model,
    load_splits,
    preprocess,
    set_seed,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data / output
    parser.add_argument("--data-dir", type=Path, default=Path("simulations"))
    parser.add_argument("--output-dir", type=Path, default=Path("optuna_outputs"))
    parser.add_argument("--n-comp", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # Three ways to supply hyperparameters (pick one)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help='JSON string, e.g. \'{"num_layers":6,"hidden_dim":512,"lr":8e-4,"weight_decay":2e-4}\'',
    )
    parser.add_argument(
        "--from-study",
        action="store_true",
        help="Load best hyperparameters from an existing Optuna study DB",
    )
    parser.add_argument("--study-name", type=str, default="emulator_optuna")
    parser.add_argument("--storage", type=str, default=None)

    return parser.parse_args()


def resolve_params(args) -> dict:
    """Return a dict with num_layers, hidden_dim, lr, weight_decay."""

    if args.from_study:
        storage = args.storage or f"sqlite:///{(args.output_dir / f'{args.study_name}.db').resolve()}"
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        params = study.best_trial.params
        print(f"Loaded best params from study '{args.study_name}':")
        print(f"  {params}")
        return params

    if args.params_json is not None:
        params = json.loads(args.params_json)
        print(f"Loaded params from JSON: {params}")
        return params

    # Fall back to individual CLI flags
    required = {"num_layers": args.num_layers, "hidden_dim": args.hidden_dim,
                 "lr": args.lr, "weight_decay": args.weight_decay}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing hyperparameters: {missing}. "
            "Provide them via --num-layers/--hidden-dim/--lr/--weight-decay, "
            "--params-json, or --from-study."
        )
    return {
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    raw_data = load_splits(args.data_dir)
    processed = preprocess(raw_data, n_comp=args.n_comp)
    for key in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        processed[key] = processed[key].to(device)

    # ── Hyperparameters ───────────────────────────────────────────────────
    params = resolve_params(args)
    print(f"\nTraining with: {params}\n")

    # ── Build model ───────────────────────────────────────────────────────
    model = Emulator(
        input_dim=processed["x_train"].shape[1],
        output_dim=args.n_comp,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimiser = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    best_valid_loss, best_train_loss, best_epoch, model = train_model(
        model=model,
        x_train=processed["x_train"],
        y_train=processed["y_train"],
        x_val=processed["x_val"],
        y_val=processed["y_val"],
        optimiser=optimiser,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        trial=None,       # no Optuna pruning
        patience=args.patience,
        device=device,
    )

    print(f"\nBest val loss: {best_valid_loss:.6f}  (epoch {best_epoch + 1})")

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate_model(model.to(device), processed, raw_data, device=device)
    print(f"Test loss (normalised space): {metrics['test_loss_normalised_space']:.6f}")
    print(f"Mean percentage error:        {metrics['mean_percentage_error']:.3f}%")
    print(f"95th percentile error:        {metrics['p95_percentage_error']:.3f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    tag = f"fixed_nl{params['num_layers']}_hd{params['hidden_dim']}"
    model_path = args.output_dir / f"model_{tag}.pt"
    preprocess_path = args.output_dir / f"preprocessing_{tag}.pkl"
    summary_path = args.output_dir / f"summary_{tag}.json"

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "params": params,
            "n_comp": args.n_comp,
            "input_dim": processed["x_train"].shape[1],
            "best_valid_loss": best_valid_loss,
            "best_train_loss": best_train_loss,
            "best_epoch": best_epoch,
        },
        model_path,
    )

    with open(preprocess_path, "wb") as f:
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

    with open(summary_path, "w") as f:
        json.dump(
            {
                "params": params,
                "best_valid_loss": best_valid_loss,
                "best_train_loss": best_train_loss,
                "best_epoch": best_epoch,
                **{k: v for k, v in metrics.items()
                   if not hasattr(v, "__len__")},  # skip arrays
            },
            f,
            indent=2,
        )

    print(f"\nSaved model to        {model_path.resolve()}")
    print(f"Saved preprocessing to {preprocess_path.resolve()}")
    print(f"Saved summary to       {summary_path.resolve()}")


if __name__ == "__main__":
    main()