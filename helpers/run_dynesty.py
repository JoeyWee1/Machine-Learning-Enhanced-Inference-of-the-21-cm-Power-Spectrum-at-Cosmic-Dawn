#!/usr/bin/env python
"""
run_dynesty.py — Nested sampling for 21-cm power spectrum inference using dynesty.

Compared to emcee, dynesty:
  - Also estimates the Bayesian log-evidence (log Z) for model comparison
  - Handles multi-modal and degenerate posteriors more robustly
  - Does not require burn-in or thinning — all samples are weighted draws
    from the posterior

Usage (command line)
--------------------
    python run_dynesty.py \\
        --model-path      optuna_outputs/model_fixed_nl4_hd512.pt \\
        --preprocess-path optuna_outputs/preprocessing_fixed_nl4_hd512.pkl \\
        --obs-index       0 \\
        --data-dir        simulations/ \\
        --nlive           500 \\
        --output-dir      dynesty_outputs/

Usage (notebook)
----------------
    from run_dynesty import build_sampler, plot_corner
    results = build_sampler(model, processed, p_obs, domains, nlive=500)
    plot_corner(results)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import corner
import dynesty
import matplotlib.pyplot as plt
import numpy as np
import torch
from dynesty.utils import resample_equal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers.emulator import Emulator
from helpers.load_files import load_splits
from helpers.sampling import ln_likelihood


# ── Prior transform ───────────────────────────────────────────────────────────

def make_prior_transform(domains: dict):
    """
    Return a function mapping u ~ Uniform[0,1]^5 to physical parameters
    under log-uniform priors (matching helpers/sampling.py).

    Parameter order: [L40, fesc10, eps, h, fnoise]

    Prior support is widened to [0.5*lo, 1.5*hi] per domain key,
    matching _build_priors() in sampling.py.
    """
    bounds = [
        (0.5 * domains['L40_xray'][0], 1.5 * domains['L40_xray'][1]),
        (0.5 * domains['fesc10'][0],   1.5 * domains['fesc10'][1]),
        (0.5 * domains['epsilon'][0],  1.5 * domains['epsilon'][1]),
        (0.5 * domains['h'][0],        1.5 * domains['h'][1]),
        (1e-3, 1e0),   # fnoise
    ]
    log_lo = np.array([np.log(lo) for lo, _ in bounds])
    log_hi = np.array([np.log(hi) for _, hi in bounds])

    def prior_transform(u):
        return np.exp(log_lo + u * (log_hi - log_lo))

    return prior_transform


# ── Log-likelihood ────────────────────────────────────────────────────────────

def make_log_likelihood(model, p_obs: np.ndarray, processed: dict):
    """
    Return a log-likelihood function compatible with dynesty.

    dynesty evaluates one point at a time; the NN forward pass is a
    single vector call per evaluation.
    """
    def log_likelihood(theta):
        L40, fesc10, eps, h, fnoise = theta

        with torch.no_grad():
            # Model input order: [L40, fesc10, eps, h]
            params_tensor = torch.tensor(
                [L40, fesc10, eps, h], dtype=torch.float32
            )
            pred_scaled = model(params_tensor).cpu().numpy().reshape(1, -1)

        pred_raw = processed["weight_scaler"].inverse_transform(pred_scaled)
        p_model  = processed["pca"].inverse_transform(pred_raw)[0]
        if processed.get("log_power", False):
            p_model = np.exp(p_model)

        return ln_likelihood(fnoise, p_model, p_obs)

    return log_likelihood


# ── Sampler ───────────────────────────────────────────────────────────────────

def build_sampler(
    model,
    processed: dict,
    p_obs: np.ndarray,
    domains: dict,
    nlive: int = 500,
    print_progress: bool = True,
):
    """
    Run dynesty DynamicNestedSampler and return the results object.

    Parameters
    ----------
    model : nn.Module
        Trained emulator in eval() mode.
    processed : dict
        Preprocessing artefacts (weight_scaler, pca, log_power).
    p_obs : np.ndarray of shape (54,)
        Observed power spectrum.
    domains : dict
        Feature domains with keys 'epsilon', 'L40_xray', 'fesc10', 'h'.
    nlive : int
        Number of live points. Higher = more accurate but slower.
        500 is a reasonable default; 1000 for publication quality.
    print_progress : bool
        Whether to print dynesty progress.

    Returns
    -------
    dynesty.results.Results
        Contains samples (physical space), weights, and log-evidence.
    """
    model.eval()

    prior_transform = make_prior_transform(domains)
    log_likelihood  = make_log_likelihood(model, p_obs, processed)

    sampler = dynesty.DynamicNestedSampler(
        log_likelihood,
        prior_transform,
        ndim=5,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
    )
    sampler.run_nested(print_progress=print_progress)
    return sampler.results


# ── Corner plot ───────────────────────────────────────────────────────────────

def plot_corner(results, truths=None, output_path=None):
    """
    Plot and optionally save a corner plot from dynesty results.

    Parameters
    ----------
    results : dynesty.results.Results
    truths : list of float or None
        True parameter values to mark, order [L40, fesc10, eps, h, fnoise].
    output_path : Path or str or None
        If given, save the figure there.
    """
    labels  = ["L40_xray", "fesc10", "epsilon", "h", "fnoise"]
    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)   # (n_samples, 5)

    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        truths=truths,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved corner plot to {output_path}")

    plt.show()
    return fig, samples


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Nested sampling for 21-cm power spectrum inference."
    )
    p.add_argument("--model-path",      type=Path, required=True)
    p.add_argument("--preprocess-path", type=Path, required=True)

    obs = p.add_mutually_exclusive_group(required=True)
    obs.add_argument("--p-obs",     type=Path, default=None,
                     help="Path to .npy observed power spectrum.")
    obs.add_argument("--obs-index", type=int,  default=None,
                     help="Index into the test set.")

    p.add_argument("--data-dir",     type=Path, default=Path("simulations"))
    p.add_argument("--output-dir",   type=Path, default=Path("dynesty_outputs"))
    p.add_argument("--nlive",        type=int,  default=500)
    p.add_argument("--run-name",     type=str,  default=None)
    p.add_argument("--domains-json", type=str,  default=None)
    return p.parse_args()


def _load_model(model_path, device):
    ckpt  = torch.load(model_path, map_location="cpu")
    model = Emulator(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["n_comp"],
        hidden_dim=ckpt["params"]["hidden_dim"],
        num_layers=ckpt["params"]["num_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def _load_preprocessing(preprocess_path):
    with open(preprocess_path, "rb") as f:
        prep = pickle.load(f)
    prep.setdefault("log_power", False)
    return prep


def _compute_domains(raw_params):
    return {
        "L40_xray": [float(raw_params[:, 0].min()), float(raw_params[:, 0].max())],
        "fesc10":   [float(raw_params[:, 1].min()), float(raw_params[:, 1].max())],
        "epsilon":  [float(raw_params[:, 2].min()), float(raw_params[:, 2].max())],
        "h":        [float(raw_params[:, 3].min()), float(raw_params[:, 3].max())],
    }


def main():
    args     = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.model_path.stem

    model     = _load_model(args.model_path, "cpu")
    processed = _load_preprocessing(args.preprocess_path)

    raw_data = None
    if args.domains_json is None or args.obs_index is not None:
        raw_data = load_splits(args.data_dir)

    if args.domains_json is not None:
        domains = json.loads(args.domains_json)
    else:
        all_params = np.concatenate([
            raw_data["raw_params_train"],
            raw_data["raw_params_val"],
            raw_data["raw_params_test"],
        ])
        domains = _compute_domains(all_params)

    truths = None
    if args.p_obs is not None:
        p_obs = np.load(args.p_obs)
    else:
        p_obs        = raw_data["power_test"][args.obs_index]
        true_params  = raw_data["raw_params_test"][args.obs_index]  # [L40, fesc10, eps, h]
        truths       = [true_params[0], true_params[1], true_params[2], true_params[3], None]
        print(f"True params [L40, fesc10, eps, h]: {truths[:4]}")

    print(f"\nRunning dynesty (nlive={args.nlive}) ...")
    results = build_sampler(model, processed, p_obs, domains, nlive=args.nlive)

    logz    = float(results.logz[-1])
    logzerr = float(results.logzerr[-1])
    print(f"\nlog Z = {logz:.3f} ± {logzerr:.3f}")

    # Save results
    with open(args.output_dir / f"{run_name}_dynesty.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(args.output_dir / f"{run_name}_dynesty_summary.json", "w") as f:
        json.dump({"logz": logz, "logzerr": logzerr}, f, indent=2)

    plot_corner(
        results,
        truths=truths,
        output_path=args.output_dir / f"{run_name}_dynesty_corner.png",
    )


if __name__ == "__main__":
    main()
