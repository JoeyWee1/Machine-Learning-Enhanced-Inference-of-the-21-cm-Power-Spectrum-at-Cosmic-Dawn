# Machine-Learning-Enhanced Inference of the 21-cm Power Spectrum at Cosmic Dawn

 Implements simulation-based inference (SBI) of the 21-cm power spectrum from Cosmic Dawn using two  approaches: an NN emulator with an explicit Gaussian likelihood, and a Neural Ratio Estimator (NRE).

## Project Structure

```
.
├── data/                          # Input data (not tracked by git - see data/data_files.txt)
│   ├── simulations/               # 21-cm power spectrum simulations (sample_XXXXXX.npz)
│   └── observations/              # Observed power spectrum (observations.npz)
├── utils/
│   ├── preprocess.py              # PCA compression and standardisation pipeline
│   ├── basic_plots.py             # General plotting utilities
│   ├── general.py                 # Shared utilities (set_seed, load_splits)
│   ├── emulator/                  # NN emulator (train, evaluate, plot)
│   ├── mcmc/                      # MCMC sampling (emcee ensemble + dynesty nested)
│   └── nre/                       # NRE (dataset construction, training, evaluation, plot)
├── optuna/                        # Hyperparameter optimisation scripts
├── outputs/                        #  Not tracked by git - see outputs_files.txt
│   ├── figs/                      # Saved figures (see outputs/outputs_files.txt)
│   ├── optuna_outputs/            # Saved emulator model checkpoints and optuna DB
│   └── samples/                   # MCMC chains, NRE checkpoint and scaler
├── ML_inference.ipynb             # Main notebook - runs full inference pipeline
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT licence
└── MPhil_DIS_SKAera_coursework.pdf
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then place data in `data/` as described in `data/data_files.txt` and run `ML_inference.ipynb`.

## Methods

### 1. Emulator + Gaussian Likelihood
A fully-connected neural network emulates the mapping from 4 astrophysical parameters `[L40_xray, fesc10, epsilon, h]` to the 21-cm power spectrum via PCA compression. A Gaussian likelihood with a free noise parameter `fnoise` is used for inference. Posterior sampling is performed with both `emcee` (ensemble MCMC) and `dynesty` (nested sampling), in log-parameter space with a Jacobian correction.

### 2. Neural Ratio Estimator (NRE)
An NRE is trained to classify joint vs. marginal `(x, theta)` pairs, implicitly learning `log r(x, theta) ≈ log p(x|theta) - log p(x)`. This is used directly as a log-likelihood in both `emcee` and `dynesty`. Noise is injected during dataset construction by drawing `fnoise` log-uniformly from `[1e-3, 1]`.

## Autogeneration Tools
LLMs, specifically Claude and ChatGPT, assisted me in this coursework. They were used to debug code, improve code efficiency, write nice docstrings/function typing, help with matplotlib, and with converting my code to run on HPC (this was quite complicated). **THE ANALYSIS IS MY OWN**

