import numpy as np
import corner
import matplotlib.pyplot as plt

def plot_corner(unthinned_chain: np.ndarray, diagnostic: dict, df: int = 10):
    tau = diagnostic["tau"]
    discard = int(df * tau)   # Safe number to discard
    flat = unthinned_chain[discard:, :, :5].reshape(-1, 5)
    
    labels  = ["L40_xray", "fesc10", "epsilon", "h", "fnoise"]
    
    plt.figure(figsize=(10, 10), dpi=150)
    corner.corner(
        flat,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )
    plt.show()