import torch
import torch.nn as nn   


class Emulator(nn.Module):
    """
    A fully-connected feedforward neural network for emulating power spectrum PCA coefficients
    from cosmological parameters.

    Architecture: Linear → ReLU → [Linear → ReLU] × num_layers → Linear

    Parameters
    ----------
    input_dim : int, optional
        Number of input features (cosmological parameters). Default: 4.
    output_dim : int, optional
        Number of output features (PCA coefficients). Default: 6.
    hidden_dim : int, optional
        Width of each hidden layer. Default: 64.
    num_layers : int, optional
        Number of intermediate hidden layers between the first and final layer. Default: 4.

    Notes
    -----
    Total depth is num_layers + 2 (one input projection layer, num_layers hidden
    layers, and one output projection layer). ReLU activations are applied after
    every layer except the last, which is linear to allow unbounded regression output.
    """
    def __init__(self, input_dim: int = 4, output_dim: int = 6, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor of shape (N, input_dim)
            Batch of standardized cosmological parameter vectors.

        Returns
        -------
        torch.Tensor of shape (N, output_dim)
            Predicted standardized PCA coefficients.
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x