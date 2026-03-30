from torch import nn
class NRE(nn.Module):
    def __init__(self,
        input_dim: int = 59,
        hidden_dim: int = 100,
        depth: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        
        super().__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, 128))
        layers.append(activation)
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(128 if _ == 0 else hidden_dim, hidden_dim))
            layers.append(activation)
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.Sequential(*layers)

        # self.layers = nn.Sequential(
        #     nn.Linear(59, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 1),
        # )
    
    def forward(self, x):
        # x is shape (n_samps, 59)
        return self.layers(x)
