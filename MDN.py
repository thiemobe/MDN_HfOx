"""
Mixture Density Network (MDN) implementation in PyTorch.
The benefit of using MDNs is that they can model complex, multimodal distributions.
The mixture captures multi-modal behavior - memristors can have multiple stable states for the same input, 
and MDN can model this complexity!
"""

import torch
import torch.nn as nn

# =========================
# Mixture Density Network
# =========================
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_components, num_hidden_layers=2, dropout_rate=0.0):
        """
        input_dim: number of input features
        hidden_dim: width of hidden layers (can be int or list of ints)
                    - if int: all layers have same width
                    - if list: specify width for each layer [layer1_dim, layer2_dim, ...]
        n_components: number of mixture components
        num_hidden_layers: 1, 2 or 3 (how many hidden linear layers to stack)
        dropout_rate: dropout probability between hidden layers (0.0 means no dropout)
        """
        super(MDN, self).__init__()

        assert num_hidden_layers >= 1 and num_hidden_layers <= 3, "num_hidden_layers must be 1..3"

        # Convert hidden_dim to list if it's a single int
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * num_hidden_layers
        else:
            hidden_dims = list(hidden_dim)
            assert len(hidden_dims) == num_hidden_layers, f"hidden_dim list length ({len(hidden_dims)}) must match num_hidden_layers ({num_hidden_layers})"

        layers = []
        # first layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.Mish())
        if dropout_rate and dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        # additional hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.Mish())
            if dropout_rate and dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        self.hidden = nn.Sequential(*layers)

        # Output heads connect from the last hidden layer
        final_hidden_dim = hidden_dims[-1]
        self.pi_head = nn.Linear(final_hidden_dim, n_components)       # mixture weights
        self.mu_head = nn.Linear(final_hidden_dim, n_components)       # means
        self.sigma_head = nn.Linear(final_hidden_dim, n_components)    # stddevs

        self.n_components = n_components

    def forward(self, x):
        h = self.hidden(x)
        pi = torch.softmax(self.pi_head(h), dim=-1)  # mixture weights sum to 1
        mu = self.mu_head(h)
        # produce positive stddev and ensure a minimum value for numerical stability
        sigma = torch.exp(self.sigma_head(h))        # positive stddev
        sigma = torch.clamp(sigma, min=1e-6)
        return pi, mu, sigma