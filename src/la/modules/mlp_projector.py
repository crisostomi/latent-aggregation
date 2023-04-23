import torch.nn as nn


class VanillaMLPProjector(nn.Module):
    def __init__(self, num_input_features: int, hidden_dim, projection_dim, *args, **kwargs):
        super(VanillaMLPProjector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.model(x)
