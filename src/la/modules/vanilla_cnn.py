from math import floor

from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        num_interm_channels: int,
        num_out_channels: int,
        embedding_dim: int,
        *args,
        **kwargs
    ):
        super(CNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_interm_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=num_interm_channels,
                out_channels=num_out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        H = W = floor((floor((input_dim) / 2)) / 2)
        C = num_out_channels

        self.proj = nn.Linear(H * W * C, embedding_dim)
        self.final_activation = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # (batch_size, num_channels, width, height)

        x = self.backbone(x)

        # (batch_size, 32 * 8 * 8)
        x = x.reshape(x.size(0), -1)

        embeds = self.proj(x)

        x = self.final_activation(embeds)

        logits = self.out(x)

        return {"embeds": embeds, "logits": logits}
