from math import floor
import torch 
from torch import nn
import logging 

pylogger = logging.getLogger(__name__)

class StudentCNN(nn.Module):
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
        super(StudentCNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        H = W = floor((floor((input_dim) / 2)) / 2)
        C = num_out_channels

        fake_tensor = torch.zeros(1, 3, input_dim, input_dim)
        shapes = self.backbone(fake_tensor).shape
        pylogger.info(f"shapes: {shapes}")
        H, W, C = shapes[2], shapes[3], shapes[1]

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
