from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        self.proj = nn.Linear(32 * 8 * 8, 256)
        self.final_activation = nn.Sigmoid()
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        # (batch_size, num_channels, width, height)

        x = self.model(x)

        # (batch_size, 32 * 8 * 8)
        x = x.reshape(x.size(0), -1)

        embeds = self.proj(x)

        x = self.final_activation(embeds)

        logits = self.out(x)

        return {"embeds": embeds, "logits": logits}
