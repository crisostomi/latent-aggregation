import timm
from timm.models import EfficientNet
from torch import nn


class EfficientNetWrapper(nn.Module):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()

        efficient_net: EfficientNet = timm.create_model(model_name, pretrained=True, num_classes=0)

        efficient_net.eval()
        efficient_net.requires_grad_(False)

        self.efficient_net = efficient_net

    def forward(self, x):
        embeds = self.efficient_net(x)
        return embeds
