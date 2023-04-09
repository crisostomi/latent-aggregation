from timm.models import EfficientNet
from timm.models.layers.classifier import create_classifier
from torch import nn
from torch.nn import functional as F


class MyEfficientNet(nn.Module):
    def __init__(self, efficient_net: EfficientNet, pre_head_dim: int, drop_out_rate: float = 0.0):

        super().__init__()
        self.pre_head = nn.Linear(efficient_net.num_features, pre_head_dim)

        self.drop_out_rate = drop_out_rate
        self.global_pool, self.classifier = create_classifier(pre_head_dim, efficient_net.num_classes, pool_type="avg")

        self.efficient_net = efficient_net
        self.efficient_net.requires_grad_(False)
        self.efficient_net.eval()
        self.relu = nn.ReLU()

    def forward_pre_head(self, x):
        x = self.efficient_net.forward_features(x)

        x = self.global_pool(x)
        x = self.pre_head(x.squeeze(-1).squeeze(-1))

        return x

    def forward(self, x):
        self.efficient_net.eval()

        x = self.forward_pre_head(x)
        x = self.forward_head(x)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.drop_out_rate > 0:
            x = F.dropout(x, p=self.drop_out_rate, training=self.training)
        return x if pre_logits else self.classifier(x)
