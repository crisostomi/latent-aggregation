import timm
from timm.models import EfficientNet
from timm.models.layers.classifier import create_classifier
from torch import nn
from torch.nn import functional as F


class MyEfficientNet(nn.Module):
    def __init__(self, num_classes, pre_head_dim: int, drop_out_rate: float = 0.0, *args, **kwargs):

        super().__init__()

        efficient_net: EfficientNet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)

        self.pre_head = nn.Sequential(
            nn.Linear(efficient_net.num_features, pre_head_dim), nn.ReLU(), nn.Linear(pre_head_dim, pre_head_dim)
        )

        self.drop_out_rate = drop_out_rate
        self.global_pool, self.classifier = create_classifier(pre_head_dim, efficient_net.num_classes, pool_type="avg")

        self.efficient_net = efficient_net

    def forward(self, x):
        embeds = self.forward_pre_head(x)

        logits = self.forward_head(embeds)

        return {"embeds": embeds, "logits": logits}

    def forward_pre_head(self, x):
        x = self.efficient_net.forward_features(x)

        x = self.global_pool(x)
        x = self.pre_head(x)

        return x

    def forward_head(self, x):
        if self.drop_out_rate > 0:
            x = F.dropout(x, p=self.drop_out_rate, training=self.training)

        return self.classifier(x)
