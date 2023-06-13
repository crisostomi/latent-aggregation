from functools import partial
import pytorch_lightning
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Model(pytorch_lightning.LightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        use_relatives: bool,
    ):
        super().__init__()
        self.classifier = classifier

        self.accuracy = torchmetrics.Accuracy()

        self.use_relatives = use_relatives
        self.embedding_key = "relative_embeddings" if self.use_relatives else "embedding"

    def forward(self, batch):
        x = batch[self.embedding_key]
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        y = batch["y"]

        y_hat = self(batch)

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["y"]

        y_hat = self(batch)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        val_acc = self.accuracy(y_hat, y)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        y = batch["y"]

        y_hat = self(batch)

        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

        test_acc = self.accuracy(y_hat, y)
        self.log("test_acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class PartSharedPartNovelModel(Model):
    """
    Same as Model but also logs the accuracy for shared and non-shared classes
    """

    def __init__(
        self,
        classifier: nn.Module,
        shared_classes: set,
        non_shared_classes: set,
        use_relatives: bool,
    ):
        super().__init__(classifier, use_relatives)

        shared_classes = torch.Tensor(list(shared_classes)).long()
        non_shared_classes = torch.Tensor(list(non_shared_classes)).long()

        self.register_buffer("shared_classes", shared_classes)
        self.register_buffer("non_shared_classes", non_shared_classes)

        self.accuracy = torchmetrics.Accuracy()

    def test_step(self, batch, batch_idx):
        x, y = batch[self.embedding_key], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=True)

        test_acc = self.accuracy(y_hat, y)
        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True)

        # compute accuracy for shared classes
        shared_classes_mask = torch.isin(y, self.shared_classes)
        shared_classes_y = y[shared_classes_mask]

        y_hat = torch.argmax(y_hat, dim=1)
        shared_classes_y_hat = y_hat[shared_classes_mask]

        shared_classes_acc = torch.sum(shared_classes_y == shared_classes_y_hat) / len(shared_classes_y)
        self.log(
            "test_acc_shared_classes",
            shared_classes_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # compute accuracy for non-shared classes
        non_shared_classes_mask = torch.isin(y, self.non_shared_classes)
        non_shared_classes_y = y[non_shared_classes_mask]
        non_shared_classes_y_hat = y_hat[non_shared_classes_mask]

        non_shared_classes_acc = torch.sum(non_shared_classes_y == non_shared_classes_y_hat) / len(non_shared_classes_y)
        self.log(
            "test_acc_non_shared_classes",
            non_shared_classes_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss


class Classifier(nn.Module):
    def __init__(self, input_dim, classifier_embed_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.InstanceNorm1d(input_dim),
            nn.Linear(input_dim, classifier_embed_dim),
            nn.ReLU(),
            nn.Linear(classifier_embed_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def run_classification_experiment(
    shared_classes,
    non_shared_classes,
    num_total_classes,
    global_cfg,
    num_anchors,
    dataset,
    use_relatives,
):
    pytorch_lightning.seed_everything(42)

    dataloader_func = partial(
        torch.utils.data.DataLoader,
        batch_size=128,
        num_workers=8,
    )

    trainer_func = partial(pytorch_lightning.Trainer, gpus=1, max_epochs=100, logger=False, enable_progress_bar=True)

    classifier = Classifier(
        input_dim=num_anchors,
        classifier_embed_dim=global_cfg.classifier_embed_dim,
        num_classes=num_total_classes,
    )
    model = Model(
        classifier=classifier,
        shared_classes=shared_classes,
        non_shared_classes=non_shared_classes,
        use_relatives=use_relatives,
    )
    trainer = trainer_func(callbacks=[pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=10)])

    # split dataset in train, val and test
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = split_dataset["train"]
    val_test_dataset = split_dataset["test"]

    split_val_test = val_test_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = split_val_test["train"]
    test_dataset = split_val_test["test"]

    train_dataloader = dataloader_func(train_dataset, shuffle=True)
    val_dataloader = dataloader_func(val_dataset, shuffle=False)
    test_dataloader = dataloader_func(test_dataset, shuffle=False)

    trainer.fit(model, train_dataloader, val_dataloader)

    results = trainer.test(model, test_dataloader)[0]

    results = {
        "total_acc": results["test_acc_epoch"],
        "shared_class_acc": results["test_acc_shared_classes_epoch"],
        "non_shared_class_acc": results["test_acc_non_shared_classes_epoch"],
    }

    return results


class KNNClassifier(pytorch_lightning.LightningModule):
    def __init__(self, train_dataset, num_classes, use_relatives):
        super().__init__()
        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.accuracy = torchmetrics.Accuracy()
        self.embedding_key = "relative_embeddings" if use_relatives else "embedding"

    def on_train_epoch_end(self) -> None:
        prototypes = compute_prototypes(
            self.train_dataset[self.embedding_key], self.train_dataset["y"], num_classes=self.num_classes
        )
        self.register_buffer("prototypes", prototypes)

    def forward(self, x):
        distances = torch.cdist(x, self.prototypes)

        predictions = torch.argmin(distances, dim=1)

        return predictions

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

    def test_step(self, batch, batch_idx):
        assert self.prototypes is not None
        x, y = batch[self.embedding_key], batch["y"]
        y_hat = self(x)

        test_acc = self.accuracy(y_hat, y)
        self.log("test_acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        pass


class TaskEmbeddingModel(Model):
    def __init__(self, num_tasks, task_embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)

    def forward(self, batch):

        x, task_ids = batch[self.embedding_key], batch["task"]

        task_embedding = self.task_embedding(task_ids - 1)
        x = torch.cat([x, task_embedding], dim=1)

        return self.classifier(x)


def compute_prototypes(x, y, num_classes):
    prototypes = []
    for i in range(num_classes):
        samples_class_i = x[y == i]
        prototype = torch.mean(samples_class_i, dim=0)
        prototypes.append(prototype)

    prototypes = torch.stack(prototypes)

    return prototypes
