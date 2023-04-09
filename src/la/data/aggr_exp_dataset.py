import logging

import hydra
import omegaconf
from nn_core.common import PROJECT_ROOT
from torch.utils.data import Dataset

pylogger = logging.getLogger()


class MyDataset(Dataset):
    def __init__(self, split, samples, **kwargs):
        super().__init__()
        self.split = split
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample, label = self.samples[index]
        return {"x": sample, "y": label}

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
