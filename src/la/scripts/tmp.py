import numpy as np
from la.data.my_dataset_dict import MyDatasetDict
from datasets import Dataset, DatasetDict, load_dataset, Array3D, Image, Features
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision.transforms import ToTensor, Normalize


#################################
# Without transform
#################################

train_dataset = load_dataset(
    "cifar100",
    split="train",
    use_auth_token=True,
)

train_dataset.set_format(type="numpy", columns=["img", "fine_label"])

train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    pin_memory=False,
    shuffle=True,
    num_workers=8,
)

for batch in tqdm(train_loader, desc="Loading data, no transform"):
    pass


#################################
# With transform
#################################


img_shape = train_dataset[0]["img"].shape

features = train_dataset.features.copy()
features["x"] = Array3D(shape=img_shape, dtype="float32")

train_dataset = train_dataset.map(
    desc=f"Preprocessing samples",
    function=lambda x: {"x": np.array(x["img"], dtype=np.uint8)},
    features=features,
)
train_dataset.cast_column("x", Array3D(shape=img_shape, dtype="float32"))
train_dataset.set_format(type="numpy", columns=["x", "fine_label"])


transform_func = torchvision.transforms.Compose(
    [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = train_dataset.map(
    desc=f"Preprocessing samples",
    function=lambda x: {"x": transform_func(x["img"])},
)

train_dataset.set_format(type="numpy", columns=["x", "fine_label"])


train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    pin_memory=False,
    shuffle=True,
    num_workers=8,
)


for batch in tqdm(train_loader, desc="Loading data after transform"):
    pass
