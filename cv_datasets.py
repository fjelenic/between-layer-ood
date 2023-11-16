import torchvision
import torchvision.transforms as transforms
from cv_models import ResNet18
import torch
from datasets import load_dataset

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# transform_cifar10_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),  # mirroring
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.4913997, 0.4821584, 0.446531],
#             [0.2470323, 0.243485, 0.2615876],
#         ),
#     ]
# )
# transform_cifar10_test = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.4913997, 0.4821584, 0.446531],
#             [0.2470323, 0.243485, 0.2615876],
#         ),
#     ]
# )

# transform_cifar100_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),  # mirroring
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5070752, 0.486549, 0.4409178],
#             [0.2673342, 0.2564384, 0.2761506],
#         ),
#     ]
# )
# transform_cifar100_test = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5070752, 0.486549, 0.4409178],
#             [0.2673342, 0.2564384, 0.2761506],
#         ),
#     ]
# )

# transform_svhn_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),  # mirroring
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.4379793, 0.4439904, 0.4729508],
#             [0.1981116, 0.2011045, 0.1970895],
#         ),
#     ]
# )
# transform_svhn_test = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.4379793, 0.4439904, 0.4729508],
#             [0.1981116, 0.2011045, 0.1970895],
#         ),
#     ]
# )

# # registries
# transform_registry_train = {
#     "cifar10": transform_cifar10_train,
#     "cifar100": transform_cifar100_train,
#     "svhn": transform_svhn_train,
# }
# transform_registry_test = {
#     "cifar10": transform_cifar10_test,
#     "cifar100": transform_cifar100_test,
#     "svhn": transform_svhn_test,
# }
n_classes_registry = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
}


# def get_dataset(dataset_name, data_dir, train: bool):
#     dataset_name = dataset_name.lower()
#     transform = (
#         transform_registry_train.get(dataset_name)
#         if train
#         else transform_registry_test.get(dataset_name)
#     )
#     if dataset_name == "cifar10":
#         dataset = torchvision.datasets.CIFAR10(
#             root=data_dir, train=train, download=True, transform=transform
#         )
#     elif dataset_name == "cifar100":
#         dataset = torchvision.datasets.CIFAR100(
#             root=data_dir, train=train, download=True, transform=transform
#         )
#     elif dataset_name == "svhn":
#         split = "train" if train else "test"
#         dataset = torchvision.datasets.SVHN(
#             root=data_dir, split=split, download=True, transform=transform
#         )
#     else:
#         raise ValueError(
#             f"Dataset '{dataset_name}' is not available. Options: cifar10, cifar100, svhn"
#         )

#     return dataset


def get_hf_dataset(name):
    # load cifar10 (only small portion for demonstration purposes)
    if name == "svhn":
        return load_dataset(name, "cropped_digits", split=["train", "test"])
    return load_dataset(name, split=["train", "test"])


def get_num_classes(dataset_name):
    return n_classes_registry.get(dataset_name.lower())


def seed_worker(worker_id):
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn_cifar10(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def collate_fn_cifar100(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def collate_fn_svhn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def get_data_loaders(
    name,
    processor,
    batch_size,
    shuffle=False,
    seed=0,
):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    if name == "svhn":
        img_key = "image"
        collate = collate_fn_svhn

    elif name == "cifar10":
        img_key = "img"
        collate = collate_fn_cifar10
    elif name == "cifar100":
        img_key = "img"
        collate = collate_fn_cifar100

    def train_transforms(examples):
        examples["pixel_values"] = [
            _train_transforms(image.convert("RGB")) for image in examples[img_key]
        ]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [
            _val_transforms(image.convert("RGB")) for image in examples[img_key]
        ]
        return examples

    train_set, test_set = get_hf_dataset(name)
    train_set.set_transform(train_transforms)
    test_set.set_transform(val_transforms)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=g,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        generator=g,
    )

    return train_loader, test_loader
