import torchvision
import torchvision.transforms as transforms
from cv_models import ResNet18
import torch

transform_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # mirroring
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4913997, 0.4821584, 0.446531],
            [0.2470323, 0.243485, 0.2615876],
        ),
    ]
)
transform_cifar10_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4913997, 0.4821584, 0.446531],
            [0.2470323, 0.243485, 0.2615876],
        ),
    ]
)

# registries
transform_registry_train = {
    "cifar10": transform_cifar10_train,
    "cifar10_data_augmentation": transform_cifar10_train,
}
transform_registry_test = {
    "cifar10": transform_cifar10_test,
}


def get_dataset(dataset_name, data_dir, train: bool):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        transform = (
            transform_registry_train.get("cifar10")
            if train
            else transform_registry_test.get("cifar10")
        )
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )

    return dataset


def seed_worker(worker_id):
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(
    dataset, batch_size, num_workers, shuffle=False, seed=0, *args, **kwargs
):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        *args,
        **kwargs,
    )
