import torch.nn.functional as F
from torch import optim
from torch import nn
import torch

from transformers import AutoImageProcessor

from scipy.optimize import minimize_scalar
import numpy as np
import argparse
import pickle
import random
import time
import copy

from cv_models import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet18PT,
    ResNet34PT,
    ResNet50PT,
    ViT,
)
from cv_datasets import get_num_classes, get_data_loaders
from cv_train import train_model
import cv_uncertainty as unc


def make_parser():
    parser = argparse.ArgumentParser(description="CV OOD")
    parser.add_argument(
        "--data-out",
        type=str,
        default="svhn",
        help="Data corpus.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["ViT"],
        choices=[
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet18PT",
            "ResNet34PT",
            "ResNet50PT",
            "ViT",
        ],
        help="Model",
    )

    # Repeat experiments
    parser.add_argument(
        "--repeat", type=int, default=5, help="number of times to repeat training"
    )

    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    # parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=5, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--l2", type=float, default=1e-2, help="l2 regularization (weight decay)"
    )
    # Gpu based arguments
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Gpu to use for experiments (-1 means no GPU)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="cv_results",
        help="Folder to store the results.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="cv_data",
        help="data folder",
    )
    parser.add_argument(
        "--scheduler",
        type=bool,
        default=False,
        help="Bool: use linear decay scheduler.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


UNC_METHODS = [
    # unc.GradQuant(),
    # unc.RepresentationChangeQuant(),
    unc.BLOODQuant(),
    unc.LeastConfidentQuant(),
    unc.EntropyQuant(),
    unc.EnergyQuant(),
]
DATA = {"cifar10", "cifar100", "svhn"}
MODEL_CLS = {
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet18PT": ResNet18PT,
    "ResNet34PT": ResNet34PT,
    "ResNet50PT": ResNet50PT,
    "ViT": ViT,
}


if __name__ == "__main__":
    START = time.time()
    args = make_parser()

    cuda = torch.cuda.is_available() and args.gpu != -1
    device = torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")
    criterion = nn.CrossEntropyLoss()

    result_dict = {}

    DATA_OUT = args.data_out
    DATA_IN = list(DATA - set([DATA_OUT]))
    MODEL_NAMES = args.model

    ROOT = args.data_dir

    n_classes_out = get_num_classes(DATA_OUT)
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_loader_out, test_loader_out = get_data_loaders(
        DATA_OUT, batch_size=args.batch_size, processor=processor
    )

    for dataset_in in DATA_IN:
        set_seed(42)
        print(f"{dataset_in} - {time.time()-START}s")

        n_classes_in = get_num_classes(dataset_in)
        train_loader_in, test_loader_in = get_data_loaders(
            dataset_in, batch_size=args.batch_size, processor=processor
        )

        result_dict[dataset_in] = {}
        for model_name in MODEL_NAMES:
            print(f"\t{model_name} - {time.time()-START}s")
            result_dict[dataset_in][model_name] = {}
            result_dict[dataset_in][model_name]["fine-tuned"] = []

            for seed in range(args.repeat):
                set_seed(seed)
                print(f"\t\tSeed: {seed+1} - {time.time()-START}s")
                result_seed = {}

                model = MODEL_CLS[model_name](
                    n_classes_in, device=device, processor=processor
                )
                model.to(device)
                base_model = copy.deepcopy(model)
                # model = nn.DataParallel(model)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.l2
                )
                # scheduler = None
                train_model(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader_in,
                    test_loader=test_loader_in,
                    device=device,
                    num_epochs=args.epochs,
                )

                for uncertainty in UNC_METHODS:
                    print(f"\t\t\t{uncertainty.name} - {time.time()-START}s")
                    result_seed[uncertainty.name] = {}

                    unc_in = uncertainty.quantify(
                        data_loader=test_loader_in,
                        model=model,
                        **dict(base_model=base_model),
                    )
                    unc_out = uncertainty.quantify(
                        data_loader=test_loader_out,
                        model=model,
                        **dict(base_model=base_model),
                    )

                    result_seed[uncertainty.name]["id"] = unc_in
                    result_seed[uncertainty.name]["ood"] = unc_out

                result_dict[dataset_in][model_name]["fine-tuned"].append(result_seed)

    with open(f"results/results_{DATA_OUT}.pkl", "wb") as f:
        pickle.dump(result_dict, f)
