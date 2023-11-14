import torch.nn.functional as F
from torch import optim
from torch import nn
import torch

from scipy.optimize import minimize_scalar
import numpy as np
import argparse
import pickle
import random
import time

from cv_datasets import get_dataset, get_num_classes, get_data_loader
from cv_models import ResNet18, ResNet34
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
        type=str,
        default="ResNet34",
        help="Model",
    )

    # Repeat experiments
    parser.add_argument(
        "--repeat", type=int, default=5, help="number of times to repeat training"
    )

    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    # parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=100, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=128, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--l2", type=float, default=1e-4, help="l2 regularization (weight decay)"
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


def temp_nll(t, logits, y_true):
    logits_scaled = logits / t
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits_scaled)
        probs = torch.cat([1.0 - probs, probs], dim=1)
    else:
        probs = F.softmax(logits_scaled, dim=1)
    probs_true = probs[np.arange(len(y_true)), y_true]
    return (-torch.log2(probs_true).sum()).item()


def get_t(model, X, y):
    with torch.no_grad():
        model.eval()
        logits = []
        for X_iter, y_iter in model.iterator(128, X, None, shuffle=False):
            logits.extend(model.forward(X_iter).tolist())
        logits = torch.tensor(logits)
        model.train()
    f = lambda t: temp_nll(t, logits, y)
    res = minimize_scalar(f, bounds=(0, 100), method="bounded")
    return res.x


UNC_METHODS = [unc.BLOODQuant(), unc.LeastConfidentQuant(), unc.EntropyQuant()]
DATA = {"cifar10", "cifar100", "svhn"}
MODEL_CLS = {"ResNet18": ResNet18, "ResNet34": ResNet34}


if __name__ == "__main__":
    START = time.time()
    args = make_parser()

    cuda = torch.cuda.is_available() and args.gpu != -1
    device = torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")
    criterion = nn.CrossEntropyLoss()

    result_dict = {}

    DATA_OUT = args.data_out
    DATA_IN = list(DATA - set([DATA_OUT]))
    MODEL_NAMES = [args.model]

    ROOT = args.data_dir

    n_classes_out = get_num_classes(DATA_OUT)
    test_set_out = get_dataset(DATA_OUT, ROOT, train=False)
    test_loader_out = get_data_loader(
        test_set_out, batch_size=args.batch_size, shuffle=False
    )

    for dataset_in in DATA_IN:
        set_seed(42)
        print(f"{dataset_in} - {time.time()-START}s")

        n_classes_in = get_num_classes(dataset_in)
        train_set_in = get_dataset(dataset_in, ROOT, train=True)
        test_set_in = get_dataset(dataset_in, data_dir=ROOT, train=False)
        train_loader_in = get_data_loader(
            train_set_in, batch_size=args.batch_size, shuffle=True
        )
        test_loader_in = get_data_loader(
            test_set_in, batch_size=args.batch_size, shuffle=False
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

                model = MODEL_CLS[model_name](num_c=n_classes_in, device=device)
                model.to(device)
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
                        data_loader=test_loader_in, model=model, **{}
                    )
                    unc_out = uncertainty.quantify(
                        data_loader=test_loader_out, model=model, **{}
                    )

                    result_seed[uncertainty.name]["id"] = unc_in
                    result_seed[uncertainty.name]["ood"] = unc_out

                result_dict[dataset_in][model_name]["fine-tuned"].append(result_seed)

    with open(f"results/results_{DATA_OUT}.pkl", "wb") as f:
        pickle.dump(result_dict, f)
