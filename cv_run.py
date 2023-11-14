import torch.nn.functional as F
from torch import optim
from torch import nn
import torch

from scipy.optimize import minimize_scalar
import numpy as np
import pickle
import random
import time

from cv_datasets import get_dataset, get_num_classes, get_data_loader
from cv_models import ResNet18, ResNet34
from cv_train import train_model
import cv_uncertainty as unc


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


START = time.time()
BATCH_SIZE = 32
DEVICE = "cuda:0"
FILE = "results_svhn.pkl"
UNC_METHODS = [unc.BLOODQuant(), unc.LeastConfidentQuant(), unc.EntropyQuant()]
DATA_IN = ["cifar10"]
DATA_OUT = "svhn"
MODEL_NAMES = ["ResNet18"]
NUM_SEEDS = 1
BATCH_SIZE_INFERENCE = 32
ROOT = "cv_data"
NUM_EPOCHS = 1
LR = 1e-5
WD = 1e-6

MODEL_CLS = {"ResNet18": ResNet18, "ResNet34": ResNet34}


criterion = nn.CrossEntropyLoss()
device = torch.device(DEVICE)

result_dict = {}

n_classes_out = get_num_classes(DATA_OUT)
test_set_out = get_dataset(DATA_OUT, ROOT, train=False)
test_loader_out = get_data_loader(test_set_out, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    for dataset_in in DATA_IN:
        set_seed(42)
        print(f"{dataset_in} - {time.time()-START}s")

        n_classes_in = get_num_classes(dataset_in)
        train_set_in = get_dataset(dataset_in, ROOT, train=True)
        test_set_in = get_dataset(dataset_in, data_dir=ROOT, train=False)
        train_loader_in = get_data_loader(
            train_set_in, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader_in = get_data_loader(
            test_set_in, batch_size=BATCH_SIZE, shuffle=False
        )

        result_dict[dataset_in] = {}
        for model_name in MODEL_NAMES:
            print(f"\t{model_name} - {time.time()-START}s")
            result_dict[dataset_in][model_name] = {}
            result_dict[dataset_in][model_name]["fine-tuned"] = []

            for seed in range(NUM_SEEDS):
                set_seed(seed)
                print(f"\t\tSeed: {seed+1} - {time.time()-START}s")
                result_seed = {}

                model = MODEL_CLS[model_name](num_c=n_classes_in, device=device)
                model.to(device)
                # model = nn.DataParallel(model)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
                # scheduler = None
                train_model(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader_in,
                    test_loader=test_loader_in,
                    device=device,
                    num_epochs=NUM_EPOCHS,
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

    with open(FILE, "wb") as f:
        pickle.dump(result_dict, f)
