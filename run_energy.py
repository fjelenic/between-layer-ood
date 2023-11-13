import my_datasets as md
import my_models as mm
import my_uncertainty as mu
import torch
from torch import nn
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pickle
import random
import math
import time
from my_utils import get_cka_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
START = time.time()
BATCH_SIZE = 32
GPU = "cuda:0"
FILE = "results-energy.pkl" 
UNCERNS = [mu.EnergyQuant(), mu.ReactQuant(), mu.ASHQuant()]
DATA = [md.SST2Data(), md.SubjectivityData(), md.AGNewsData(), md.TrecData(), md.BigPatentData(), md.AmazonReviewData(), md.FilmGenreData(), md.NewsGroupsData(), md.BigPatent2Data(), md.AmazonReview2Data(), md.FilmGenre2Data()]
MODEL_NAMES = ["RoBERTa", "ELECTRA"]
NUM_SEEDS = 5
BATCH_SIZE_INFERENCE = 32
LONG_DATA = [md.NewsGroupsData().name, md.AGNewsData().name, md.BigPatentData().name, md.FilmGenreData().name, md.AmazonReviewData().name, md.BigPatent2Data().name, md.FilmGenre2Data().name, md.AmazonReview2Data().name]

rez = {}

_, X_test_ood_all, _, _, _ = md.OneBillionData().load()
    
for data in DATA:
    set_seed(42+len(data.name))
    print(f"{data.name} - {time.time()-START}s")
    X_train_id, X_test_id, y_train_id, y_test_id, _ = data.load()
    X_test_ood_all_temp = X_test_ood_all[:]
    random.shuffle(X_test_ood_all_temp)
    X_test_ood = X_test_ood_all_temp[:len(X_test_id)]
    rez[data.name] = {}
    
    if data.num_out == 1:
        data.num_out = 2
    
    for model_name in MODEL_NAMES:
        print(f"\t{model_name} - {time.time()-START}s")
        rez[data.name][model_name] = {}
        model = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(GPU))
        criterion = nn.BCEWithLogitsLoss() if data.num_out == 1 else nn.CrossEntropyLoss()
        criterion.to(model.device)
        
        rez[data.name][model_name]["fine-tuned"] = []
        for seed in range(NUM_SEEDS):
            set_seed(seed)
            print(f"\t\tSeed: {seed+1} - {time.time()-START}s")
            rez_seed = {}
            model = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(GPU))
            criterion = nn.BCEWithLogitsLoss() if data.num_out == 1 else nn.CrossEntropyLoss()
            criterion.to(model.device)
            model.train_loop(X_train_id, y_train_id, criterion=criterion, batch_size=BATCH_SIZE//2 if data.name in LONG_DATA else BATCH_SIZE, cartography=False, X_val=X_test_id, y_val=y_test_id)
            model_orig = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(GPU))

            for uncertainty in UNCERNS:
                print(f"\t\t\t{uncertainty.name} - {time.time()-START}s")
                rez_seed[uncertainty.name] = {}
                for X, distrib_type in zip([X_train_id, X_test_id, X_test_ood], ["train", "id", "ood"]):
                    print(f"\t\t\t\t{distrib_type} - {time.time()-START}s")
                    kwargs = {"X_eval": X, "X_anchor": X_train_id, "y_anchor": y_train_id, "model": model, "model_orig":model_orig, "criterion": criterion, "batch_size": 16 if data.name in LONG_DATA else BATCH_SIZE_INFERENCE, "path": True}
                    u = uncertainty.quantify(**kwargs)
                    rez_seed[uncertainty.name][distrib_type] = u
                    print(f"\t\t\t\t{distrib_type} - {time.time()-START}s")

            rez[data.name][model_name]["fine-tuned"].append(rez_seed)
            with open(FILE, 'wb') as f:
                pickle.dump(rez, f)
            
print("DONE")


