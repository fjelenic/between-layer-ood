import my_datasets as md
import my_models as mm
import my_uncertainty as mu
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pickle
import random
import math
import time
from scipy.optimize import minimize_scalar


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def temp_nll(t, logits, y_true):
    logits_scaled = logits/t
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
    res = minimize_scalar(f, bounds=(0, 100), method='bounded')
    return res.x
    
START = time.time()
BATCH_SIZE = 32
GPU = "cuda:0" 
FILE = "results.pkl"
UNCERNS = [mu.BLOODQuant(), mu.LeastConfidentQuant(), mu.EntropyQuant(), mu.MCDropoutQuant(), mu.GradQuant()] + [mu.TemperatureScalingQuant(), mu.EnsambleQuant(), mu.MahalanobisQuant()] + [mu.RepresentationChangeQuant()]
DATA = [md.SST2Data(), md.SubjectivityData(), md.AGNewsData(), md.TrecData(), md.BigPatentData(), md.AmazonReviewData(), md.FilmGenreData(), md.NewsGroupsData(), md.BigPatent2Data(), md.AmazonReview2Data(), md.FilmGenre2Data()]
MODEL_NAMES = ["RoBERTa", "ELECTRA"]
NUM_SEEDS = 5
BATCH_SIZE_INFERENCE = 32
NUM_ENSAMBLE_EST = 5
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
    for model_name in MODEL_NAMES:
        print(f"\t{model_name} - {time.time()-START}s")
        rez[data.name][model_name] = {}     
        
        model = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(GPU))
        criterion = nn.BCEWithLogitsLoss() if data.num_out == 1 else nn.CrossEntropyLoss()
        criterion.to(model.device)         
        rez[data.name][model_name]["pre-trained"] = {}
        rez[data.name][model_name]["pre-trained"]["CKA"] = {}
        for X, distrib_type in zip([X_train_id, X_test_id, X_test_ood], ["train", "id", "ood"]):
            print(f"\t\t{distrib_type} (PRE-TRAINED) - {time.time()-START}s")
            kwargs = {"X_eval": X, "X_anchor": X_train_id, "y_anchor": y_train_id, "model": model, "criterion": criterion, "batch_size": 16 if data.name in LONG_DATA else BATCH_SIZE_INFERENCE, "path": True}
            u = mu.BLOODQuant().quantify(**kwargs)
            rez[data.name][model_name]["pre-trained"][distrib_type] = u
            print(f"\t\t{distrib_type} (PRE-TRAINED) - {time.time()-START}s")
        
        rez[data.name][model_name]["fine-tuned"] = []
        for seed in range(NUM_SEEDS):
            set_seed(seed)
            print(f"\t\tSeed: {seed+1} - {time.time()-START}s")
            rez_seed = {}
            models = []
            for m in range(NUM_ENSAMBLE_EST):
                print(f"\t\tTraining model #{m+1} - {time.time()-START}s")
                model = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(GPU))
                criterion = nn.BCEWithLogitsLoss() if data.num_out == 1 else nn.CrossEntropyLoss()
                criterion.to(model.device)
                model.train_loop(X_train_id, y_train_id, criterion=criterion, batch_size=BATCH_SIZE//2 if data.name in LONG_DATA else BATCH_SIZE, cartography=False, X_val=X_test_id, y_val=y_test_id if m==0 else None)
                models.append(model)
                
            model_orig = mm.TransformerClassifier(model_name, data.num_out, device=torch.device(GPU))

            inds_val = random.sample(range(len(X_test_id)), round(len(X_test_id)*0.2))
            rez_seed["inds_val"] = inds_val
            X_val, y_val, X_non_val, y_non_val = [], [], [], []
            for i in range(len(X_test_id)):
                if i in inds_val:
                    X_val.append(X_test_id[i])
                    y_val.append(y_test_id[i])
                else:
                    X_non_val.append(X_test_id[i])
                    y_non_val.append(y_test_id[i])
            print(len(X_val))
            print(len(X_non_val))
            print(len(X_test_id))
            T = get_t(models[0], X_val, y_val)
            rez_seed["T"] = T
            print(T)

            for uncertainty in UNCERNS:
                print(f"\t\t\t{uncertainty.name} - {time.time()-START}s")
                rez_seed[uncertainty.name] = {}
                for X, distrib_type in zip([X_non_val if uncertainty.name == mu.TemperatureScalingQuant().name else X_test_id, X_test_ood], ["id", "ood"]):
                    print(f"\t\t\t\t{distrib_type} - {time.time()-START}s")
                    kwargs = {"X_eval": X, "X_anchor": X_train_id, "y_anchor": y_train_id, "model": models[0], "criterion": criterion, "models": models, "batch_size": 16 if data.name in LONG_DATA else BATCH_SIZE_INFERENCE, "path": True, "T": T, "model_orig": model_orig}
                    u = uncertainty.quantify(**kwargs)
                    rez_seed[uncertainty.name][distrib_type] = u
                    print(f"\t\t\t\t{distrib_type} - {time.time()-START}s")

            rez[data.name][model_name]["fine-tuned"].append(rez_seed)
            with open(FILE, 'wb') as f:
                pickle.dump(rez, f)
            
print("DONE")


