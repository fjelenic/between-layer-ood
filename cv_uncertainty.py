import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F


class LeastConfidentQuant:
    name = "least_confident"

    def quantify(self, model, data_loader, **kwargs):
        probs = model.predict_proba(data_loader).numpy()
        max_probs = np.max(probs, axis=1)
        return -max_probs


class EntropyQuant:
    name = "entropy"

    def quantify(self, model, data_loader, **kwargs):
        probs = model.predict_proba(data_loader).numpy()
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        return entropies


class EnergyQuant():
    name = "energy"

    def quantify(self, model, data_loader, **kwargs):
        logits = []
        for batch in data_loader:
            X, _ = batch
            X = X.to(model.device)
            logits.extend(model(X).tolist())
        logits = np.array(logits)
        return -np.log(np.sum(np.exp(logits), axis=1))


# class MCDropoutQuant():
#     name = "MC"

#     def quantify(self, X_eval, model, k=30, **kwargs):
#         probs_dropout = model.predict_proba_dropout(X_eval)  # (batch_size, num_labels)

#         entropies_dropout = []
#         for _ in range(k-1):
#             probs_dropout += model.predict_proba_dropout(X_eval)

#         probs_dropout /= k
#         probs_dropout = np.clip(probs_dropout, a_min=1e-6, a_max=None)
#         entropies_dropout = np.sum(-probs_dropout * np.log(probs_dropout), axis=1)

#         return entropies_dropout


class GradQuant():
    name = "gradient"

    def quantify(self, model, data_loader, criterion, **kwargs):
        grad_embedding = model.get_grad_embedding(data_loader) # (batch_size, embedding_size)
        return np.linalg.norm(grad_embedding, ord=2, axis=1)


class BLOODQuant:
    name = "BLOOD"

    def quantify(self, model, data_loader, estimator=True, **kwargs):
        norms = model.get_grad_layers(data_loader, estimator=estimator)
        return norms


class RepresentationChangeQuant:
    name = "repr_change"

    def quantify(self, model, data_loader, base_model, **kwargs):
        repr_base = base_model.get_encoded_layers(
            data_loader
        )  # (num_layers, batch_size, embeding_size)
        repr_new = model.get_encoded_layers(
            data_loader
        )  # (num_layers, batch_size, embeding_size)

        return np.linalg.norm(
            (repr_new - repr_base).cpu().detach().numpy(), ord=2, axis=-1
        )  # (num_layers, batch_size)
