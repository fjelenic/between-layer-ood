import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F


class LeastConfidentQuant():
    name = "least_confident"
    
    def quantify(self, X_eval, model, **kwargs):
        probs = model.predict_proba(X_eval)
        max_probs = np.max(probs, axis=1)
        return -max_probs
        
        
class EntropyQuant():
    name = "entropy"
    
    def quantify(self, X_eval, model, **kwargs):
        probs = model.predict_proba(X_eval)
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        return entropies
    
    
class EnergyQuant():
    name = "energy"
    
    def quantify(self, X_eval, model, **kwargs):
        logits = []
        for X_iter, _ in model.iterator(16, X_eval, None, shuffle=False):
            logits.extend(model.forward(X_iter).tolist())
        logits = np.array(logits)
        return -np.log(np.sum(np.exp(logits), axis=1))
    

class MCDropoutQuant():
    name = "MC"
    
    def quantify(self, X_eval, model, k=30, **kwargs):
        probs_dropout = model.predict_proba_dropout(X_eval)  # (batch_size, num_labels)
        
        entropies_dropout = []
        for _ in range(k-1):
            probs_dropout += model.predict_proba_dropout(X_eval)

        probs_dropout /= k
        probs_dropout = np.clip(probs_dropout, a_min=1e-6, a_max=None)
        entropies_dropout = np.sum(-probs_dropout * np.log(probs_dropout), axis=1)
            
        return entropies_dropout
    
    
class GradQuant():
    name = "gradient"
    
    def quantify(self, X_eval, model, criterion, batch_size=64, **kwargs):
        grad_embedding = model.get_grad_embedding(X_eval, criterion, batch_size=batch_size, grad_embedding_type="linear") # (batch_size, embedding_size)
        return np.linalg.norm(grad_embedding, ord=2, axis=1)  
        
        
class BLOODQuant():
    name = "BLOOD"
    
    def quantify(self, X_eval, model, batch_size=64, path=False, estimator=True, **kwargs):
        norms = model.get_grad_layers(X_eval, batch_size=batch_size, estimator=estimator)  # (num_layers-1 (11), batch_size)
        
        if path:
            return norms
        else:
            return norms.mean(axis=0)
        
    
class MahalanobisQuant():  
    name = "mahalanobis"
    
    def quantify(self, X_eval, X_anchor, y_anchor, model, batch_size=64, **kwargs):
        anchor_emb = model.get_encoded(X_anchor, batch_size=batch_size)  # (batch_size, embeding_size)
        eval_emb = model.get_encoded(X_eval, batch_size=batch_size)  # (batch_size, embeding_size)
        
        classes = list(set(y_anchor))
        mi = []
        
        for c in classes:
            class_anchor = anchor_emb[[i for i in range(len(y_anchor)) if y_anchor[i] == c]]
            mi.append(np.expand_dims(class_anchor.mean(axis=0), axis=1))
        
        sigma = np.zeros((anchor_emb.shape[1], anchor_emb.shape[1]))
        for i, c in enumerate(classes):
            for emb in anchor_emb[[i for i in range(len(y_anchor)) if y_anchor[i] == c]]:
                diff = np.expand_dims(emb, axis=1) - mi[i]
                sigma += diff @ diff.T
        sigma_inv = np.linalg.inv(sigma/len(anchor_emb))
        
        uncertainties = []
        for emb in eval_emb:
            class_uncertainties = []
            emb_exp = np.expand_dims(emb, axis=1)
            for mi_c in mi:
                diff = emb_exp - mi_c
                class_uncertainties.append((-(diff.T @ sigma_inv @ diff)).squeeze())
            uncertainties.append(class_uncertainties)
        
        return -np.min(np.array(uncertainties), axis=1)
     
    
class EnsambleQuant():
    name = "ensamble"
    
    def quantify(self, X_eval, models, **kwargs):
        probs_ens = models[0].predict_proba_dropout(X_eval)  # (batch_size, num_labels)
        
        for model in models[1:]:
            probs_ens += model.predict_proba(X_eval)  # (batch_size, num_labels)

        probs_ens /= len(models)
        probs_ens = np.clip(probs_ens, a_min=1e-6, a_max=None)
        entropies_ens = np.sum(-probs_ens * np.log(probs_ens), axis=1)
            
        return entropies_ens
    
class TemperatureScalingQuant():
    name = "temperature"
    
    def quantify(self, X_eval, model, T, **kwargs):
        logits = []
        with torch.no_grad():
            model.eval()
            for X_iter, _ in model.iterator(128, X_eval, None, shuffle=False):
                logits.extend(model.forward(X_iter).tolist())
            logits = torch.tensor(logits)
            model.train()
        logits_scaled = logits/T
        
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits_scaled)
            probs = torch.cat([1.0 - probs, probs], dim=1)
        else:
            probs = F.softmax(logits_scaled, dim=1)
        
        probs = probs.cpu().numpy()
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
            
        return entropies
    
    
class RepresentationChangeQuant():
    name = "repr_change"
    
    def quantify(self, X_eval, model, model_orig, **kwargs):
        reprs_orig = model_orig.get_encoded_layers(X_eval)  # (num_layers (12), batch_size, embeding_size)
        reprs = model.get_encoded_layers(X_eval)  # (num_layers (12), batch_size, embeding_size)
        
        return np.linalg.norm((reprs_orig - reprs).cpu().detach().numpy(), ord=2, axis=-1)  # (num_layers (12), batch_size)
        
        
    
    
    
    
    