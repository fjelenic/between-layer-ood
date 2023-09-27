import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import random
import math
from torch.autograd import grad
import copy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

transformer_architecture = {"ELECTRA":"google/electra-base-discriminator",
                           "RoBERTa":"roberta-base"}

def get_transformer_vectorizer(arh, device):
    tokenizer = AutoTokenizer.from_pretrained(arh)

    def func(x, x_pair=None):
        if x_pair:
            return tokenizer(x, x_pair, truncation=True, padding="longest", return_tensors="pt").to(device)
        else:
            return tokenizer(x, truncation=True, padding="longest", return_tensors="pt").to(device)

    return func

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        name,
        output_dim,
        device=torch.device(GPU if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.name = name
        self.device = device
        self.multilabel = False
        self.output_dim = output_dim
        self.transformer = AutoModel.from_pretrained(transformer_architecture[name])
        self.vectorizer = get_transformer_vectorizer(transformer_architecture[name], self.device)
        config = self.transformer.config
        self.hidden_size = config.hidden_size
        # last layer has to be named 'out' for calculating gradient embedding
        self.out = nn.Linear(config.hidden_size, output_dim)
        self.to(self.device)
        # Xavier initalization
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, X):
        X_vec = self.vectorizer(X)
        out = self.transformer(**X_vec)
        logits = self.out(out.last_hidden_state[:, 0, :])
        return logits

    def get_encoded(self, X, batch_size=128):
        self.eval()
        with torch.no_grad():
            embeddings = []
            for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
                X_vec = self.vectorizer(X_iter)#.to("cpu")
                out = self.transformer(**X_vec)
                embeddings.extend(out.last_hidden_state[:, 0, :].tolist())
        self.train()
        return torch.tensor(embeddings)  # (batch_size, embedding_size)
    
    def get_encoded_layers(self, X, batch_size=128):
        self.eval()
        with torch.no_grad():
            embeddings = []
            for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
                X_vec = self.vectorizer(X_iter)#.to("cpu")
                out = self.transformer(**X_vec, output_hidden_states=True)
                embeddings.append(torch.stack(out.hidden_states[1:], dim=0)[:,:,0,:].cpu())   
        self.train()
        return torch.cat(embeddings, dim=1)  # (num_layers (12), batch_size, embeding_size)
    
    def get_grad_layers(self, X, batch_size=64, n_estimators=50, estimator=True): #50 estimators is base
        self.eval()
        layer_norms = []
        cnt = 0
        for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
            self.zero_grad()
            X_vec = self.vectorizer(X_iter)
            out = self.transformer(**X_vec, output_hidden_states=True)
            norms = []
            for i in range(1, len(out.hidden_states)-1):
                emb_X = out.hidden_states[i]  # (batch_size, squence_length, embeding_size)
                emb_Y = out.hidden_states[i+1]  # (batch_size, squence_length, embeding_size)
                
                if estimator:
                    ests = []
                    for n in range(n_estimators):
                        v = torch.randn((emb_Y.shape[0], emb_Y.shape[2])).to(self.device)
                        est = grad((emb_Y[:,0,:]*v).sum(), emb_X, retain_graph=True)[0][:,0,:]  # (batch_size, embedding_size)
                        w = torch.randn((emb_Y.shape[0], emb_Y.shape[2])).to(self.device)
                        ests.append(((est*w).sum(dim=1)**2).cpu())  # (batch_size)

                    norm_ests = torch.stack(ests, dim=1)  # (batch_size, n_estimators)   
                    norms.append(norm_ests.mean(dim=1))  # (batch_size)
                    
                else:
                    grads = [grad(emb_Y[:,0,j].sum(), emb_X, retain_graph=True)[0][:,0,:].cpu() for j in range(emb_Y.shape[2])]
                    norm_ests = torch.cat(grads, dim=1)  # (batch_size, embeding_size*embeding_size)
                    norms.append((norm_ests**2).sum(dim=1))  # (batch_size)
            
            layer_norms.append(torch.stack(norms, dim=0))  # (num_layers-1 (11), batch_size)
                
        self.train()
        return torch.cat(layer_norms, dim=1)  # (num_layers-1 (11), batch_size)
    
    def get_grad_embedding(self, X, criterion, batch_size=32, grad_embedding_type="bias_linear"):
        self.eval()
        criterion = copy.deepcopy(criterion)
        criterion.reduction = "none"
        criterion.to(self.device)
        grad_embeddings = []

        for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
            self.zero_grad()
            
            logits = self(X_iter)

            if logits.shape[1] == 1 or self.multilabel:
                logits = logits.ravel()
                y = torch.as_tensor(
                    logits > 0,
                    dtype=torch.float,
                    device=self.device,
                )
            else:
                y = torch.argmax(logits, dim=1)
                
            loss = criterion(logits, y)
            for l in loss:
                if grad_embedding_type == "bias":
                    embedding = grad(l, self.out.bias, retain_graph=True)[0]
                elif grad_embedding_type == "linear":
                    embedding = grad(l, self.out.weight, retain_graph=True)[0]
                elif grad_embedding_type == "bias_linear":
                    weight_emb = grad(l, self.out.weight, retain_graph=True)[0]
                    bias_emb = grad(l, self.out.bias, retain_graph=True)[0]
                    embedding = torch.cat(
                        (weight_emb, bias_emb.unsqueeze(dim=1)), dim=1
                    )
                else:
                    raise ValueError(
                        f"Grad embedding type '{grad_embedding_type}' not supported."
                        "Viable options: 'bias', 'linear', or 'bias_linear'"
                    )
                grad_embeddings.append(embedding.flatten().to("cpu"))

        self.train()
        return torch.stack(grad_embeddings).cpu().detach().numpy()  # (batch_size, embedding_size)
       
    
    def train_loop(self, X, y, criterion, X_val=None, y_val=None, num_epochs=10, batch_size=64, lr=2e-5, cartography=True, viz=False, X_ood=None, X_shifted=None, viz_batch=128, optim=torch.optim.Adam, **kwargs):
        optimizer = optim(self.parameters(), lr=lr)
        cartography_data = {"train": [], "id": [], "ood": [], "shifted": []} if cartography else None
        viz_data = {"train":[], "id":[], "ood":[], "shifted": []} if viz else None
        self.train()
        for i in range(num_epochs):
            print(f"Epoch: {i+1}")
            # 5 step training routine
            # ------------------------------------------
            for X_iter, y_iter in self.iterator(batch_size, X, y):
                # 1) zero the gradients
                optimizer.zero_grad()

                # 2) compute the output
                y_pred = self(X_iter)
                y_iter = torch.tensor(y_iter).to(self.device)
                if y_pred.shape[1] == 1 or self.multilabel:
                    y_iter = y_iter.float()

                # 3) compute the loss
                loss = criterion(y_pred.squeeze(), y_iter.squeeze())

                # 4) use loss to produce gradients
                loss.backward()

                # 5) use optimizer to take gradient step
                optimizer.step()
            
            if cartography:
                cartography_data["train"].append(self.predict_proba(X))
                if X_val:
                    cartography_data["id"].append(self.predict_proba(X_val))
                if X_ood:
                    cartography_data["ood"].append(self.predict_proba(X_ood))
                if X_shifted:
                    cartography_data["shifted"].append(self.predict_proba(X_shifted))
            if X_val and y_val:
                self.validate(X_val, y_val)
            if viz:
                viz_data["train"].append(self.get_encoded_layers(X, batch_size=viz_batch))
                if X_val:
                    viz_data["id"].append(self.get_encoded_layers(X_val, batch_size=viz_batch))
                if X_ood:
                    viz_data["ood"].append(self.get_encoded_layers(X_ood, batch_size=viz_batch))
                if X_shifted:
                    viz_data["shifted"].append(self.get_encoded_layers(X_ood, batch_size=X_shifted))
        
        return viz_data, cartography_data
       
    def validate(self, X_val, y_val):
        pred = self.predict(X_val)
        print(f"Accuracy: {accuracy_score(y_val, pred)}")
        print(f"F1 micro: {f1_score(y_val, pred, average='micro')}")
        print(f"F1 macro: {f1_score(y_val, pred, average='macro')}")
        if len(set(y_val)) == 2:
            print(f"F1 binary: {f1_score(y_val, pred, average='binary')}")
        print(f"Confusion matrix: {confusion_matrix(y_val, pred)}")
        return
            
    def _predict_proba(self, X):
        self.eval()
        y_pred = []
        for X_iter, y_iter in self.iterator(128, X, None, shuffle=False):
            y_pred.extend(self.forward(X_iter).tolist())
        y_pred = torch.tensor(y_pred)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
        elif self.multilabel:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax(y_pred, dim=1)
        self.train()
        return y_pred

    def _predict(self,X):
        self.eval()
        y_pred = []
        for X_iter, y_iter in self.iterator(128, X, None, shuffle=False):
            y_pred.extend(self.forward(X_iter).tolist())
        y_pred = torch.tensor(y_pred)
        if self.output_dim == 1 or self.multilabel:
            out = torch.as_tensor(
                y_pred > 0,
                dtype=torch.long,
                device=self.device,
            )
        else:
            out = torch.argmax(y_pred, dim=1)

        self.train()
        return out

    def predict_proba(self, X):
        with torch.no_grad():
            y_pred = self._predict_proba(X)
            return y_pred.cpu().numpy()

    def predict(self, X):
        with torch.no_grad():
            out = self._predict(X)
            return out.cpu().numpy()
        
    def loss(self, X, y, criterion, batch_size=128):
        criterion = copy.deepcopy(criterion)
        criterion.reduction = "none"
        criterion.to(self.device)
        with torch.no_grad():
            loss = []
            for X_iter, y_iter in self.iterator(batch_size, X, y):
                y_pred = self(X_iter)
                y_iter = torch.tensor(y_iter).to(self.device)
                if y_pred.shape[1] == 1 or self.multilabel:
                    y_iter = y_iter.float()
                loss.extend(criterion(y_pred.squeeze(), y_iter.squeeze()).tolist())
        return loss
    
    def predict_proba_dropout(self, X):
        with torch.no_grad():
            y_pred = []
            for X_iter, y_iter in self.iterator(32, X, None, shuffle=False):
                y_pred.extend(self.forward(X_iter).tolist())
            y_pred = torch.tensor(y_pred)
            if self.output_dim == 1:
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
            elif self.multilabel:
                y_pred = torch.sigmoid(y_pred)
            else:
                y_pred = F.softmax(y_pred, dim=1)
        return y_pred.cpu().numpy()
        
    def iterator(self, batch_size, X, y=None, shuffle=True):
        if shuffle:
            X_sort = sorted(X, key=lambda x:len(x.split()), reverse=True)
            y_sort = [y for y, _ in sorted(zip(y, X), key=lambda pair: len(pair[1].split()), reverse=True)] if y else None
        else:
            X_sort = X
            y_sort = y
        X_batched =[]
        y_batched = []
        for i in range(math.ceil(len(X) / batch_size)):
                    y_batched.append(y_sort[i*batch_size:(i+1)*batch_size] if y else None)
                    X_batched.append(X_sort[i*batch_size:(i+1)*batch_size])
        inds = list(range(len(X_batched)))
        if shuffle:
            random.shuffle(inds)
        for i in inds:
            yield X_batched[i], y_batched[i]

            
            
            
