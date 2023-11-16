"""ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import grad
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision.models as models

from transformers import AutoImageProcessor, ViTModel


# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, device, pretrained=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.pretrained = pretrained

        if pretrained:
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4
            self.linear = pretrained.fc
        else:
            self.conv1 = conv3x3(3, 64)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.device = device
        self.num_layers = 6

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, X):
        if self.pretrained:
            return self.pretrained(X)

        conv1 = F.relu(self.bn1(self.conv1(X)))
        out1 = self.layer1(conv1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out6 = out5.view(out5.size(0), -1)
        y = self.linear(out6)
        return y

    def _predict_proba(self, X):
        logits = self(X)
        probs = F.softmax(logits, dim=1)
        return probs

    def predict_proba(self, data_loader):
        self.eval()
        probs_list = []
        with torch.inference_mode():
            for batch in data_loader:
                X, _ = batch
                X = X.to(self.device)
                probs = self._predict_proba(X)
                probs_list.append(probs.cpu())
        return torch.cat(probs_list)

    def _pretrained_feature_list(self, X):
        out_list = []
        out = self.pretrained.conv1(X)
        out = self.pretrained.bn1(out)
        out = self.pretrained.relu(out)
        out = self.pretrained.maxpool(out)
        out_list.append(out)
        out = self.pretrained.layer1(out)
        out_list.append(out)
        out = self.pretrained.layer2(out)
        out_list.append(out)
        out = self.pretrained.layer3(out)
        out_list.append(out)
        out = self.pretrained.layer4(out)
        out_list.append(out)
        out = self.pretrained.avgpool(out)
        out = torch.flatten(out, 1)
        y = self.pretrained.fc(out)
        out_list.append(y)

        return y, out_list

    # function to extact the multiple features
    def feature_list(self, X):
        if self.pretrained:
            return self._pretrained_feature_list(X)

        out_list = []
        out = F.relu(self.bn1(self.conv1(X)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        out_list.append(y)
        return y, out_list

    def get_grad_layers(self, data_loader, n_estimators=10, estimator=True):
        self.eval()
        layer_norms = []
        cnt = 0
        for batch in data_loader:
            X, _ = batch
            X = X.to(self.device)

            self.zero_grad()
            _, layers = self.feature_list(X)
            norms = []
            # J: I think it makes sense to start from 0 for ResNet
            for i in range(0, len(layers) - 1):
                emb_X = layers[i]  # (batch_size, squence_length, embeding_size)
                emb_Y = layers[i + 1]  # (batch_size, squence_length, embeding_size)

                ests = []
                for _ in range(n_estimators):
                    v = torch.randn(emb_Y.shape).to(self.device)
                    vjp = grad((v * emb_Y).sum(), emb_X, retain_graph=True)[0]
                    vjp = vjp.reshape(vjp.shape[0], -1)
                    fro_norms = vjp.norm(p="fro", dim=1) ** 2
                    ests.append(fro_norms.cpu())  # (batch_size)

                norm_ests = torch.stack(ests, dim=1)  # (batch_size, n_estimators)
                norms.append(norm_ests.mean(dim=1))  # (batch_size)

            layer_norms.append(torch.stack(norms, dim=0))  # (num_layers-1, batch_size)

        self.train()
        return torch.cat(layer_norms, dim=1)  # (num_layers-1, batch_size)

    def get_grad_embedding_norms(self, data_loader):
        self.eval()

        criterion = nn.CrossEntropyLoss(reduction="none")
        grad_embeddings = []

        for X, _ in data_loader:
            self.zero_grad()

            X = X.to(self.device)
            logits = self(X)

            if logits.shape[1] == 1:
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
                embedding = grad(l, self.linear.weight, retain_graph=True)[0]
                grad_embeddings.append(
                    embedding.flatten(start_dim=1).cpu().norm(p="fro", dim=1)
                )

        return torch.stack(grad_embeddings).cpu().detach()

    def get_batch_encoded_layers(self, X):
        self.eval()
        with torch.no_grad():
            _, layers = self.feature_list(X)
            embeddings = [layer.flatten(start_dim=1).cpu().detach() for layer in layers]
            return embeddings


def ResNet18(num_c, device):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_c, device=device)


def ResNet34(num_c, device):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_c, device=device)


def ResNet50(num_c, device):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_c, device=device)


def ResNet18PT(num_c, device):
    return ResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_c,
        device=device,
        pretrained=models.resnet18(pretrained=True),
    )


def ResNet34PT(num_c, device):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_c,
        device=device,
        pretrained=models.resnet34(pretrained=True),
    )


def ResNet50PT(num_c, device):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_c,
        device=device,
        pretrained=models.resnet50(pretrained=True),
    )


class ViT(nn.Module):
    def __init__(
        self,
        output_dim,
        device,
        processor,
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.transformer = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vectorizer = processor
        config = self.transformer.config
        self.hidden_size = config.hidden_size
        # last layer has to be named 'out' for calculating gradient embedding
        self.out = nn.Linear(config.hidden_size, output_dim)
        self.to(self.device)
        # Xavier initalization
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, X):
        out = self.transformer(X)
        logits = self.out(out.last_hidden_state[:, 0, :])
        return logits

    # def forward_react(self, X, t):
    #     X_vec = self.vectorizer(X)
    #     out = self.transformer(**X_vec)
    #     activations = out.last_hidden_state[:, 0, :]

    #     t = torch.fill(activations, t)
    #     X_react = torch.clip(activations, max=t)

    #     logits = self.out(X_react)
    #     return logits

    def forward_ash(self, X, p=0.9):
        out = self.transformer(X)
        activations = out.last_hidden_state[:, 0, :]

        X_abs = activations
        t = torch.quantile(X_abs, p, dim=1)
        t = t.reshape(-1, 1).repeat(1, X_abs.shape[1])
        s1 = X_abs.sum(dim=1)

        X_ash = torch.where(activations < t, 0, activations)
        s2 = X_ash.sum(dim=1)

        m = s1 / s2
        X_ash = X_ash * torch.exp(m.reshape(-1, 1).repeat(1, X_ash.shape[1]))

        logits = self.out(X_ash)
        return logits

    def get_encoded(self, X, batch_size=128):
        self.eval()
        with torch.no_grad():
            embeddings = []
            for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
                X_vec = self.vectorizer(X_iter)  # .to("cpu")
                out = self.transformer(**X_vec)
                embeddings.extend(out.last_hidden_state[:, 0, :].tolist())
        self.train()
        return torch.tensor(embeddings)  # (batch_size, embedding_size)

    def get_encoded_layers(self, X, batch_size=128):
        self.eval()
        with torch.no_grad():
            embeddings = []
            for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
                X_vec = self.vectorizer(X_iter)  # .to("cpu")
                out = self.transformer(**X_vec, output_hidden_states=True)
                embeddings.append(
                    torch.stack(out.hidden_states[1:], dim=0)[:, :, 0, :].cpu()
                )
        self.train()
        return torch.cat(
            embeddings, dim=1
        )  # (num_layers (12), batch_size, embeding_size)

    def get_grad_layers(self, data_loader, n_estimators=15, estimator=True):
        self.eval()
        layer_norms = []
        cnt = 0
        for batch in data_loader:
            self.zero_grad()
            X = batch["pixel_values"]
            X = X.to(self.device)
            out = self.transformer(X, output_hidden_states=True)
            norms = []
            for i in range(1, len(out.hidden_states) - 1):
                emb_X = out.hidden_states[
                    i
                ]  # (batch_size, squence_length, embeding_size)
                emb_Y = out.hidden_states[
                    i + 1
                ]  # (batch_size, squence_length, embeding_size)

                if estimator:
                    ests = []
                    for n in range(n_estimators):
                        v = torch.randn((emb_Y.shape[0], emb_Y.shape[2])).to(
                            self.device
                        )
                        est = grad(
                            (emb_Y[:, 0, :] * v).sum(), emb_X, retain_graph=True
                        )[0][
                            :, 0, :
                        ]  # (batch_size, embedding_size)
                        w = torch.randn((emb_Y.shape[0], emb_Y.shape[2])).to(
                            self.device
                        )
                        ests.append(((est * w).sum(dim=1) ** 2).cpu())  # (batch_size)

                    norm_ests = torch.stack(ests, dim=1)  # (batch_size, n_estimators)
                    norms.append(norm_ests.mean(dim=1))  # (batch_size)

                else:
                    grads = [
                        grad(emb_Y[:, 0, j].sum(), emb_X, retain_graph=True)[0][
                            :, 0, :
                        ].cpu()
                        for j in range(emb_Y.shape[2])
                    ]
                    norm_ests = torch.cat(
                        grads, dim=1
                    )  # (batch_size, embeding_size*embeding_size)
                    norms.append((norm_ests**2).sum(dim=1))  # (batch_size)

            layer_norms.append(
                torch.stack(norms, dim=0)
            )  # (num_layers-1 (11), batch_size)

        self.train()
        return torch.cat(layer_norms, dim=1)  # (num_layers-1 (11), batch_size)

    # def get_grad_embedding(
    #     self, X, criterion, batch_size=32, grad_embedding_type="bias_linear"
    # ):
    #     self.eval()
    #     criterion = copy.deepcopy(criterion)
    #     criterion.reduction = "none"
    #     criterion.to(self.device)
    #     grad_embeddings = []

    #     for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
    #         self.zero_grad()

    #         logits = self(X_iter)

    #         if logits.shape[1] == 1 or self.multilabel:
    #             logits = logits.ravel()
    #             y = torch.as_tensor(
    #                 logits > 0,
    #                 dtype=torch.float,
    #                 device=self.device,
    #             )
    #         else:
    #             y = torch.argmax(logits, dim=1)

    #         loss = criterion(logits, y)
    #         for l in loss:
    #             if grad_embedding_type == "bias":
    #                 embedding = grad(l, self.out.bias, retain_graph=True)[0]
    #             elif grad_embedding_type == "linear":
    #                 embedding = grad(l, self.out.weight, retain_graph=True)[0]
    #             elif grad_embedding_type == "bias_linear":
    #                 weight_emb = grad(l, self.out.weight, retain_graph=True)[0]
    #                 bias_emb = grad(l, self.out.bias, retain_graph=True)[0]
    #                 embedding = torch.cat(
    #                     (weight_emb, bias_emb.unsqueeze(dim=1)), dim=1
    #                 )
    #             else:
    #                 raise ValueError(
    #                     f"Grad embedding type '{grad_embedding_type}' not supported."
    #                     "Viable options: 'bias', 'linear', or 'bias_linear'"
    #                 )
    #             grad_embeddings.append(embedding.flatten().to("cpu"))

    #     self.train()
    #     return (
    #         torch.stack(grad_embeddings).cpu().detach().numpy()
    #     )  # (batch_size, embedding_size)

    # def train_loop(
    #     self,
    #     X,
    #     y,
    #     criterion,
    #     X_val=None,
    #     y_val=None,
    #     num_epochs=10,
    #     batch_size=64,
    #     lr=2e-5,
    #     cartography=True,
    #     viz=False,
    #     X_ood=None,
    #     X_shifted=None,
    #     viz_batch=128,
    #     optim=torch.optim.Adam,
    #     **kwargs,
    # ):
    #     optimizer = optim(self.parameters(), lr=lr)
    #     cartography_data = (
    #         {"train": [], "id": [], "ood": [], "shifted": []} if cartography else None
    #     )
    #     viz_data = {"train": [], "id": [], "ood": [], "shifted": []} if viz else None
    #     self.train()
    #     for i in range(num_epochs):
    #         print(f"Epoch: {i+1}")
    #         # 5 step training routine
    #         # ------------------------------------------
    #         for X_iter, y_iter in self.iterator(batch_size, X, y):
    #             # 1) zero the gradients
    #             optimizer.zero_grad()

    #             # 2) compute the output
    #             y_pred = self(X_iter)
    #             y_iter = torch.tensor(y_iter).to(self.device)
    #             if y_pred.shape[1] == 1 or self.multilabel:
    #                 y_iter = y_iter.float()

    #             # 3) compute the loss
    #             loss = criterion(y_pred.squeeze(), y_iter.squeeze())

    #             # 4) use loss to produce gradients
    #             loss.backward()

    #             # 5) use optimizer to take gradient step
    #             optimizer.step()

    #         if cartography:
    #             cartography_data["train"].append(self.predict_proba(X))
    #             if X_val:
    #                 cartography_data["id"].append(self.predict_proba(X_val))
    #             if X_ood:
    #                 cartography_data["ood"].append(self.predict_proba(X_ood))
    #             if X_shifted:
    #                 cartography_data["shifted"].append(self.predict_proba(X_shifted))
    #         if X_val and y_val:
    #             self.validate(X_val, y_val)
    #         if viz:
    #             viz_data["train"].append(
    #                 self.get_encoded_layers(X, batch_size=viz_batch)
    #             )
    #             if X_val:
    #                 viz_data["id"].append(
    #                     self.get_encoded_layers(X_val, batch_size=viz_batch)
    #                 )
    #             if X_ood:
    #                 viz_data["ood"].append(
    #                     self.get_encoded_layers(X_ood, batch_size=viz_batch)
    #                 )
    #             if X_shifted:
    #                 viz_data["shifted"].append(
    #                     self.get_encoded_layers(X_ood, batch_size=X_shifted)
    #                 )

    #     return viz_data, cartography_data

    # def validate(self, X_val, y_val):
    #     pred = self.predict(X_val)
    #     print(f"Accuracy: {accuracy_score(y_val, pred)}")
    #     print(f"F1 micro: {f1_score(y_val, pred, average='micro')}")
    #     print(f"F1 macro: {f1_score(y_val, pred, average='macro')}")
    #     if len(set(y_val)) == 2:
    #         print(f"F1 binary: {f1_score(y_val, pred, average='binary')}")
    #     print(f"Confusion matrix: {confusion_matrix(y_val, pred)}")
    #     return

    # def _predict_proba(self, X):
    #     self.eval()
    #     y_pred = []
    #     for batch in data_loader:
    #         X = batch["pixel_values"].to(self.device)
    #         y_pred.extend(self.forward(X).tolist())
    #     y_pred = torch.tensor(y_pred)
    #     if self.output_dim == 1:
    #         y_pred = torch.sigmoid(y_pred)
    #         y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
    #     elif self.multilabel:
    #         y_pred = torch.sigmoid(y_pred)
    #     else:
    #         y_pred = F.softmax(y_pred, dim=1)
    #     self.train()
    #     return y_pred

    def predict_proba(self, data_loader):
        self.eval()
        y_pred = []
        for batch in data_loader:
            X = batch["pixel_values"].to(self.device)
            y_pred.extend(self.forward(X).tolist())
        y_pred = torch.tensor(y_pred)
        y_pred = F.softmax(y_pred, dim=1)

        return y_pred

    # def loss(self, X, y, criterion, batch_size=128):
    #     criterion = copy.deepcopy(criterion)
    #     criterion.reduction = "none"
    #     criterion.to(self.device)
    #     with torch.no_grad():
    #         loss = []
    #         for X_iter, y_iter in self.iterator(batch_size, X, y):
    #             y_pred = self(X_iter)
    #             y_iter = torch.tensor(y_iter).to(self.device)
    #             if y_pred.shape[1] == 1 or self.multilabel:
    #                 y_iter = y_iter.float()
    #             loss.extend(criterion(y_pred.squeeze(), y_iter.squeeze()).tolist())
    #     return loss

    # def predict_proba_dropout(self, X):
    #     with torch.no_grad():
    #         y_pred = []
    #         for X_iter, y_iter in self.iterator(32, X, None, shuffle=False):
    #             y_pred.extend(self.forward(X_iter).tolist())
    #         y_pred = torch.tensor(y_pred)
    #         if self.output_dim == 1:
    #             y_pred = torch.sigmoid(y_pred)
    #             y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
    #         elif self.multilabel:
    #             y_pred = torch.sigmoid(y_pred)
    #         else:
    #             y_pred = F.softmax(y_pred, dim=1)
    #     return y_pred.cpu().numpy()

    # def iterator(self, batch_size, X, y=None, shuffle=True):
    #     if shuffle:
    #         X_sort = sorted(X, key=lambda x: len(x.split()), reverse=True)
    #         y_sort = (
    #             [
    #                 y
    #                 for y, _ in sorted(
    #                     zip(y, X), key=lambda pair: len(pair[1].split()), reverse=True
    #                 )
    #             ]
    #             if y
    #             else None
    #         )
    #     else:
    #         X_sort = X
    #         y_sort = y
    #     X_batched = []
    #     y_batched = []
    #     for i in range(math.ceil(len(X) / batch_size)):
    #         y_batched.append(
    #             y_sort[i * batch_size : (i + 1) * batch_size] if y else None
    #         )
    #         X_batched.append(X_sort[i * batch_size : (i + 1) * batch_size])
    #     inds = list(range(len(X_batched)))
    #     if shuffle:
    #         random.shuffle(inds)
    #     for i in inds:
    #         yield X_batched[i], y_batched[i]
