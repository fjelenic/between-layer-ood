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
    def __init__(self, block, num_blocks, num_classes, device):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.device = device

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, X):
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

    # function to extact the multiple features
    def feature_list(self, X):
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
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

    def get_grad_layers(self, data_loader, n_estimators=10, estimator=True):
        self.eval()
        layer_norms = []
        cnt = 0
        for batch in data_loader:
            X, _ = batch
            X = X.to(self.device)

            self.zero_grad()
            out, layers = self.feature_list(X)
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
    

    def get_grad_embedding(self, data_loader):
        self.eval()
        
        criterion = nn.CrossEntropyLoss(reduction=None)
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
                grad_embeddings.append(embedding.flatten(start_dim=1).cpu())

        return torch.stack(grad_embeddings).cpu().detach()  # (batch_size, embedding_size)
       
    def get_encoded_layers(self, data_loader):
        self.eval()
        with torch.no_grad():
            embeddings = []
            for X, _ in data_loader:
                X = X.to(self.device)
                out_list = self.feature_list(X)
                embeddings.append(torch.stack(out_list, dim=0).flatten(start_dim=2).cpu())   
        return torch.cat(embeddings, dim=1)  # (num_layers, batch_size, embeding_size)


def ResNet18(num_c, device):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_c, device=device)


def ResNet34(num_c, device):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_c, device=device)

def ResNet50(num_c, device):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_c, device=device)

