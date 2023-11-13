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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(conv1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out6 = out5.view(out5.size(0), -1)
        y = self.linear(out6)
        return y, [conv1, out1, out2, out3, out4, out5]

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
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

    def get_grad_layers(
        self, X, batch_size=64, n_estimators=50, estimator=True
    ):  # 50 estimators is base
        self.eval()
        layer_norms = []
        cnt = 0
        for X_iter, y_iter in self.iterator(batch_size, X, None, shuffle=False):
            self.zero_grad()
            X_vec = self.vectorizer(X_iter)
            out, layers = self(**X_vec, output_hidden_states=True)
            norms = []
            # J: I think it makes sense to start from 0 in the case of a CNN
            for i in range(0, len(layers) - 1):
                emb_X = layers[i]  # (batch_size, squence_length, embeding_size)
                emb_Y = layers[i + 1]  # (batch_size, squence_length, embeding_size)

                ests = []
                for n in range(n_estimators):
                    emb_Y_view = emb_Y.flatten(start_dim=1, end_dim=-1)
                    emb_X_view = emb_X.flatten(start_dim=1, end_dim=-1)
                    v = torch.randn(emb_Y_view.shape).to(self.device)
                    est = grad((emb_Y_view * v).sum(), emb_X_view, retain_graph=True)[0]
                    w = torch.randn(emb_Y_view.shape).to(self.device)
                    ests.append(((est * w).sum(dim=1) ** 2).cpu())  # (batch_size)

                norm_ests = torch.stack(ests, dim=1)  # (batch_size, n_estimators)
                norms.append(norm_ests.mean(dim=1))  # (batch_size)

            layer_norms.append(
                torch.stack(norms, dim=0)
            )  # (num_layers-1 (11), batch_size)

        self.train()
        return torch.cat(layer_norms, dim=1)  # (num_layers-1 (11), batch_size)

    def training_loop(
        model,
        criterion,
        optimizer,
        trainloader,
        testloader,
        device,
        epochs,
        log_interval,
        checkpoint_interval=10,
        checkpoint_fn=None,
        start_epoch=0,
        scheduler=None,
    ):
        # TODO: adjust training loop
        """Train a model with data"""
        start = datetime.now()
        for epoch in tqdm(range(start_epoch, epochs + start_epoch)):
            running_loss = 0.0
            model.train()
            train_accuracy, test_accuracy = 0, 0
            acc_sum, total = 0, 0
            for _, (batch, labels) in enumerate(trainloader):
                batch = batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # forward + backward + optimize
                outputs = model(batch)
                loss = criterion(outputs, labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()

                # optimizer step
                optimizer.step()

                # training statistics
                running_loss += loss.item()
                # training accuracy
                _, pred = torch.max(outputs.detach(), 1)
                acc_sum += torch.sum(pred == labels).item()
                total += batch.shape[0]

            # normalizing the loss by the total number of train batches
            running_loss /= len(trainloader)

            # scheduler
            if scheduler is not None:
                scheduler.step()

            # logging
            if (epoch) % log_interval == 0:
                # calculate training and test set accuracy of the existing model
                test_accuracy = test_step(model, testloader, device)
                train_accuracy = acc_sum / total
                print(
                    "Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(
                        epoch + 1,
                        epochs,
                        running_loss,
                        train_accuracy * 100,
                        test_accuracy * 100,
                    )
                )

            # save checkpoint
            if (epoch + 1) % checkpoint_interval == 0 and checkpoint_fn is not None:
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": loss,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                }
                checkpoint_fn(epoch, state)

        print("==> Finished Training ...")
        print("Training completed in: {}s".format(str(datetime.now() - start)))


def ResNet18(num_c):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_c)


def ResNet34(num_c):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_c)


def ResNet50(num_c):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_c)


def ResNet101(num_c):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_c)


def ResNet152(num_c):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_c)
