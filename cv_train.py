import torch
from torch import nn
from torch import optim
from datetime import datetime
from tqdm import tqdm

from transformers import AutoImageProcessor

from cv_datasets import (
    get_num_classes,
    get_data_loaders,
)
from cv_models import ResNet18, ResNet34, ViT


def test_step(model, dataloader, device):
    model.eval()
    acc_sum, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            out = model(X)
            _, pred = torch.max(out, 1)

            acc_sum += torch.sum(pred == labels).item()
            total += X.shape[0]

    return acc_sum / total


def training_loop(
    model,
    criterion,
    optimizer,
    trainloader,
    testloader,
    device,
    epochs,
    log_interval=1,
    start_epoch=0,
    scheduler=None,
):
    """Main training loop"""
    start = datetime.now()
    for epoch in tqdm(range(start_epoch, epochs + start_epoch)):
        running_loss = 0.0
        model.train()
        train_accuracy, test_accuracy = 0, 0
        acc_sum, total = 0, 0
        for _, batch in enumerate(trainloader):
            X = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward + backward + optimize
            outputs = model(X)
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
            total += X.shape[0]

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

    print("==> Finished Training ...")
    print("Training completed in: {}s".format(str(datetime.now() - start)))


def train_model(
    model, criterion, optimizer, train_loader, test_loader, device, num_epochs
):
    training_loop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        trainloader=train_loader,
        testloader=test_loader,
        device=device,
        epochs=num_epochs,
    )
    return model


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


if __name__ == "__main__":
    dataset_name = "cifar100"

    device = torch.device("cuda:0")

    n_classes = get_num_classes(dataset_name)

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViT(n_classes, device, processor=processor)
    model.to(device)

    train_loader, test_loader = get_data_loaders(dataset_name, processor, 32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-3)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR()

    training_loop(model, criterion, optimizer, train_loader, test_loader, device, 10, 1)
