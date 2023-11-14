import torch
from torch import nn
from torch import optim
from datetime import datetime
from tqdm import tqdm

from cv_datasets import get_dataset, get_data_loader, get_num_classes
from cv_models import ResNet18, ResNet34


def test_step(model, dataloader, device):
    model.eval()
    acc_sum, total = 0, 0
    with torch.no_grad():
        for batch, labels in dataloader:
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(batch)
            _, pred = torch.max(out, 1)

            acc_sum += torch.sum(pred == labels).item()
            total += batch.shape[0]

    return acc_sum / total


def training_loop(
    model,
    criterion,
    optimizer,
    trainloader,
    testloader,
    device,
    epochs,
    log_interval,
    start_epoch=0,
    scheduler=None,
):
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

    print("==> Finished Training ...")
    print("Training completed in: {}s".format(str(datetime.now() - start)))


if __name__ == "__main__":
    root = "cv_data"
    dataset_name = "svhn"
    train_set = get_dataset(dataset_name, root, train=True)
    test_set = get_dataset(dataset_name, root, train=False)

    device = torch.device("cuda:0")

    train_loader = get_data_loader(
        train_set,
        batch_size=32,
        num_workers=1,
        shuffle=True,
    )

    test_loader = get_data_loader(
        test_set,
        batch_size=32,
        num_workers=1,
    )

    # for batch in train_loader:
    #     input, target = batch
    #     print(input.shape)
    #     break

    n_classes = get_num_classes(dataset_name)
    model = ResNet34(n_classes)
    model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR()

    training_loop(model, criterion, optimizer, train_loader, test_loader, device, 10, 1)
