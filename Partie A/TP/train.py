import data
import generate_data
import eval
import torch
import os
import sys
import lstm
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

def train_epoch(model, criterion, optimizer, dataset, device):

    model.train()
    avg_loss = 0

    for i, (x, y) in enumerate(dataset):
        # TODO

        if i%1000 == 0:
            print("[iteration: {}/{}]".format(i, len(dataset)))

    return avg_loss/len(train_loader)

def save_model(filename, args, model, criterion):
    torch.save({
        "model": model,
        "criterion": criterion,
        **args
    }, filename)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Watning: CUDA not available, using CPU")
        device = torch.device("cpu")

    board = SummaryWriter(comment="Wavenet")

    seq_len = 64
    batch_size = 32

    train_data = data.MaestroDataset("train.npy", seq_len, 10)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    val_data = data.MaestroDataset("validation.npy", seq_len, 100)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    epoch = 20
    labels = generate_data.read_labels("labels")
    class_number = len(labels)

    args = {
        "class_number": class_number,
        "sequence_length": seq_len,
        "embeding_size": 100,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
    }

    model = lstm.LSTMClassifier(
        class_number=args["class_number"],
        num_layers=args["num_layers"],
        dropout=args["dropout"],
    )

    board.add_graph(model, torch.randint(high=class_number, size=(batch_size, seq_len)))

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_loss = float("inf")
    best_model = None

    for e in range(epoch):
        loss = train_epoch(model, criterion, optimizer, train_loader, device)
        board.add_scalar("Train loss", loss, e)

        save_model("last.pt", args, model, criterion)
        loss, top1, top5 = eval.run(model, criterion, val_loader, device)
        board.add_scalar("Val loss", loss, e)
        board.add_scalar("Val Top1", top1, e)
        board.add_scalar("Val Top5", top5, e)
        print("[epoch: {}/{}] loss: {:.4f}\ttop1: {:.2f}%\ttop5: {:.2f}%".format(e, epoch, loss, top1, top5))

        if loss < best_loss:
            best_loss = loss
            save_model("best.pt", args, model, criterion)

        scheduler.step()