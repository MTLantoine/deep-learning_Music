import argparse
import torch
import data
import eval
from torch.utils.data import DataLoader

def load_model(filename):
    args = torch.load(filename)
    model = args["model"]
    criterion = args["criterion"]
    return model, criterion, args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Model")

    parser.add_argument("model")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Watning: CUDA not available, using CPU")
        device = torch.device("cpu")

    model, criterion, model_args = load_model(args.model)

    # TODO


    print("size: {}\nloss: {:.4f}\ntop1: {:.2f}%\ntop5: {:.2f}%".format(len(test_data), loss, top1, top5))
