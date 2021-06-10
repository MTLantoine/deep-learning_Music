from torch.utils.data import Dataset
import torch
import os
import random

import midi
import generate_data


class MaestroDataset(Dataset):

    # Q10
    def __init__(self, data_file, sequence_length, stride):
        pass # TODO

    # Q11
    def __len__(self):
        pass # TODO

    # Q12
    def __getitem__(self, idx):
        pass # TODO

if __name__ == "__main__":

    # Q10
    test_data = MaestroDataset("test.npy", 100, 10)

    # Q11
    print("Dataset size: ", len(test_data))

    # Q12
    x, y = test_data[0]
    print("X: ", x.size(), x.type())
    print("y: ", y.size(), y.type())