import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        class_number,
        embeding_size = 100,
        hidden_size = 128,
        num_layers=2,
        dropout=0.2
    ):
        super(LSTMClassifier, self).__init__()

        pass # TODO

    def forward(self, x):
        pass # TODO
        return x

if __name__ == "__main__":
    # Q11
    batch_size = 8
    sequence_length = 10
    class_number = 1000
    model = LSTMClassifier(class_number)

    input = torch.randint(high=class_number, size=(batch_size, sequence_length))
    print("Input tensor size: ", input.size())
    output = model(input)
    print("Output tensor size: ", output.size())
     


