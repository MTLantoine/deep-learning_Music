import generate_data
import random
import torch
import midi
import argparse
import torch.nn as nn


def generate_rand_full(seq_len, labels):
    pass # TODO

def generate_rand_one(seq_len, labels):
    pass # TODO

def load_model(filename):
    args = torch.load(filename)
    model = args["model"]
    criterion = args["criterion"]
    return model, criterion, args

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate")

    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--frequency", type=int, default=5)
    parser.add_argument("--init", type=str, default="full")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Watning: CUDA not available, using CPU")
        device = torch.device("cpu")

    model, _, model_args = load_model("best.pt")

    labels = generate_data.read_labels("labels")
    class_number = len(labels)

    seq_len = model_args["sequence_length"]

    if args.init == "full":
        sequence = generate_rand_full(seq_len, labels)
    elif args.init == "one":
        sequence = generate_rand_one(seq_len, labels)
    else:
        raise("Unknown init type "+args.init)

    model.eval()
    model.to(device)

    softmax = nn.Softmax(dim=1)

    for i in range(args.length):
        pass # TODO

    notes = generate_data.index_2_notes(sequence, labels)
    piano_roll = midi.notes_2_piano_roll(notes)
    midi_format = midi.piano_roll_2_midi(piano_roll, args.frequency)
    midi_format.write(args.output)