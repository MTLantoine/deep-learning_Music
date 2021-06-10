import argparse
import midi
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def load_data(annotations, limit):
    data = []
    for index, row in annotations.iterrows():
        piano_roll = midi.midi_2_piano_roll(
            os.path.join(args.dataset_path, row["midi_filename"]),
            args.frequency
        )
        notes = midi.piano_roll_2_notes(piano_roll)
        data.append({
            "notes": notes,
            "split": row["split"]
        })
        if index%100 == 0:
            print("Load data {}/{}".format(index, annotations.shape[0]))

        if limit != None and index > limit:
            break
            
    return data

def read_labels(file):
    with open(file, "r") as f:
        return f.read().split("\n")

def write_labels(file, labels):
    with open(file, "w") as f:
        f.write("\n".join(labels))

def make_label_2_index_dict(labels):
    return { l: i for i, l in enumerate(labels)}

def make_index_2_label_dict(labels):
    return { i: l for i, l in enumerate(labels)}

def transform_data(data, labels_2_index):
    outputs = []
    for entry in data:
        outputs.append([note_2_index(note, labels_2_index) for note in entry["notes"]])
    return outputs

def read_data(file):
    return np.load(file, allow_pickle=True).tolist()

def write_data(file, data):
    np.save(file, np.array(data, dtype=object), allow_pickle=True)

def get_all_labels(data):
    pass # TODO

def make_label_list(labels, threshold):
    pass # TODO

def note_2_index(note, labels_2_index):
    pass # TODO


def notes_2_index(sequence, labels_2_index):
    pass # TODO

def index_2_notes(sequence, index_2_labels):
    pass # TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data")

    parser.add_argument("dataset_path")
    parser.add_argument("--csv_name", default="maestro-v3.0.0.csv")
    parser.add_argument("--frequency", type=int, default=5)
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()
    
    annotations = pd.read_csv(os.path.join(args.dataset_path, args.csv_name))

    data = load_data(annotations, args.limit)

    # Q5
    labels = get_all_labels(data)
    print("Total sample number: ", len(labels))

    # Q6
    labels = make_label_list(labels, 10)
    print("Filtered label number: ", len(labels))

    write_labels("labels", labels)

    labels_2_index = make_label_2_index_dict(labels)

    for split in ['train', 'validation', 'test']:
        curr = [el for el in data if el['split'] == split]
        curr_data = transform_data(curr, labels_2_index)
        write_data(split+'.npy', curr_data)