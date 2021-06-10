import os
import argparse
import pretty_midi
import numpy as np


def midi_2_piano_roll(path, frequency):
    midi = pretty_midi.PrettyMIDI(path)
    instrument = midi.instruments[0]
    return instrument.get_piano_roll(fs=frequency)


def piano_roll_2_midi(data, frequency):
    pm = pretty_midi.PrettyMIDI()
    notes, frames = data.shape
    instrument = pretty_midi.Instrument(program=0)
    piano_roll = np.pad(data, [(0, 0), (1, 1)], 'constant')
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / frequency
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    for note in pm.instruments[0].notes:
        note.velocity = 100
    return pm


def notes_2_label(notes):
    new_notes = [note for note in notes]
    new_notes[:] = (value for value in notes if value > 0)
    new_notes.sort()
    res = ""
    for i, label in enumerate(new_notes):
        res += str(label)
        if i < len(new_notes) - 1:
            res += ","
    return res


def label_2_notes(label):
    return [int(note) for note in label.split(',')] if label else []


def piano_roll_2_notes(piano_roll):
    res = []
    for p in piano_roll:
        tmp = notes_2_label(p)
        if tmp:
            res.append(tmp)
    return res


def notes_2_piano_roll(notes):
    notes_length = len(notes)
    maxi = 0
    for note in notes:
        l_2_n = label_2_notes(note)
        for n in l_2_n:
            if n > maxi:
                maxi = n
    res = np.zeros((notes_length, maxi))
    for i, note in enumerate(notes):
        l_2_n = label_2_notes(note)
        for n in l_2_n:
            res[i][n - 1] = n
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="midi.py")

    parser.add_argument("dataset_path")
    args = parser.parse_args()

    frequency = 5

    # Load midi
    data = midi_2_piano_roll(os.path.join(
        args.dataset_path,
        "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
    ), frequency)

    print("Piano roll shape:", data.shape)

    # Save midi
    generate_to_midi = piano_roll_2_midi(data, frequency)
    generate_to_midi.write("output.midi")

    # Q1
    print("Label for []: ", '"{}"'.format(notes_2_label([])))
    print("Label for [28]: ", '"{}"'.format(notes_2_label([28])))
    print("Label for [42, 23, 45]: ", '"{}"'.format(notes_2_label([42, 23, 45])))
    print("Label for [42.0 23.0 45.0]: ", '"{}"'.format(notes_2_label(np.array([42, 23, 45]))))


    # Q2
    print('Notes for "": ', label_2_notes(""))
    print('Notes for "28": ', label_2_notes("28"))
    print('Notes for "23,42,45": ', label_2_notes("23,42,45"))

    # Q3
    sequence = piano_roll_2_notes(data)
    print("10 first notes: ", sequence[0:10])

    # Q4
    piano_roll = notes_2_piano_roll(["", "42", "42,45"])
    print("Piano roll shape: ", piano_roll.shape)
    print(piano_roll[42:46])
