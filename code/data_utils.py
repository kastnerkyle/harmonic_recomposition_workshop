from tfbldr.datasets import quantized_imlike_to_image_array
from tfbldr.datasets import save_image_array
from tfbldr.datasets import quantized_to_pretty_midi
from tfbldr.datasets import midi_to_notes
import numpy as np
from collections import Counter
import os
import copy

def dump_subroll_samples(x_rec, sample_labels, num_each, seed, temp, chords, offset_to_pitch_map, label_to_chord_function_map, tag=""):
    for offset in np.arange(0, len(x_rec), num_each):
        x_rec_i = x_rec[offset:offset + num_each]
        x_ts = quantized_imlike_to_image_array(x_rec_i, 0.25)

        if not os.path.exists("samples"):
            os.mkdir("samples")

        if chords is None:
            save_image_array(x_ts, "samples/subroll_{}_multichannel_pixel_cnn_gen_{:04d}_seed_{:04d}_temp_{}.png".format(tag, offset, seed, temp), resize_multiplier=(4, 1), gamma_multiplier=7, flat_wide=True)
        else:
            save_image_array(x_ts, "samples/subroll_{}_multichannel_pixel_cnn_chords_{}_gen_{:04d}_seed_{:04d}_temp_{}.png".format(tag, chords, offset, seed, temp), resize_multiplier=(4, 1), gamma_multiplier=7, flat_wide=True)

        satb_midi = [[], [], [], []]
        satb_notes = [[], [], [], []]
        for n in range(len(x_rec_i)):
            measure_len = x_rec_i[n].shape[1]
            # 46 x 16 2 measures in
            events = {}
            for v in range(x_rec_i.shape[-1]):
                all_up = zip(*np.where(x_rec_i[n][..., v]))
                time_ordered = [au for i in range(measure_len) for au in all_up if au[1] == i]
                for to in time_ordered:
                    if to[1] not in events:
                        # fill with rests
                        events[to[1]] = [0, 0, 0, 0]
                    events[to[1]][v] = to[0]

            satb =[[], [], [], []]
            for v in range(x_rec_i.shape[-1]):
                for ts in range(measure_len):
                    if ts in events and events[ts][v] in offset_to_pitch_map:
                        satb[v].append(offset_to_pitch_map[events[ts][v]])
                    else:
                        # edge case if ALL voices rest or get an out of range note
                        satb[v].append(0)
            # was ordered btas
            satb = satb[::-1]
            for i in range(len(satb)):
                satb_midi[i].extend(satb[i])
                satb_notes[i].extend(midi_to_notes([satb[i]])[0])

        if chords is None:
            name_tag="subroll_{}_multichannel_sample_{:04d}_seed_{:04d}_temp_{}".format(tag, offset, seed, temp) + "_{}.mid"
        else:
            name_tag="subroll_{}_multichannel_chords_{}_sample_{:04d}_seed_{:04d}_temp_{}".format(tag, chords, offset, seed, temp) + "_{}.mid"

        these_sample_labels = sample_labels[offset:offset + num_each]
        these_labelnames = [tuple([label_to_chord_function_map[sl] for sl in sli]) for sli in these_sample_labels]

        if chords is None:
            np.savez("samples/{}_sample_{:04d}_seed_{:04d}.npz".format(tag, offset, seed), pr=x_rec_i, midi=satb_midi, notes=satb_notes, labelnames=these_labelnames)
        else:
            np.savez("samples/{}_chords_{}_sample_{:04d}_seed_{:04d}.npz".format(tag, chords, offset, seed), pr=x_rec_i, midi=satb_midi, notes=satb_notes, labelnames=these_labelnames)
        # http://www.janvanbiezen.nl/18century.html
        quantized_to_pretty_midi([satb_midi],
                                 0.25,
                                 save_dir="samples",
                                 name_tag=name_tag,
                                 default_quarter_length=60,
                                 voice_params="woodwinds")
                                 #voice_params="harpsichord")
                                 #voice_params="legend")
        print("saved sample {}".format(offset))


def music_pitch_and_chord_to_imagelike_and_label(music_dict, divisible_by=4, augment=False):
    all_quantized_16th_pitches = music_dict["list_of_data_quantized_16th_pitches_no_hold"]
    all_quantized_16th_chord_functions = music_dict["list_of_data_quantized_16th_chord_functions"]
    if augment:
        raise ValueError("augmentation not yet debugged")
        aug = []
        aug_cf = []
        for n in range(len(all_quantized_16th_pitches)):
            for o in np.arange(-6, 6):
                qp = copy.deepcopy(all_quantized_16th_pitches[n])
                # unchanged?
                qcf = copy.deepcopy(all_quantized_16th_chord_functions[n])
                qp[qp > 0] = qp[qp > 0] + o
                aug.append(qp)

        all_quantized_16th_pitches = aug
        all_quantized_16th_chord_functions = aug_cf

    chord_functions_to_labels = {}
    labels_to_chord_functions = {}
    label_count = 0

    pitch_chunks = []
    labels = []
    for n in range(len(all_quantized_16th_pitches)):
        qp = all_quantized_16th_pitches[n]
        qcf = all_quantized_16th_chord_functions[n]
        if len(qp) != len(qcf):
            qp = qp[:len(qcf)]
        assert len(qp) == len(qcf)
        total_len = len(qp)
        # 1 bars at a time, stepping by 1 bars at a time
        step = 16
        size = 16
        pos = np.arange(0, total_len, step)
        pos = pos[(pos + size) <  total_len]
        assert len(pos) > 1
        for p in pos:
            start = p
            stop = p + size
            qpi = qp[start:stop]
            qcfi = qcf[start:stop]
            c = Counter(qcfi)
            cur = c.most_common(1)[0][0]

            if start < size:
                prv = cur
            else:
                prv = qcf[start - size:start]
                c = Counter(prv)
                prv = c.most_common(1)[0][0]

            if stop >= pos[-2]:
                nxt = cur
            else:
                nxt = qcf[stop:stop + size]
                c = Counter(nxt)
                nxt = c.most_common(1)[0][0]

            for k in [prv, cur, nxt]:
                if k not in chord_functions_to_labels:
                    chord_functions_to_labels[k] = label_count
                    labels_to_chord_functions[label_count] = k
                    label_count += 1
            pl = chord_functions_to_labels[prv]
            cl = chord_functions_to_labels[cur]
            nl = chord_functions_to_labels[nxt]
            labels.append((pl, cl, nl))
            pitch_chunks.append(qpi)
    pitch_set = np.unique(np.concatenate([np.unique(pc) for pc in pitch_chunks]))

    pitch_to_offset_lookup = {v: k for k, v in enumerate(np.sort(pitch_set))}
    offset_to_pitch_lookup = {v: k for k, v in pitch_to_offset_lookup.items()}

    best_dividible = int((len(pitch_to_offset_lookup) // int(divisible_by) + 1) * int(divisible_by))
    oh_lu = np.eye(best_dividible)
    imagelikes = []
    for n in range(len(pitch_chunks)):
        pc = pitch_chunks[n]
        new_imlike = np.zeros((best_dividible, pc.shape[0], pc.shape[1]))
        for v in range(pc.shape[1]):
            vpc = pc[:, v]
            lu = np.array([oh_lu[pitch_to_offset_lookup[pi]] for pi in vpc]).T
            new_imlike[:, :, v] = lu
        imagelikes.append(new_imlike)
    lookups = {}
    lookups["pitch_to_offset"] = pitch_to_offset_lookup
    lookups["offset_to_pitch"] = offset_to_pitch_lookup
    lookups["chord_functions_to_labels"] = chord_functions_to_labels
    lookups["labels_to_chord_functions"] = labels_to_chord_functions
    return np.array(imagelikes), np.array(labels), lookups
