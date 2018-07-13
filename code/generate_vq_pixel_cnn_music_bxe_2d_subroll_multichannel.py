import argparse
import numpy as np
from lib.datasets import quantized_imlike_to_image_array
from lib.datasets import save_image_array
from lib.datasets import notes_to_midi
from lib.datasets import midi_to_notes
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from tfbldr.datasets import quantized_to_pretty_midi
import os
from data_utils import dump_subroll_samples

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('pixelcnn_model', nargs=1, default=None)
parser.add_argument('vqvae_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--temp', dest='temp', type=float, default=1.)
parser.add_argument('--num', dest='num', type=int, default=16)
parser.add_argument('--chords', dest='chords', type=str, default=None,
                    help="Example, `--chords=I,I6,IV,V7,I`")
args = parser.parse_args()
vqvae_model_path = args.vqvae_model[0]
pixelcnn_model_path = args.pixelcnn_model[0]

num_to_generate = 160
num_each = args.num
random_state = np.random.RandomState(args.seed)

d = np.load("vq_vae_encoded_music_2d_subroll_multichannel.npz")

offset_to_pitch = {int(k): int(v) for k, v in d["offset_to_pitch_kv"]}
label_to_chord_function = {int(k): v for k, v in d["label_to_chord_function_kv"]}
chord_function_to_label = {v: k for k, v in label_to_chord_function.items()}

labels = d["labels"]
train_labels = labels[:-num_to_generate]
valid_labels = labels[-num_to_generate:]
sample_labels = valid_labels

if args.chords is not None:
    chord_labels = []
    chord_seq = args.chords.split(",")
    prev = None
    for n, chs in enumerate(chord_seq):
        if chs not in chord_function_to_label:
            print("Possible chords {}".format(sorted(chord_function_to_label.keys())))
            raise ValueError("Unable to find chord {} in chord set".format(chs))
        cur = chord_function_to_label[chs]
        if prev == None:
            prev = cur
        if n == (len(chord_seq) - 1):
            nxt = cur
        else:
            nxt = chord_function_to_label[chord_seq[n + 1]]
        chord_labels.append((prev, cur, nxt))
    chord_labels = chord_labels * 5
    num_to_generate = len(chord_labels)
    num_each = len(chord_seq)
    sample_labels = np.array(chord_labels)


def sample_gumbel(logits, temperature=args.temp):
    noise = random_state.uniform(1E-5, 1. - 1E-5, np.shape(logits))
    return np.argmax((logits - logits.max() - 1) / float(temperature) - np.log(-np.log(noise)), axis=-1)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

with tf.Session(config=config) as sess1:
    saver = tf.train.import_meta_graph(pixelcnn_model_path + '.meta')
    saver.restore(sess1, pixelcnn_model_path)
    fields = ['images',
              'labels',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    y = sample_labels[:num_to_generate]

    pix_z = np.zeros((num_to_generate, 13, 4))
    for i in range(pix_z.shape[1]):
        for j in range(pix_z.shape[2]):
            print("Sampling v completion pixel {}, {}".format(i, j))
            feed = {vs.images: pix_z[..., None],
                    vs.labels: y}
            outs = [vs.x_tilde]
            r = sess1.run(outs, feed_dict=feed)
            x_rec = sample_gumbel(r[-1])

            for k in range(pix_z.shape[0]):
                pix_z[k, i, j] = float(x_rec[k, i, j])


sess1.close()
tf.reset_default_graph()

with tf.Session(config=config) as sess2:
    saver = tf.train.import_meta_graph(vqvae_model_path + '.meta')
    saver.restore(sess2, vqvae_model_path)
    """
    # test by faking like we sampled these from pixelcnn
    d = np.load("vq_vae_encoded_mnist.npz")
    valid_z_i = d["valid_z_i"]
    """
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    z_i = pix_z[:num_to_generate]
    fake_image_data = np.zeros((num_to_generate, 52, 16, 4))
    feed = {vs.images: fake_image_data,
            vs.z_i_x: z_i,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[-1]

# binarize the predictions
x_rec[x_rec > 0.5] = 1.
x_rec[x_rec <= 0.5] = 0.

dump_subroll_samples(x_rec, sample_labels, num_each, args.seed, args.temp, args.chords, offset_to_pitch, label_to_chord_function, tag="base")
from IPython import embed; embed(); raise ValueError()
