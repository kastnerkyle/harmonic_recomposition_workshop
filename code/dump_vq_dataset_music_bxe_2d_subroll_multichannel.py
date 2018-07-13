import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_mnist
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from tfbldr.datasets import fetch_josquin
from tfbldr.datasets import quantized_imlike_to_image_array
from data_utils import music_pitch_and_chord_to_imagelike_and_label


parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--model', dest='model_path', type=str, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
if args.model_path == None:
    if args.direct_model == None:
        raise ValueError("Must pass first positional argument as model, or --model argument, e.g. summary/experiment-0/models/model-7")
    else:
        model_path = args.direct_model[0]
else:
    model_path = args.model_path

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

josquin = fetch_josquin()
images, labels, lookups = music_pitch_and_chord_to_imagelike_and_label(josquin)

image_data = images

bs = 50

image_data = image_data[:len(image_data) - len(image_data) % bs]
labels = labels[:len(image_data)]

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    z = []
    for i in range(len(image_data) // bs):
        print("Minibatch {}".format(i))
        x = image_data[i * bs:(i + 1) * bs]
        feed = {vs.images: x,
                vs.bn_flag: 1.}

        outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
        r = sess.run(outs, feed_dict=feed)
        x_rec = r[-1]
        z_i = r[-2]
        z += [zz[:, :, None] for zz in z_i]
    z = np.array(z)
    zcn = labels[:len(z)]

label_to_chord_function_kv = [(k, v) for k, v in lookups["labels_to_chord_functions"].items()]
offset_to_pitch_kv = [(k, v) for k, v in lookups["offset_to_pitch"].items()]

np.savez("vq_vae_encoded_music_2d_subroll_multichannel.npz",
         z=z,
         labels=zcn,
         label_to_chord_function_kv=label_to_chord_function_kv,
         offset_to_pitch_kv=offset_to_pitch_kv,
         )
print("dumped to vq_vae_encoded_music_2d_subroll_multichannel.npz")
from IPython import embed; embed(); raise ValueError()
