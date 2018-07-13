from tfbldr.datasets import fetch_mnist
from tfbldr.nodes import GatedMaskedConv2d
from tfbldr.nodes import Conv2d
from tfbldr.nodes import Embedding
from tfbldr.nodes import BatchNorm2d
from tfbldr.nodes import ReLU
from tfbldr.nodes import Softmax
from tfbldr.nodes import CategoricalCrossEntropyLinearIndexCost
from tfbldr.datasets import list_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
import tensorflow as tf
import numpy as np
from collections import namedtuple
import copy

d = np.load("vq_vae_encoded_music_2d_subroll_multichannel.npz")
image_data = copy.deepcopy(d["z"])
train_image_data = image_data[:-500]
valid_image_data = image_data[-500:]

labels = d["labels"]
train_labels = labels[:-500]
valid_labels = labels[-500:]

n_labels = len(set(list(np.unique(train_labels)) + list(np.unique(valid_labels))))
print("n_labels {}".format(n_labels))

train_condition = 0. * train_image_data
valid_condition = 0. * valid_image_data
train_condition[1:] = train_image_data[:-1]
valid_condition[1:] = valid_image_data[:-1]

train_itr_random_state = np.random.RandomState(1122)
val_itr_random_state = np.random.RandomState(1)
train_itr = list_iterator([train_image_data, train_labels, train_condition], 50, random_state=train_itr_random_state)
valid_itr = list_iterator([valid_image_data, valid_labels, valid_condition], 50, random_state=val_itr_random_state)

random_state = np.random.RandomState(1999)

# only 12x8 in the first place...
kernel_size0 = (7, 7)
kernel_size1 = (3, 3)
n_channels = 64
n_layers = 15

def create_pixel_cnn(inp, lbl, cond):
    e_inp, emb = Embedding(inp, 256, n_channels, random_state=random_state, name="inp_emb")
    c_inp, emb = Embedding(cond, 256, n_channels, random_state=random_state, name="cond_emb")
    l1_v, l1_h = GatedMaskedConv2d([e_inp], [n_channels], [e_inp], [n_channels],
                                   n_channels,
                                   residual=False,
                                   conditioning_class_input=lbl,
                                   conditioning_num_classes=n_labels,
                                   conditioning_spatial_map=c_inp,
                                   kernel_size=kernel_size0, name="pcnn0",
                                   mask_type="img_A",
                                   random_state=random_state)
    o_v = l1_v
    o_h = l1_h
    for i in range(n_layers - 1):
        t_v, t_h = GatedMaskedConv2d([o_v], [n_channels], [o_h], [n_channels],
                                     n_channels,
                                     conditioning_class_input=lbl,
                                     conditioning_num_classes=n_labels,
                                     conditioning_spatial_map=c_inp,
                                     kernel_size=kernel_size1, name="pcnn{}".format(i + 1),
                                     mask_type="img_B",
                                     random_state=random_state)
        o_v = t_v
        o_h = t_h

    cleanup = Conv2d([o_h], [n_channels], n_channels, kernel_size=(1, 1),
                     name="conv_c",
                     random_state=random_state)
    r_p = ReLU(cleanup)
    out = Conv2d([r_p], [n_channels], 256, kernel_size=(1, 1),
                 name="conv_o",
                 random_state=random_state)
    #s_out = Softmax(out)
    return out#s_out


def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, shape=[None, 13, 4, 1])
        conds = tf.placeholder(tf.float32, shape=[None, 13, 4, 1])
        labels = tf.placeholder(tf.float32, shape=[None, 3])
        x_tilde = create_pixel_cnn(images, labels, conds)
        loss = tf.reduce_mean(CategoricalCrossEntropyLinearIndexCost(x_tilde, images))
        #loss = tf.reduce_mean(BernoulliCrossEntropyCost(x_tilde, images))
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_tilde, labels=images))
        #loss = tf.reduce_mean((x_tilde - images) ** 2)
        params = get_params_dict()
        grads = tf.gradients(loss, params.values())

        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(grads) == len(params)
        j = [(g, p) for g, p in zip(grads, params.values())]
        train_step = optimizer.apply_gradients(j)

    things_names = ["images",
                    "labels",
                    "conds",
                    "x_tilde",
                    "loss",
                    "train_step"]
    things_tf = [eval(name) for name in things_names]
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

def loop(sess, itr, extras, stateful_args):
    x, y, c = itr.next_batch()
    if extras["train"]:
        feed = {vs.images: x,
                vs.labels: y,
                vs.conds: c}
        outs = [vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        step = r[1]
    else:
        feed = {vs.images: x,
                vs.labels: y,
                vs.conds: c}
        outs = [vs.loss]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
    return l, None, stateful_args

with tf.Session(graph=g) as sess:
    run_loop(sess,
             loop, train_itr,
             loop, valid_itr,
             n_steps=100 * 1000,
             n_train_steps_per=5000,
             n_valid_steps_per=1000)

print("training done")
from IPython import embed; embed(); raise ValueError()
