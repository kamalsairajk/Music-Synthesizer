
import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import midi_manipulation
import os
from midi2audio import FluidSynth

sess=tf.Session()    
saver = tf.train.import_meta_graph('model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
def sample(probs):

    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
def gibbs_sample(k):
    def gibbs_step(count, k, xk):
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
        return count + 1, k, xk

    ct = tf.constant(0)  # counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
W = graph.get_tensor_by_name("W:0")
bh = graph.get_tensor_by_name("bh:0")
bv = graph.get_tensor_by_name("bv:0")

 
n_visible = 2340
num_timesteps = 15
n_hidden = 50 
note_range=78
sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
for i in range(sample.shape[0]):
    if not any(sample[i, :]):
        continue
    S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
    midi_manipulation.noteStateMatrixToMidi(S, "out/generated_chord_{}".format(i))
print("Sucessful")
