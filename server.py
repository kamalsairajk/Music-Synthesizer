import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def gibbs_sample(k):
    # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))  # Propagate the visible values to sample the hidden values
        xk = sample(
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))  # Propagate the hidden values to sample the visible values
        return count + 1, k, xk

    # Run gibbs steps for k iterations
    ct = tf.constant(0)  # counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), xk])
    # This is not strictly necessary in this implementation,
    # but if you want to adapt this code to use one of TensorFlow's
    # optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
# Load the model

@app.route('/api',methods=['POST','GET'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        # Here we reshape the vector to be time x notes, and then save the vector as a midi file
        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "out/generated_chord_{}".format(i))
if __name__ == '__main__':
    app.run(port=5000, debug=True)
