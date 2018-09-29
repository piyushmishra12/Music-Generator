import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import msgpack

import midi_manipulation

def retrieve_song(folder):
    files = glob.glob('{}/*.mid*'.format(folder))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

songs = retrieve_song('Training_set')

lowest = midi_manipulation.lowerBound
highest = midi_manipulation.upperBound
noteRange = highest - lowest

timesteps = 15
nv = 2 * noteRange * timesteps # number of visible nodes
nh = 2340 # number of hidden nodes
epochs = 200
batch_size = 100
lr = tf.constant(0.005, tf.float32)

# Initialising the placeholder, weight and biases for the RBM
x = tf.placeholder(tf.float32, [None, nv], name = 'x') # stores the data in the visible layer
w = tf.Variable(tf.random_normal([nv, nh], mean = 0.0), name = 'w') # initialise a random matrix with
# random samples from a normal distribution with mean 0.0
a = tf.Variable(tf.zeros([1, nh], tf.float32, name = 'a')) # bias for hidden layer
b = tf.Variable(tf.zeros([1, nv], tf.float32, name = 'b')) # bias for visible layer

# Constructing the Restricted Boltzmann Machine
# Creating a Gibb's sampling function
def sample(prob):
    return tf.floor(prob + tf.random_uniform(tf.shape(prob), 0, 1))
# returns a random matrix of 0s and 1s sampled from the input matrix of probabilities

def GibbsSampling(k):
    def SingleGibbsStep(count, k, xk):
        # visible values initialised to xk and hidden values are sampled using that value
        hk = sample(tf.sigmoid(tf.matmul(xk, w) + a))
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(w)) + b))
        return count + 1, k, xk
    # Run a k-step Gibb's chain to sample from probaility distribution of the RBM
    # defined by the weights and the biases
    c = tf.constant(0) # count
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, iterations,
                                                    *args: count < iterations,
                                                    SingleGibbsStep, [c, tf.constant(k), x])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

# Contrastive Divergence
# First the Gibb's sampling
x_sample = GibbsSampling(1) # setting x1
h = sample(tf.sigmoid(tf.matmul(x, w) + a)) # sampling hidden nodes
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, w) + a))

# Updating weights and biases with contrastive divergence
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
delta_w = tf.multiply(lr / size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
# using delta_w = x'*h - xtilde'*htilde
delta_a = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
delta_b = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

update = [w.assign_add(delta_w), b.assign_add(delta_b), a.assign_add(delta_a)]


# Training the model
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in tqdm(range(epochs)):
        for song in songs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] // timesteps) * timesteps)]
            song = np.reshape(song, [song.shape[0] // timesteps, song.shape[1] * timesteps])
            # Training 'batch_size' songs at a time
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})