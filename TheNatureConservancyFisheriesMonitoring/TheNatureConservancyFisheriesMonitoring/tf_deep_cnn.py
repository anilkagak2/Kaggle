from __future__ import print_function

import classifier as NCF
import numpy as np
import tensorflow as tf
import os
import sys

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X_train, X_test, y_train, y_test = NCF.get_features_and_labels(os.path.join(NCF.Data_Dir, 'train'))
print("n_input = {0}".format(X_train[0].size))
print("n_classes = {0}".format(np.unique(y_train).size))
#sys.exit(1)

# Parameters
learning_rate = 0.001
#training_iters = 10000
training_iters = 2000
batch_size = 64
#batch_size = 256
display_step = 100

# Network Parameters
n_input = X_train[0].size #784 # MNIST data input (img shape: 28*28)
n_classes = np.unique(y_train).size #10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

num_train, num_test = len(y_train), len(y_test)
old_y_train, old_y_test = y_train, y_test
y_train = np.zeros((num_train, n_classes))
y_train[np.arange(num_train), old_y_train] = 1.0
y_test = np.zeros((num_test, n_classes))
y_test[np.arange(num_test), old_y_test] = 1.0

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, strides=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 227, 227, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 4)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=3, strides=2)
    conv1 = tf.nn.lrn(conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3, strides=2)
    conv2 = tf.nn.lrn(conv2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=3, strides=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=3, strides=2)

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=3, strides=2)
    conv5 = tf.nn.lrn(conv5)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.tanh(fc1)
    #fc1 = tf.nn.sigmoid(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    #out = tf.nn.softmax(out)
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

model_dir = "C:\\Users\\t-anik\\Desktop\\personal\\MachineLearning\\NeuralNets\\deep_cnn_model\\"
model_file = model_dir+"deep_cnn_model"

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    for step in range(0, training_iters):
    #while step * batch_size < training_iters:
        next_batch = np.random.choice(num_train, size=batch_size, replace=False)
        #batch_x, batch_y = X_train[step*(batch_size-1):step*batch_size], y_train[step*(batch_size-1):step*batch_size] #mnist.train.next_batch(batch_size)
        batch_x, batch_y = X_train[next_batch], y_train[next_batch] #mnist.train.next_batch(batch_size)
        #print(batch_x.size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        #step += 1
    print("Optimization Finished!")

    saver.save(sess, model_file)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_file + ".meta")
    new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Calculate accuracy for 256 mnist test images
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    #                                  y: mnist.test.labels[:256],
    #                                  keep_prob: 1.}))

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X_test,
                                      y: y_test,
                                      keep_prob: 1.}))

    prediction = tf.nn.softmax( pred )
    #prediction = pred 
    print("predictions")
    print(prediction.eval(feed_dict={x: X_test[0].reshape((1,784*3)), keep_prob: 1.}, session=sess))
    print("actual value = {0}".format(y_test[0]))

    print("Prediction..")
    X_test, filenames = NCF.get_feature_test_points(os.path.join(NCF.Data_Dir, 'test_stg1'))
    num_test = X_test.shape[0]
    predictions = np.zeros((num_test, n_classes))
    batch_size = 256
    num_batches = int(num_test / batch_size)
    for i in range(num_batches):
        predictions[i*batch_size:(i+1)*batch_size] = prediction.eval(feed_dict={x: X_test[i*batch_size:(i+1)*batch_size], keep_prob: 1.}, session=sess)

    #from sklearn.preprocessing import normalize
    #predictions = normalize(1.0/( 1+np.exp(-1*predictions)), axis=1, norm='l1')
    print(predictions[0])

    NCF.writePredictionsToCsv(NCF.Data_Dir, predictions, filenames)
