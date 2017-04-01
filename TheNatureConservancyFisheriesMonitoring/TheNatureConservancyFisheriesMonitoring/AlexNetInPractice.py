from __future__ import print_function
import classifier as NCF
import numpy as np
import tensorflow as tf
import os
import sys

X_train, X_test, y_train, y_test = NCF.get_images_labels(os.path.join(NCF.Data_Dir, 'train'))
print("X_train shape")
print(X_train.shape)
print("y_train shape")
print(y_train.shape)

print("X_test shape")
print(X_test.shape)
print("y_test shape")
print(y_test.shape)

learning_rate = 0.001
training_iters = 400
batch_size = 64
display_step = 100

Ht, Wt, num_channels = 64, 64, 3
num_train, num_test = len(y_train), len(y_test)
n_input, n_classes = Ht*Wt*num_channels, len(NCF.classLabels)

old_y_train, old_y_test = y_train, y_test
y_train = np.zeros((num_train, n_classes))
y_train[np.arange(num_train), old_y_train] = 1.0
y_test = np.zeros((num_test, n_classes))
y_test[np.arange(num_test), old_y_test] = 1.0

weights = {
    'wc1' : tf.get_variable("wc1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
    'out' : tf.get_variable("out", shape=[32*32*32, n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, Ht, Wt, num_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

print(x)
conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
print(conv1)
conv1 = tf.nn.bias_add(conv1, biases['bc1'])
print(conv1)
conv1 = tf.nn.relu(conv1)
print(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(conv1)
fc1 = tf.reshape(conv1, [-1, weights['out'].get_shape().as_list()[0]])
print(fc1)
fc1 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
print(fc1)
fc1 = tf.nn.relu(fc1)
print(fc1)
#sys.exit(1)

pred = fc1
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
dropout = 0.75

with tf.Session() as sess:
    sess.run(init)
    for step in range(0, training_iters):
        batch = np.random.choice(num_train, size=batch_size)
        batch_x, batch_y = X_train[batch], y_train[batch]
        batch_x = [a.astype(np.float) for a in batch_x]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter " + str(step) + \
                ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                ", Training Accuracy= " + "{:.5f}".format(acc))

    losses, accuracies = [], []
    s = 0
    test_batch_size = 64
    while (s+1)*test_batch_size < num_test:
        start, end = s*test_batch_size, (s+1)*test_batch_size
        batch_x, batch_y = X_test[start:end], y_test[start:end]
        batch_x = [a.astype(np.float) for a in batch_x]
        loss, acc = sess.run([cost, accuracy], \
            feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        print(loss)
        losses.append(loss)
        accuracies.append(acc)
        s += 1

    loss, acc = np.mean(np.array(losses)), np.mean(np.array(accuracies))
    print("Test Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))