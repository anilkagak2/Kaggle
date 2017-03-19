import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


learning_rate = 0.001
training_iters = 20000
#training_iters = 2000
batch_size = 64
display_step = 10

num_train = len(mnist.train.labels)
Ht, Wt, num_channels = 28, 28, 1
n_input, n_classes = Ht*Wt, 10

# Define variables 
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
dropout = 0.75

# This defines the input for the layer
input = tf.reshape(x, shape=[-1, Ht, Wt, num_channels])

conv1 = tf.nn.conv2d(input, tf.Variable(tf.random_normal([5, 5, num_channels, 32])), strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, tf.Variable(tf.random_normal([32])))
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = tf.nn.conv2d(conv1, tf.Variable(tf.random_normal([5, 5, 32, 64])), strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv2, tf.Variable(tf.random_normal([64])))
conv1 = tf.nn.relu(conv2)
conv1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

fc1 = tf.reshape(conv1, [-1, 7*7*64])
fc1 = tf.matmul(fc1, tf.Variable(tf.random_normal([7*7*64, 1024])))
fc1 = tf.add(fc1, tf.Variable(tf.random_normal([1024])))
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, dropout)

fc2 = tf.matmul(fc1, tf.Variable(tf.random_normal([1024, n_classes])))
fc2 = tf.add(fc2, tf.Variable(tf.random_normal([n_classes])))
#fc2 = tf.nn.relu(fc2)

pred = tf.nn.softmax(fc2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + \
                ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                ", Training Accuracy= " + "{:.5f}".format(acc))

            losses, accuracies = [], []
            s = 0
            test_batch_size = 1024
            while (s+1)*batch_size < num_train:
                start, end = s*batch_size, (s+1)*batch_size
                loss, acc = sess.run([cost, accuracy], \
                    feed_dict={x: mnist.train.images[start:end], y: mnist.train.labels[start:end]})
                losses.append(loss)
                accuracies.append(acc)
                s += 1

            loss, acc = np.mean(np.array(losses)), np.mean(np.array(accuracies))
            print("Train Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")

    loss, acc = sess.run([cost, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("TestSet Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))