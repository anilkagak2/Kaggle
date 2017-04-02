from __future__ import print_function
import classifier as NCF
import numpy as np
import tensorflow as tf
import os
import sys

X_train, X_test, X_valid, y_train, y_test, y_valid = NCF.get_images_labels(os.path.join(NCF.Data_Dir, 'train'))
mean_intensity = np.mean(X_train)
X_train = X_train - mean_intensity
X_test = X_test - mean_intensity
X_valid = X_valid - mean_intensity
print("X_train shape"); print(X_train.shape); print("y_train shape"); print(y_train.shape)
print("X_test shape"); print(X_test.shape); print("y_test shape"); print(y_test.shape)
print("X_valid shape"); print(X_valid.shape); print("y_valid shape"); print(y_valid.shape)

learning_rate = 0.0001
training_iters = 4000
batch_size = 64
display_step = 100

Ht, Wt, num_channels = 128, 128, 3
num_train, num_test, num_valid = len(y_train), len(y_test), len(y_valid)
n_input, n_classes = Ht*Wt*num_channels, len(NCF.classLabels)

#print(y_train[:10])
#print(y_test[:10])
#print(y_valid[:10])

old_y_train, old_y_test, old_y_valid = y_train, y_test, y_valid
y_train = np.zeros((num_train, n_classes))
y_train[np.arange(num_train), old_y_train] = 1.0
y_test = np.zeros((num_test, n_classes))
y_test[np.arange(num_test), old_y_test] = 1.0
y_valid = np.zeros((num_valid, n_classes))
y_valid[np.arange(num_valid), old_y_valid] = 1.0

print("X_train shape"); print(X_train.shape); print("y_train shape"); print(y_train.shape)
print("X_test shape"); print(X_test.shape); print("y_test shape"); print(y_test.shape)
print("X_valid shape"); print(X_valid.shape); print("y_valid shape"); print(y_valid.shape)

'''
weights = {
    'wc1' : tf.get_variable("wc1", shape=[3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer()),
    'out' : tf.get_variable("out", shape=[112*112*64, n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
'''
weights = {
    'wc1' : tf.get_variable("wc1", shape=[3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
    'wc2' : tf.get_variable("wc2", shape=[64, 64, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
    'out' : tf.get_variable("out", shape=[32*32*64, n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, Ht, Wt, num_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 64 conv nets then pool
conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, biases['bc1'])
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 128 conv nets then pool
conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, biases['bc2'])
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

fc1 = tf.reshape(conv2, [-1, weights['out'].get_shape().as_list()[0]])

out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
out = tf.nn.relu(out)
#sys.exit(1)

pred = out
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
dropout = 0.75

with tf.Session() as sess:
    sess.run(init)
    X_valid = [a.astype(np.float) for a in X_valid]
    for step in range(0, training_iters):
        batch = np.random.choice(num_train, size=batch_size)
        batch_x, batch_y = X_train[batch], y_train[batch]
        batch_x = [a.astype(np.float) for a in batch_x]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            valid_loss, valid_acc = 0.0, 0.0
            
            losses, accuracies = [], []
            s = 0
            test_batch_size = 64
            while (s+1)*test_batch_size < num_valid:
                start, end = s*test_batch_size, (s+1)*test_batch_size
                batch_x, batch_y = X_valid[start:end], y_valid[start:end]
                batch_x = [a.astype(np.float) for a in batch_x]
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                losses.append(loss)
                accuracies.append(acc)
                s += 1

            valid_loss, valid_acc = np.mean(np.array(losses)), np.mean(np.array(accuracies))
            #valid_loss, valid_acc = sess.run([cost, accuracy], feed_dict={x: X_valid, y: y_valid, keep_prob: 1.})
            
            print("Iter " + str(step) + \
                ", Train Minibatch Loss= " + "{:.6f}".format(loss) + \
                ", Train Accuracy= " + "{:.5f}".format(acc) + \
                ", Validation Minibatch Loss= " + "{:.6f}".format(valid_loss) + \
                ", Validation Accuracy= " + "{:.5f}".format(valid_acc))

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

    prediction = tf.argmax(pred, 1)
    print("predictions")
    print(prediction.eval(feed_dict={x: X_test[:1], keep_prob: 1.}, session=sess))
    print("actual value = {0}".format(y_test[0]))

    print("Prediction..")
    X_test, filenames = NCF.get_feature_test_points_preprocessed(os.path.join(NCF.Data_Dir, 'test_stg1'))
    X_test = X_test - mean_intensity
    num_test = X_test.shape[0]
    predictions = np.zeros((num_test, 1)) #np.zeros((num_test, n_classes))
    batch_size = 256
    num_batches = int(num_test / batch_size)
    for i in range(num_batches):
        results = prediction.eval(feed_dict={x: X_test[i*batch_size:(i+1)*batch_size], keep_prob: 1.}, session=sess)
        predictions[i*batch_size:(i+1)*batch_size] = results.reshape((results.shape[0], 1))

    predictions = predictions.astype(np.int)
    old_predictions = predictions
    print("Predictions just a single dimension")
    print(predictions[:10,:])
    predictions = np.zeros((num_test, n_classes))
    for i in range(num_test):
        predictions[i, old_predictions[i]] = 1.0
    #predictions[np.arange(num_test), old_predictions] = 1.0
    #from sklearn.preprocessing import normalize
    #predictions = normalize(1.0/( 1+np.exp(-1*predictions)), axis=1, norm='l1')
    print("Predictions one hot encoding")
    print(predictions[:10,:])

    NCF.writePredictionsToCsv(NCF.Data_Dir, predictions, filenames)