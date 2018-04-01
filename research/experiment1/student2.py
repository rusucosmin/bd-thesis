from model import Model

from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf


class Student2(Model):
  def __init__(self, name):
    print("Student2::__init__")
    super().__init__(name)

    # placeholders for input and output variables in the dataset (x = features, y = labels)

    # x = 28 x 28 pixels from images
    # y = one hot vector where 1 denotes the correct label
    self.x = tf.placeholder(tf.float32, shape=[None, 784])
    self.y_ = tf.placeholder(tf.float32, shape = [None, 10])

    # reshape the input to a 4d tensor (-1 since we don't know how many images we have)
    # the second and third dimension is the srze of the image
    # and the last dimension repressents the number of color channels
    self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

    # The first convolutional layer
    self.W_conv1 = Model.weight_variable([5, 5, 1, 3])
    self.b_conv1 = Model.bias_variable([3])

    # convolve the image with a relu actiovation function
    self.h_conv1 = tf.nn.relu(Model.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
    # add max_pooling layer
    self.h_pool1 = Model.max_pool_2x2(self.h_conv1)

    # create the fully connected layer
    self.W_fc1 = Model.weight_variable([14 * 14 * 3, 10])
    self.b_fc1 = Model.bias_variable([10])

    # reshape the pooling layer to be flat
    self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, 14 * 14 * 3])
    self.y_conv = tf.matmul(self.h_pool1_flat, self.W_fc1) + self.b_fc1

    # learning rate
    self.learning_rate = 0.0001

    # train and evaluate
    self.cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = self.y_, logits = self.y_conv))
    self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

  def train(self, mnist):
    print("Student2::train")

    n_epochs = 50
    batch_size = 50
    n_batches = len(mnist.train.images) // batch_size

    losses = []
    accs = []
    test_accs = []

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for epoch in range(n_epochs):
          x_shuffle, y_shuffle \
                  = shuffle(mnist.train.images, mnist.train.labels)
          print("Starting training opoch %d" % epoch)
          for i in range(n_batches):
              start = i * batch_size
              end = start + batch_size
              batch_x, batch_y \
                      = x_shuffle[start:end], y_shuffle[start:end]
              sess.run(self.train_step, feed_dict = {
                  self.x: batch_x,
                  self.y_: batch_y })
          x_shuffle, y_shuffle \
                  = shuffle(mnist.train.images, mnist.train.labels)
          batch_x, batch_y \
                  = x_shuffle[0:250], y_shuffle[0:250]
          train_loss = sess.run(self.cross_entropy, feed_dict = {
              self.x: batch_x,
              self.y_: batch_y })
          train_accuracy = sess.run(self.accuracy, feed_dict = {
              self.x: batch_x,
              self.y_: batch_y })
          test_accuracy = sess.run(self.accuracy, feed_dict = {
              self.x: mnist.test.images,
              self.y_: mnist.test.labels })
          print("Epoch : %i, Loss : %f, Accuracy: %f, Test accuracy: %f" % (
                  epoch + 1, train_loss, train_accuracy, test_accuracy))
          losses.append(train_loss)
          accs.append(train_accuracy)
          test_accs.append(test_accuracy)

    return (losses, accs, test_accs)


  def test(self, x_test, y_test):
    print("Student2::test")

