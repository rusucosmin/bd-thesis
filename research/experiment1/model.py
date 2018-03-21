import tensorflow as tf

class Model:
  def __init__(self, name):
    self.name = name

  # will return a new weight variables containing a random, variable in the interval [-0.2, 0.2]
  @staticmethod
  def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev = 0.1)
      return tf.Variable(initial)

  # will return a new bias variable with a value of 0.1
  @staticmethod
  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  # temperature denotes the way logits are transformed in softmax variables
  @staticmethod
  def softmax_with_temperature(logits, temp=1.0, axis=1, name=None):
    logits_with_temp = logits / temp
    _softmax = tf.exp(logits_with_temp) / tf.reduce_sum(tf.exp(logits_with_temp), axis=axis,
        keep_dims=True)
    return _softmax

  # will return a new convolutional layer, with a stride of 1 and zero padding
  @staticmethod
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # will return a new max pooling layer over 2x2 blocks
  @staticmethod
  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides = [1, 2, 2, 1], padding='SAME')

  def __str__(self):
    return self.name
