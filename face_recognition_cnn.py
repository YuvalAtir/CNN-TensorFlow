"""
    A deep CNN to classify images to self vs rest.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

from preprocessing import Data


img_row = (64 * 3)
img_col = (64 * 3)

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
    x: an input tensor with the dimensions (N_examples, 6912), where 784 is the
    number of pixels in an image.

    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 2), with values
    equal to the logits of classifying the digit into one of 2 classes .
    keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    global img_row, img_col
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, img_row, img_col, 1])
    # tensorboard visualization
    tf.summary.image('input', x_image, 3)

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    h_pool1 = conv_layer(x_image, 1, 32, name='conv1')

    # Second convolutional layer -- maps 32 feature maps to 64.
    h_pool2 = conv_layer(h_pool1, 32, 64, name='conv2')

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    fc1_input_size = (img_row//4)*(img_col//4) * 64   # // to get result in int
    h_pool2_flat = tf.reshape(h_pool2, [-1, fc1_input_size])
    h_fc1 = fc_layer(h_pool2_flat, fc1_input_size, 1024, name='fc1')  # logits

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    y_conv = fc_layer(h_fc1_drop, 1024, 2, name='fc2')

    return y_conv, keep_prob


def conv_layer(input, channels_in, channels_out, name='conv'):
    with tf.name_scope(name):
        W_conv1 = weight_variable([5, 5, channels_in, channels_out],'W')
        b_conv1 = bias_variable([channels_out], 'B')
        conv = conv2d(input, W_conv1)
        h_conv1 = tf.nn.relu(conv + b_conv1)  # activation
        h_pool1 = max_pool_2x2(h_conv1)
        # tensorboard visualization
        tf.summary.histogram('weights'+name, W_conv1)
        tf.summary.histogram('biases'+name, b_conv1)
        tf.summary.histogram('activation'+name, h_conv1)
        tf.summary.histogram('pooling'+name, h_pool1)
        return h_pool1


def fc_layer(input, channels_in, channels_out, name='fc'):
    with tf.name_scope(name):
        W_fc = weight_variable([channels_in, channels_out], 'W')
        b_fc = bias_variable([channels_out], 'b')
        h_fc1 = tf.nn.relu(tf.matmul(input, W_fc) + b_fc)
        # tensorboard visualization
        tf.summary.histogram('weights+name', W_fc)
        tf.summary.histogram('biases+name', b_fc)
        tf.summary.histogram('activation+name', h_fc1)
        return h_fc1


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape,name):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(shape, name):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def main(_):
    global img_row, img_col
    max_steps = 15
    # intit tensorboard
    logdir_train = 'C:\\Users\\Yuval\\Documents\\tensorboard\\1'+'\\train'
    logdir_test = 'C:\\Users\\Yuval\\Documents\\tensorboard\\1' + '\\test'
    train_writer = tf.summary.FileWriter(logdir=logdir_train)
    test_writer = tf.summary.FileWriter(logdir=logdir_test)
    # TODO: add Embedded Visualizer is a cool 3D visualization of tensorboard data

    # Import data
    mydata = Data()

    # Create the model
    x = tf.placeholder(tf.float32, [None, img_row*img_col], name='x')

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='labels')

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    # merges all summaries to be passed to fileWriter
    merged_summary = tf.summary.merge_all()

    # cost function to minimize
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # use AdamOptimizer instead of Gradient Descent Algo
    with tf.name_scope('accuracy'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Measure prediction accuracy by then frequency of correct classifications
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # take time for performance analysis
    start_time = time.time()

    with tf.Session() as sess:
        # tensorboard add graph
        train_writer.add_graph(sess.graph)
        test_writer.add_graph(sess.graph)

        # initialize CNN weights
        sess.run(tf.global_variables_initializer())

        # batch-stochastic gradient descent
        for i in range(max_steps):

            batch_x, batch_y, dropout = mydata.get_batch(i, train=True)
            # run optimizer to calculate gradients
            summary, _ = sess.run([merged_summary, train_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout})
            train_writer.add_summary(summary, i)
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout})
            tf.summary.scalar('accuracy', train_accuracy)
            if i % 5 == 0:
                batch_x, batch_y, dropout = mydata.get_batch(i, train=False)
                summary, test_accuracy = sess.run([merged_summary, accuracy],
                                                  feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout})
                test_writer.add_summary(summary, i)
                print('step %d, test accuracy %g' % (i, test_accuracy))
                print("--- %s seconds ---" % (time.time() - start_time))

            print('i=',i)
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mydata.test_imgs, y_: mydata.test_labels, keep_prob: 1.0}))


if __name__ == '__main__':
    tf.app.run(main=main)
