import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

train_data_file_name = 'train/train.tfrecords'
filename_queue = tf.train.string_input_producer([train_data_file_name], num_epochs=None)
reader = tf.TFRecordReader()
_, serialized_data = reader.read(filename_queue)

data = tf.parse_single_example(serialized_data,
                               features={
                                   'label': tf.FixedLenFeature([36], tf.float32),
                                   'img': tf.FixedLenFeature([784], tf.float32),
                               })

label_batch, img_batch = tf.train.shuffle_batch([data['label'], data['img']],
                                                batch_size=20, capacity=2000, min_after_dequeue=100)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(img_batch.dtype, shape=img_batch.shape, name='img')
y_ = tf.placeholder(label_batch.dtype, shape=label_batch.shape, name='label')
# First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Readout Layer
W_fc2 = weight_variable([1024, 36])
b_fc2 = bias_variable([36])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# session = tf.InteractiveSession()

# W = tf.Variable(tf.zeros([784, 36]))
# b = tf.Variable(tf.zeros([36]))
# y = tf.matmul(x, W) + b
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# session.run(tf.global_variables_initializer())
# tf.train.start_queue_runners(sess=session)
#
# for i in range(200):
#     label, img = session.run([label_batch, img_batch])
#     train_step.run(feed_dict={x: img, y_: label})
#
# saver = tf.train.Saver()
# saver.save(session, 'result/model.chkp')
