import tensorflow as tf
import tflearn
import h5py


class CaptchaModel(object):
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def get_model():
        x = tflearn.input_data(shape=[None, 28, 28], dtype=tf.float32, name='img')
        x = tflearn.reshape(x, [-1, 28, 28, 1])
        # First Convolutional Layer
        net = tflearn.conv_2d(x, 32, 5, strides=[1, 1, 1, 1], activation='relu', name='conv1')
        net = tflearn.max_pool_2d(net, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='maxpool1')
        # Second Convolutional Layer
        net = tflearn.conv_2d(net, 64, 5, activation='relu', name='conv2')
        net = tflearn.max_pool_2d(net, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='maxpool2')
        # Densely Connected Layer
        net = tflearn.reshape(net, [-1, 7*7*64])
        net = tflearn.fully_connected(net, 1024, activation='relu')
        # Dropout
        net = tflearn.dropout(net, 0.5)
        # Readout Layer
        W_fc2 = CaptchaModel.weight_variable([1024, 36])
        b_fc2 = CaptchaModel.bias_variable([36])
        net = tf.matmul(net, W_fc2) + b_fc2
        regression = tflearn.regression(net, name='label', learning_rate=0.0001, loss='softmax_categorical_crossentropy')

        model = tflearn.DNN(network=net, tensorboard_dir='tmp/tf.log', tensorboard_verbose=3)
        return model


if __name__ == '__main__':
    with tf.Graph().as_default():
        model = CaptchaModel.get_model()
        # with h5py.File('train/train.h5', 'r') as f:
        with h5py.File('train/train_letter_tag.h5', 'r') as f:
            img = f['img']
            label = f['label']
            model.fit(f['img'], f['label'], n_epoch=8, shuffle=True, snapshot_epoch=True)
        model.save('result/model.1.1.tflearn')