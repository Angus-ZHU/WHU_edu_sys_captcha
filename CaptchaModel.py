import tensorflow as tf
import tflearn
import h5py
import numpy as np
import cv2
import os


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
        net = tflearn.reshape(net, [-1, 7 * 7 * 64])
        net = tflearn.fully_connected(net, 1024, activation='relu')
        # Dropout
        net = tflearn.dropout(net, 0.5)
        # Readout Layer
        W_fc2 = CaptchaModel.weight_variable([1024, 36])
        b_fc2 = CaptchaModel.bias_variable([36])
        net = tf.matmul(net, W_fc2) + b_fc2
        regression = tflearn.regression(net, name='label', learning_rate=0.0001,
                                        loss='softmax_categorical_crossentropy')
        model = tflearn.DNN(network=regression, tensorboard_dir='tmp/tf.log', tensorboard_verbose=1)
        return model

    @staticmethod
    def get_model_with_meta_data():
        with tf.Graph().as_default():
            model = CaptchaModel.get_model()
            model.load('result/model.tflearn')
        return model

    @staticmethod
    def create_hdf5_data_set():
        # must be modified if to use a different training data set
        tagged_img_dir = 'train/tagged_img'
        mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                   'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        label_data_set = []
        img_data_set = []
        for img_name in os.listdir(tagged_img_dir):
            img_name_stripped = img_name.replace('.png', '')
            tag = img_name_stripped.split('_')[1]
            img = cv2.imread(os.path.join(tagged_img_dir, img_name))
            upper_not_red_bgr = np.array([255, 255, 255])
            lower_not_red_bgr = np.array([100, 100, 100])
            binary_img = cv2.inRange(img, lower_not_red_bgr, upper_not_red_bgr)
            tag_array = np.zeros(36, dtype=np.int8)
            tag_array[mapping.index(tag)] = 1
            label_data_set.append(tag_array)
            img_data_set.append(binary_img)
        with h5py.File('train/train.h5', 'w') as f:
            f.create_dataset('img', data=img_data_set)
            f.create_dataset('label', data=label_data_set)

    @staticmethod
    def train_model():
        with tf.Graph().as_default():
            model = CaptchaModel.get_model()
            with h5py.File('train/train.h5', 'r') as f:
                model.fit(f['img'], f['label'], n_epoch=8, shuffle=True)
            model.save('result/model.tflearn')