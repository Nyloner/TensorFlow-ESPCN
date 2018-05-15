"""
ESPCN算法
"""
import tensorflow as tf
import time
import os
import h5py
import numpy as np


class ESPCN:
    def __init__(self, sess, is_train, image_height, image_width, image_channel, ratio):
        self.sess = sess
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.ratio = ratio
        self.is_train = is_train

        self.batch_size = 100
        self.learning_rate = 0.0001
        self.epoch = 100
        self.checkpoint_dir = 'checkpoint'
        self.train_data_path = self.checkpoint_dir + '/' + 'train_data.h5'

        self.create_network()

    def create_network(self):
        """
        创建卷积神经网络
        :return:
        """
        # input
        self.images = tf.placeholder(
            tf.float32, [None, self.image_height, self.image_width, self.image_channel], name='images')

        self.labels = tf.placeholder(
            tf.float32, [None, self.image_height * self.ratio,
                         self.image_width * self.ratio, self.image_channel],
            name='labels')
        self.pred = self.inference()

        # 损失函数
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def inference(self):
        """
        定义卷积神经网络
        :return:
        """
        # 第一层卷积，filter size 5*5*64
        # input image_height*image_width*image_channel
        # output image_height*image_width*64

        with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE):
            conv1_weights = tf.get_variable(
                'weights',
                shape=[5, 5, self.image_channel, 64],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1)
            )
            conv1_bias = tf.get_variable(
                'bias',
                shape=[64],
                initializer=tf.constant_initializer(0.0)
            )
            conv1_temp = tf.nn.bias_add(tf.nn.conv2d(
                self.images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'), conv1_bias)
            conv1 = tf.nn.relu(conv1_temp)

        # 第二层卷积，filter size 3*3*32
        # input image_height*image_width*64
        # output image_height*image_width*32
        with tf.variable_scope('layer2-conv2', reuse=tf.AUTO_REUSE):
            conv2_weights = tf.get_variable(
                'weights',
                shape=[3, 3, 64, 32],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1)
            )
            conv2_bias = tf.get_variable(
                'bias',
                shape=[32],
                initializer=tf.constant_initializer(0.0)
            )
            conv2_temp = tf.nn.bias_add(tf.nn.conv2d(
                conv1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'), conv2_bias)
            conv2 = tf.nn.relu(conv2_temp)

        # 第三层卷积，filter size 3*3*(ratio*ratio*image_channel)
        # input image_height*image_width*32
        # output image_height*image_width*ratio*ratio*image_channel
        with tf.variable_scope('layer3-conv3', reuse=tf.AUTO_REUSE):
            conv3_weights = tf.get_variable(
                'weights',
                shape=[3, 3, 32,
                       self.ratio * self.ratio * self.image_channel],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1)
            )
            conv3_bias = tf.get_variable(
                'bias',
                shape=[self.ratio * self.ratio * self.image_channel],
                initializer=tf.constant_initializer(0.0)
            )
            conv3 = tf.nn.bias_add(tf.nn.conv2d(
                conv2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME'), conv3_bias)

        # output (image_height*ratio)*(image_width*ratio)*image_channel
        return tf.nn.tanh(self.PS(conv3, self.ratio))

    def _phase_shift(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a * r, b * r, 1))

    def _phase_shift_gene(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))

    def PS(self, X, r):
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r)
                           for x in Xc], 3)
        else:
            X = tf.concat([self._phase_shift_gene(x, r)
                           for x in Xc], 3)
        return X

    def load_train_data(self):
        """
        加载训练数据
        :return:
        """
        with h5py.File(self.train_data_path, 'r') as hf:
            input_ = np.array(hf.get('input'))
            label_ = np.array(hf.get('label'))
        return input_, label_

    def save(self, step):
        """
        保存模型
        :param step:
        :return:
        """
        model_name = "ESPCN.model"
        model_dir = "%s_%s" % (
            "espcn", self.ratio)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_checkpoint(self):
        """
        加载训练模型
        :return:
        """
        model_dir = "%s_%s" % (
            "espcn", self.ratio)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("Checkpoint Loading Success! %s\n" % ckpt_path)
        else:
            print("Checkpoint Loading Failed! \n")

    def train(self):
        """
        训练
        :return:
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        trainable = tf.trainable_variables()
        self.train_op = optimizer.minimize(self.loss, var_list=trainable)
        tf.global_variables_initializer().run()

        self.load_checkpoint()

        current_time = time.time()
        input_, label_ = self.load_train_data()
        input_size = len(input_)
        step = 0

        for ep in range(self.epoch):
            for index in range(input_size - self.batch_size):
                batch_images = input_[
                    index * self.batch_size: (index + 1) * self.batch_size]
                batch_labels = label_[
                    index * self.batch_size: (index + 1) * self.batch_size]
                if len(batch_images) != self.batch_size:
                    continue
                _, loss_value = self.sess.run([self.train_op, self.loss],
                                              feed_dict={self.images: batch_images, self.labels: batch_labels})
                step += 1
                if step % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time consumed: [%4.4f], loss value: [%.8f]" % (
                        (ep + 1), step, time.time() - current_time, loss_value))
                if step % 500 == 0:
                    self.save(step)

    def generate(self, lr_image):
        """
        生成高分辨率图
        :param lr_image:
        :return:
        """
        self.load_checkpoint()
        result = self.pred.eval({self.images: lr_image.reshape(
            1, self.image_height, self.image_width, self.image_channel)})
        result[result < 0] = 0
        sr_image = np.squeeze(result) * 255.
        return np.uint8(sr_image)
