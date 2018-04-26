import tensorflow as tf

from espcn import ESPCN
import prepare_data


def train():
    """
    训练
    :return:
    """
    prepare_data.prepare_data()
    with tf.Session() as sess:
        espcn = ESPCN(sess,
                      is_train=True,
                      image_height=prepare_data.IMAGE_SIZE,
                      image_width=prepare_data.IMAGE_SIZE,
                      image_channel=prepare_data.IMAGE_CHANNEl,
                      ratio=prepare_data.RATIO)
        espcn.train()


if __name__ == '__main__':
    train()
