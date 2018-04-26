import tensorflow as tf
import os
from PIL import Image

from espcn import ESPCN
import prepare_data
import util

def generate(img_path):
    """
    图像超分辨率重建
    :return:
    """
    lr_image, ori_image = prepare_data.preprocess_img(img_path)
    # lr_image = ori_image
    image_height, image_width, _ = lr_image.shape
    with tf.Session() as sess:
        espcn = ESPCN(sess,
                      is_train=False,
                      image_height=image_height,
                      image_width=image_width,
                      image_channel=prepare_data.IMAGE_CHANNEl,
                      ratio=prepare_data.RATIO)
        sr_image = espcn.generate(lr_image / 255.0)

    # lr image
    util.show_img_from_array(lr_image)
    # original image
    util.show_img_from_array(ori_image)
    # sr image
    util.show_img_from_array(sr_image)

    util.sharpen_from_img_array(sr_image)

    # util.edge_enhance_from_img_array(sr_image)

if __name__ == '__main__':
    generate('./images/butterfly_GT.jpg')
