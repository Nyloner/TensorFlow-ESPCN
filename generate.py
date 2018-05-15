import tensorflow as tf
import os
from PIL import Image
import glob

from espcn import ESPCN
import prepare_data
import util

TEST_IMAGE_DIR = './test_images/Set5/'
TEST_RESULT_DIR = './result/Set5/'


def generate(img_name):
    """
    图像超分辨率重建
    :return:
    """
    lr_image, ori_image = prepare_data.preprocess_img(TEST_IMAGE_DIR+img_name)
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
    # util.show_img_from_array(lr_image)
    util.save_img_from_array(
        lr_image, TEST_RESULT_DIR+img_name.split('.')[0]+'_lr.'+img_name.split('.')[-1])
    # original image
    # util.show_img_from_array(ori_image)
    util.save_img_from_array(ori_image, TEST_RESULT_DIR +
                             img_name.split('.')[0]+'_hr.'+img_name.split('.')[-1])
    # sr image
    # util.show_img_from_array(sr_image)
    util.save_img_from_array(sr_image, TEST_RESULT_DIR +
                             img_name.split('.')[0]+'_sr.'+img_name.split('.')[-1])


if __name__ == '__main__':
    img_list = [filename for filename in os.listdir(
        TEST_IMAGE_DIR) if filename.endswith('bmp')]
    for filename in img_list:
        generate(filename)
