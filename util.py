from PIL import Image
from PIL import ImageFilter


def show_img_from_array(img_data):
    """
    显示图片
    :param img_data:
    :return:
    """
    img = Image.fromarray(img_data)
    img.show()


def save_img_from_array(img_data, img_save_path):
    """
    保存图片
    :param img_data:
    :return:
    """
    img = Image.fromarray(img_data)
    img.save(img_save_path)


def sharpen_from_img_array(img_array):
    """
    锐化
    """
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.SHARPEN)
    img.show()


def edge_enhance_from_img_array(img_array):
    """
    边界加强
    """
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    img.show()
