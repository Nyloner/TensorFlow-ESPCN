# TensorFlow-ESPCN

超分辨率技术（Super-Resolution）是指从观测到的低分辨率图像重建出相应的高分辨率图像，是计算机视觉领域中的一个非常经典问题，在军事、医学、公共安全、计算机视觉等方面都有着重要的应用前景和应用价值。

本文采用ESPCN(Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network，CVPR 2016)算法，基于TensorFlow实现图像超分辨率重建。

## ESPCN

ESPCN是一种在低分辨率图像上直接计算卷积得到高分辨率图像的高效率方法，如下图所示，网络的输入是原始的低分辨率图像，通过三个卷积层后，得到通道数为r^2，大小与输入图像一致的特征图像，再将特征图像中每个像素的r^2个通道重新排列成一个r * r的区域，对应高分辨率图像中一个r * r子图像，这样，一个H * W * r^2的特征图像，就被重建成一个rH * rW * 1的高分辨率图像了。

<img src="https://nyloner.cn/static/files/ESPCN%E5%8E%9F%E7%90%86.jpg" style="height:200px;width:700px">

### 网络结构

- 第一层卷积网络：输入(lr image) height * width * image_channel，filter size 为 5 * 5 ，深度为 64，输出为 height * width * 64
- 第二层卷积网络：输入 height * width * 64，filter size 为 3 * 3，深度为 32，输出为 height * width * 32
- 第三层亚像素卷积层：输入 height * width * 32，filter size 为 3 * 3，深度为 r * r * image_channel，卷积操作输出为 height * width * (r * r * image_channel)，最后再经过PS重排，输出 (r * height) * (width * r) * image_channel

TensorFlow 实现
```Python
# 第一层卷积，filter size 5*5*64
# input image_height*image_width*image_channel
# output image_height*image_width*64
with tf.variable_scope('layer1-conv1'):
    conv1_weights = tf.get_variable(
        'weights',
        shape=[5, 5, self.image_channel, 64],
        initializer=tf.truncated_normal_initializer(
            stddev=np.sqrt(2.0 / 25 / 3))
    )
    conv1_bias = tf.get_variable(
        'bias',
        shape=[64],
        initializer=tf.constant_initializer(0.0)
    )
    conv1_temp = tf.nn.bias_add(tf.nn.conv2d(
        self.images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'), conv1_bias)
    conv1 = tf.nn.tanh(conv1_temp)

# 第二层卷积，filter size 3*3*32
# input image_height*image_width*64
# output image_height*image_width*32
with tf.variable_scope('layer2-conv2'):
    conv2_weights = tf.get_variable(
        'weights',
        shape=[3, 3, 64, 32],
        initializer=tf.truncated_normal_initializer(
            stddev=np.sqrt(2.0 / 9 / 64))
    )
    conv2_bias = tf.get_variable(
        'bias',
        shape=[32],
        initializer=tf.constant_initializer(0.0)
    )
    conv2_temp = tf.nn.bias_add(tf.nn.conv2d(
        conv1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'), conv2_bias)
    conv2 = tf.nn.tanh(conv2_temp)

# 第三层卷积，filter size 3*3*ratio*ratio*image_channel
# input image_height*image_width*32
# output image_height*image_width*ratio*ratio*image_channel
with tf.variable_scope('layer3-conv3'):
    conv3_weights = tf.get_variable(
        'weights',
        shape=[3, 3, 32,
                self.ratio * self.ratio * self.image_channel],
        initializer=tf.truncated_normal_initializer(
            stddev=np.sqrt(2.0 / 9 / 32))
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
```

ESPCN代码实现：https://github.com/Nyloner/TensorFlow-ESPCN/blob/master/espcn.py

## 训练模型

### 准备训练数据

将大图裁剪为 17 * 17 大小的子图像集

```Python
RATIO = 3  # 放大比例
IMAGE_SIZE = 17  # 训练图片大小
STRIDE = 8  # 裁剪步长
IMAGE_CHANNEl = 3  # 图片通道

def preprocess_img(file_path):
    """
    处理图片
    :param file_path:
    :return:
    """
    img = Image.open(file_path)
    # 缩放为 RATIO 的整数倍
    img_label = img.resize(
        ((img.size[0] // RATIO) * RATIO, (img.size[1] // RATIO) * RATIO))
    img_input = img_label.resize(
        (img_label.size[0] // RATIO, img_label.size[1] // RATIO))
    return np.asarray(img_input), np.asarray(img_label)

def make_sub_data(img_list):
    """
    将大图裁剪为小图
    :param img_list:
    :return:
    """
    sub_input_sequence = []
    sub_label_sequence = []
    for file_path in img_list:
        input_, label_ = preprocess_img(file_path)
        h, w, c = input_.shape
        if c != IMAGE_CHANNEl:
            continue
        # 裁剪图片
        for x in range(0, h - IMAGE_SIZE + 1, STRIDE):
            for y in range(0, w - IMAGE_SIZE + 1, STRIDE):
                # input
                sub_input = input_[x: x + IMAGE_SIZE,
                            y: y + IMAGE_SIZE]
                sub_input = sub_input.reshape(
                    [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEl])
                sub_input = sub_input / 255.0
                sub_input_sequence.append(sub_input)

                # label
                label_x = x * RATIO
                label_y = y * RATIO
                sub_label = label_[label_x: label_x + IMAGE_SIZE * RATIO,
                            label_y: label_y + IMAGE_SIZE * RATIO]
                sub_label = sub_label.reshape(
                    [IMAGE_SIZE * RATIO, IMAGE_SIZE * RATIO, IMAGE_CHANNEl])
                sub_label = sub_label / 255.0
                sub_label_sequence.append(sub_label)

    return sub_input_sequence, sub_label_sequence
```

将处理后的数据保存为HDF5(一种针对大量数据进行组织和存储的文件格式)格式

```Python
def make_data_hf(input_, label_):
    """
    保存训练数据
    """
    checkpoint_dir = 'checkpoint'
    if not os.path.isdir(os.path.join(os.getcwd(), checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), checkpoint_dir))
    savepath = os.path.join(
        os.getcwd(), checkpoint_dir + '/train_data.h5')
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)


def prepare_data(dataset='images'):
    """
    :param dataset:
    :return:
    """
    data_dir = os.path.join(os.getcwd(), dataset)
    # 获取文件夹中的图片
    filenames = glob.glob(os.path.join(data_dir, "*.jpg"))
    filenames += glob.glob(os.path.join(data_dir, "*.jpeg"))

    # 裁剪图片
    sub_input_sequence, sub_label_sequence = make_sub_data(filenames)

    arrinput = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)
    # 保存为HDF5文件
    make_data_hf(arrinput, arrlabel)
```

数据处理代码：https://github.com/Nyloner/TensorFlow-ESPCN/blob/master/prepare_data.py

### 训练

训练代码：
https://github.com/Nyloner/TensorFlow-ESPCN/blob/master/espcn.py
https://github.com/Nyloner/TensorFlow-ESPCN/blob/master/train.py


## 重建

### 重建效果

- LR Image

<img src="https://nyloner.cn/static/files/lr_image.jpg" style="height:100px;width:100px">

- HR Image

<img src="https://nyloner.cn/static/files/hr_image.jpg" style="height:300px;width:300px">

- SR Image

<img src="https://nyloner.cn/static/files/sr_image.jpg" style="height:300px;width:300px">











