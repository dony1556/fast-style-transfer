import tensorflow as tf
import numpy as np
import scipy.io
import pdb

#imagenet 给定的均值（训练的几千万张图的每个通道的均值）
MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])


def net(data_path, input_image):
    '''
    重新定义vgg19网络模型
    '''
    #提取vgg19的网络模型
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    #对数据进行处理
    data = scipy.io.loadmat(data_path)
    #
    mean = data['normalization'][0][0][0]
    #内部存储好的结构
    mean_pixel = np.mean(mean, axis=(0, 1))
    #权重
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        #前四个字母决定执行了什么操作
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # 对于.matw文件的convnet: weights are [width, height, in_channels, out_channels]
            # 对于tensorflow: weights are [height, width, in_channels, out_channels]
            #用numpy的transpose把前两维的数据位置调换一下
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    """
    @Descripttion: 卷积层
    @param {type} 
    @return: 
    '''    
    """
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    '''
    @Descripttion: 池化层
    @param {type} 
    @return: 
    '''
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')


def preprocess(image):
    """
    对图像进行预处理，减除均值
    """
    return image - MEAN_PIXEL


def unprocess(image):
    """
    对图像进行逆向预处理，加上均值
    """
    return image + MEAN_PIXEL



