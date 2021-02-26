# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

l2 = tf.keras.regularizers.l2

def fix_adaIN(inputs, style, epsilon=1e-5):
    
    _, H, W, C = inputs.get_shape()
    C_num = C // 3
    img_buf = []

    in_mean_var = [tf.nn.moments(inputs[:, :, :, i*3:(i+1)*3], axes=[1,2], keepdims=True) for i in range(C_num)]
    in_mean = [in_mean_var[i][0] for i in range(C_num)]
    in_var = [in_mean_var[i][1] for i in range(C_num)]
    st_mean, st_var = tf.nn.moments(style, axes=[1,2], keepdims=True)

    in_std = [tf.sqrt(in_var[i] + epsilon) for i in range(C_num)]
    st_std = tf.sqrt(st_var + epsilon)

    img = [st_std * (style - in_mean[i]) / in_std[i] + st_mean for i in range(C_num)]
    img = tf.concat(img, -1)

    return img

def adaIN(inputs, style, epsilon=1e-5):

    in_mean, in_var = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
    st_mean, st_var = tf.nn.moments(style, axes=[1,2], keepdims=True)
    in_std, st_std = tf.sqrt(in_var + epsilon), tf.sqrt(st_var + epsilon)

    return st_std * (style - in_mean) / in_std + st_mean

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def conv_block(input, filters, dilation_rate, weight_decay):
    h = tf.keras.layers.Conv2D(filters=filters // 2,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(input)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation_rate,dilation_rate],[dilation_rate,dilation_rate],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=filters // 2,
                               kernel_size=3,
                               strides=1,
                               dilation_rate=dilation_rate,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h += input
    h = tf.keras.layers.ReLU()(h)

    return h

def style_conv_block(input, filters, weight_decay, repeat):
    s = tf.keras.layers.Conv2D(filters=filters // 2,
                               kernel_size=1,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(input)
    s = InstanceNormalization()(s)
    s = tf.keras.layers.ReLU()(s)

    s = tf.pad(s, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=filters // 2,
                               kernel_size=3,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)
    s = InstanceNormalization()(s)
    s = tf.keras.layers.ReLU()(s)

    s = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)
    s = InstanceNormalization()(s)
    if repeat == 0:
        s = tf.keras.layers.ReLU()(s)
    else:
        s += input
        s = tf.keras.layers.ReLU()(s)

    return s

def V9_generator(input_shape=(256, 256, 3),
                 style_shape=(256, 256, 3),
                 weight_decay=0.0000002,
                 repeat=4):
    # 앞단의 정보를 최대한 많이 handling 해야힐것 같다. 파라미터의 수도 많을것 같은데??
    h = inputs = tf.keras.Input(input_shape)
    s = styles = tf.keras.Input(style_shape)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)

    s = tf.pad(s, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)
    s1 = s

    h = h + s
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    for i in range(repeat):
        h = conv_block(h, 64, i + 1, weight_decay)
        s = style_conv_block(s, 64, weight_decay, i)
        h +=s

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)

    s = tf.pad(s, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)
    s2 = s

    h = h + s
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    for i in range(repeat):
        h = conv_block(h, 128, i + 1, weight_decay)
        s = style_conv_block(s, 128, weight_decay, i)
        h += s

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)

    s = tf.pad(s, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)

    h = h + s
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

    for i in range(repeat-2):
        h = conv_block(h, 256, i + 1, weight_decay)
        s = style_conv_block(s, 256, weight_decay, i)
        h += s

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    for i in range(repeat-1):
        h = conv_block(h, 128, i + 1, weight_decay)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s2)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    for i in range(repeat-1):
        h = conv_block(h, 64, i + 1, weight_decay)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s1)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7)(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=[inputs, styles], outputs=h)

def style_network(input_shape=(256, 256, 3),
                  style_shape=(256, 256, 3),
                  weight_decay=0.0000002,
                  repeat=4):
    h = inputs = tf.keras.Input(input_shape)
    s = styles = tf.keras.Input(style_shape)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)

    s = tf.pad(s, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)
    s1 = s

    h = h + s
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    for i in range(repeat):
        h = conv_block(h, 64, i + 1, weight_decay)
        s = style_conv_block(s, 64, weight_decay, i)
        h +=s

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)

    s = tf.pad(s, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)
    s2 = s

    h = h + s
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    for i in range(repeat):
        h = conv_block(h, 128, i + 1, weight_decay)
        s = style_conv_block(s, 128, weight_decay, i)
        h += s

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)

    s = tf.pad(s, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    s = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(s)

    h = h + s
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

    #for i in range(repeat-2):
    #    h = conv_block(h, 256, i + 1, weight_decay)
    #    s = style_conv_block(s, 256, weight_decay, i)
    #    h += s

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=3)(h)
    h = tf.keras.layers.Flatten()(h)

    return tf.keras.Model(inputs=[inputs, styles], outputs=h)

def discriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):

    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)

