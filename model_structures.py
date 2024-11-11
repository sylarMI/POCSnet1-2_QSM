import tensorflow as tf
from keras import backend as K
import numpy as np
from custom_layers import *
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.layers import Input, Add, Concatenate

from keras.models import Model

ker_size = 3

def unet_at1(opt, data_size, f_size=8, ker_size=(3, 3, 3)):
    act = 'relu'
    conv_1c0 = Conv3D(f_size, kernel_size=(1, 1, 1), padding='same', activation=act)
    conv_1c1 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c12 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    pool1 = Conv3D(2 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)  # AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_1c2 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c22 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    pool2 = Conv3D(4 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)  # AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_1c3 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c32 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    pool3 = Conv3D(8 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)
    
    conv_1c4 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c42 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    pool4 = Conv3D(16 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)
    upS4 = Conv3DTranspose(8 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)
    
    conv_1e3 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e32 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e33 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    upS3 = Conv3DTranspose(4 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)
    
    conv_1e2 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e22 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e23 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    upS2 = Conv3DTranspose(2 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)
    
    # UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_1e1 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e12 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e13 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)

    upS1 = Conv3DTranspose(f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)  # UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_1e0 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e02 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e03 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)

    conv_output_tc = Conv3D(data_size, kernel_size=(1, 1, 1), padding='same', activation='tanh')
    
    input_img = Input(shape=opt['in_shape'])
    # unet start
    c0 = conv_1c0(input_img)
    c1 = conv_1c1(c0)
    c1 = conv_1c12(c1) + c0

    c1_ = pool1(c1)
    c2 = conv_1c2(c1_)
    c2 = conv_1c22(c2) + c1_

    c2_ = pool2(c2)
    c3 = conv_1c3(c2_)
    c3 = conv_1c32(c3) + c2_

    c3_ = pool3(c3)
    c4 = conv_1c4(c3_)
    c4 = conv_1c42(c4) + c3_

    c5 = pool4(c4)

    e1 = upS4(c5)
    e12 = conv_1e3(tf.concat([e1, c4], -1))
    e13 = conv_1e32(e12)
    e13 = conv_1e33(e13) + e1

    e2 = upS3(e13)
    e22 = conv_1e2(tf.concat([e2, c3], -1))
    e23 = conv_1e22(e22)
    e23 = conv_1e23(e23) + e2

    e3 = upS2(e23)
    e32 = conv_1e1(tf.concat([e3, c2], -1))
    e33 = conv_1e12(e32)
    e33 = conv_1e13(e33) + e3

    e4 = upS1(e33)
    e42 = conv_1e0(tf.concat([e4, c1], -1))
    e43 = conv_1e02(e42)
    e43 = conv_1e03(e43) + e4

    output = conv_output_tc(e43)

    model_compile = Model(inputs=[input_img], outputs=[output])
    return model_compile

def unet_at2(opt, data_size, f_size=16, ker_size=(3, 3, 3)):
    act = 'relu'
    conv_1c0 = Conv3D(f_size, kernel_size=(1, 1, 1), padding='same', activation=act)
    conv_1c1 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c12 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    pool1 = Conv3D(2 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)  # AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_1c2 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c22 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    pool2 = Conv3D(4 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)  # AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_1c3 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c32 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    pool3 = Conv3D(8 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)
    
    conv_1c4 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1c42 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    pool4 = Conv3D(16 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                   activation=act)
    upS4 = Conv3DTranspose(8 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)
    
    conv_1e3 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e32 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e33 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation=act)
    upS3 = Conv3DTranspose(4 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)
    
    conv_1e2 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e22 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e23 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation=act)
    upS2 = Conv3DTranspose(2 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)
    
    # UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_1e1 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e12 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e13 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation=act)

    upS1 = Conv3DTranspose(f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           activation=act)  # UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_1e0 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e02 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)
    conv_1e03 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation=act)

    conv_output_tc = Conv3D(data_size, kernel_size=(1, 1, 1), padding='same', activation='tanh')
    
    input_img = Input(shape=opt['in_shape'])
    # unet start
    c0 = conv_1c0(input_img)
    c1 = conv_1c1(c0)
    c1 = Add()([conv_1c12(c1) , c0])

    c1_ = pool1(c1)
    c1_ = conv_1c2(c1_)
    c2 = Add()([conv_1c22(c1_) , c1_])

    c2_ = pool2(c2)
    c2_ = conv_1c3(c2_)
    c3 = Add()([conv_1c32(c2_) , c2_])

    c3_ = pool3(c3)
    c3_ = conv_1c4(c3_)
    c4 = Add()([conv_1c42(c3_), c3_])

    c5 = pool4(c4)

    e1 = upS4(c5)
    e12 = conv_1e3(Concatenate()([e1, c4]))
    e13 = conv_1e32(e12)
    e13 = Add()([conv_1e33(e13), e12])

    e2 = upS3(e13)
    e22 = conv_1e2(Concatenate()([e2, c3]))
    e23 = conv_1e22(e22)
    e23 = Add()([conv_1e23(e23), e22])

    e3 = upS2(e23)
    e32 = conv_1e1(Concatenate()([e3, c2]))
    e33 = conv_1e12(e32)
    e33 = Add()([conv_1e13(e33), e32])

    e4 = upS1(e33)
    e42 = conv_1e0(Concatenate()([e4, c1]))
    e43 = conv_1e02(e42)
    e43 = Add()([conv_1e03(e43), e42])

    output = conv_output_tc(e43)

    model_compile = Model(inputs=[input_img], outputs=[output])
    return model_compile


def cg_br_grad_model(x,x_init,smv,trun,opt):
    x_ref = Input(shape=opt['in_shape'])
    cus_grad = CustomLayer_br(x_init,x,smv,trun,opt)(x_ref)
    model_compile = Model(inputs=[x_ref], outputs=[cus_grad])
    return model_compile

