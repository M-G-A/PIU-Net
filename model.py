#!/usr/bin/env python
# coding: utf8

import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import losses
from ccnn_layers import CConv2D
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50V2

def perceptual_net(input_size):
    backbone = VGG16(include_top=False, weights='weights/vgg16_notop.h5', input_shape=(256,512,3), pooling='avg')
    input = Input(input_size)
    input_d = AveragePooling2D(pool_size=(4, 4))(input)
    feature = backbone(input_d)
    #feature = MaxPooling2D(pool_size=(1, 512),data_format='channels_last')(feature)
    #feature = keras.sum(feature,axis=-1)
    model = Model(input = input, output = feature)
    #print(model.summary())
    #exit(1)
    return model

def combinedLoss(input_size):
    #TODO: test different backbones for perceptualoss
    #model = VGG16(include_top=False, weights='weights/vgg16_notop.h5', input_shape=input_size, pooling='avg')
    #model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_size, pooling='max')

    model = perceptual_net(input_size)
    delta = 1.0/input_size[0]
    weights = np.expand_dims(np.repeat([np.sin(np.arange(0.5*delta,1,delta))], input_size[1], axis = 0).T,axis=0)
    weights = keras.constant(weights)

    def lossFunction(y_true, y_pred):
        l1 = losses.mean_squared_error(y_true,y_pred)
        #TODO: for x,y Cropping2D, create sectionwise loss, with overlap / pano
        #out_true = model(y_true)
        #out_pred = model(y_pred)
        #print(out_true.shape)

        #l2 = keras.sum(losses.mean_squared_error(out_true,out_pred))
        l2 = keras.sum(keras.abs(model(y_true)-model(y_pred)))
        #l2 = losses.mean_squared_error(out_true,out_pred)
        #l2 = keras.expand_dims(l2, axis=-1)
        #l2 = UpSampling2D(size=(32, 32))(l2)
        #l2 = keras.squeeze(l2, axis=-1)

        #return (50000*l1+0.1*l2)
        return Lambda(multiply)([(l1+l2), weights])
    return lossFunction

def unet(pretrained_weights = None,input_size = (2048,4096,1),n=16):
    inputs = Input(input_size)
    conv1 = CConv2D(n, (1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = CConv2D(n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = CConv2D(2*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = CConv2D(2*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = CConv2D(4*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = CConv2D(4*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = CConv2D(8*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = CConv2D(8*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = CConv2D(16*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = CConv2D(16*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = CConv2D(8*n, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = CConv2D(8*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = CConv2D(8*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = CConv2D(4*n, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = CConv2D(4*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = CConv2D(4*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = CConv2D(2*n, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = CConv2D(2*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = CConv2D(2*n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = CConv2D(n, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = CConv2D(n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = CConv2D(n, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = CConv2D(2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = CConv2D(3, (1,1), activation = 'sigmoid')(conv9)

    #model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs,input_shape=input_size, pooling=None)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = combinedLoss(input_size=input_size), metrics = ['accuracy'])
    #combinedLoss(inputs,input_size)

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


