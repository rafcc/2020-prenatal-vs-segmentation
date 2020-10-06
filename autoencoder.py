#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, input_channel_count, output_channel_count, size, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = size
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = (3, 3)
        self.CONV_STRIDE = 1
        self.CONV_PADDING = (1, 1)
        self.input_channel = input_channel_count
        self.output_channel = output_channel_count
        # exception
        if  first_layer_filter_count<8:
             first_layer_filter_count=8
        self.first_filter = first_layer_filter_count

    def create_model(self, vgg_on=False, dec_new=False, fine_tuning=False, pretrained_modelpath=None):
        # encoder
        encoder, filter_count = self.create_encoder()
        if fine_tuning:
            encoder.load_weights(pretrained_modelpath, by_name=True)
            encoder.trainable=False

        # knot
        x = self.INPUT_IMAGE_SIZE//(2**4)
        dense_len = x*x*filter_count
        knot_dense = self.add_knot(dense_len, encoder.output)
        if vgg_on:
            # add knot_dense + vgg_dense
            vgg_input = Input((224,224,3))
            knot_dense = self.add_vgg_feature(dense_len, vgg_input, knot_dense)
        knot_reshape = Reshape((x,x,filter_count))(knot_dense)

        # decoder
        if dec_new:
            decoder = self.add_new_decoder(filter_count, knot_reshape)
        else:
            decoder = self.add_decoder(filter_count, knot_reshape)
        
        # make Model
        if vgg_on:
            self.MODEL = Model(inputs=[encoder.input, vgg_input], outputs=decoder)
        else:
            self.MODEL = Model(inputs=encoder.inputs, outputs=decoder)
        return self.MODEL

    def create_encoder(self):
        # input 
        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, self.input_channel))
        with tf.name_scope("enc"):
            # encoder
            # (128 x 128 x N) →　(64 x 64 x N/2)　→　 (32 x 32 x N/4)　→　(16 x 16 x N/8)
            enc = inputs
            for n in range(4):
                filter_count = self.first_filter//(2**n)
                enc = self._add_encoding_layer(filter_count, n, enc)
            encoder = Model(inputs=inputs, outputs=enc)
        return encoder, filter_count


    def add_knot(self, dense_len, sequence):
        with tf.name_scope("knot"):
            # knot
            knot_flat = Flatten()(sequence)
            knot_dense = Dense(dense_len, activation='relu', name='knot1_dense')(knot_flat)
        return knot_dense

    def add_vgg_feature(self, dense_len, vgg_input, sequence):
        json_string = open(os.path.join("nn/model_data/vgg16_model.json")).read()
        vgg_model = model_from_json(json_string)
        vgg_model.trainable=False
        vgg = vgg_model(inputs=vgg_input)
        vgg = Flatten()(vgg)
        vgg = Dense(dense_len,activation='relu')(vgg)
        new_sequence = concatenate([sequence,vgg]) # now, len(sequence) == len(vgg)
        new_sequence = Dense(dense_len, activation='relu')(new_sequence)
        return new_sequence

    def add_decoder(self, filter_count, sequence,):
        with tf.name_scope("dec"):
            # decoder
            # (16 x 16 x N/8)
            dec1 = self._add_decoding_layer(filter_count, True, sequence)

            # (32 x 32 x N/4)
            filter_count = self.first_filter//8
            dec2 = self._add_decoding_layer(filter_count, True, dec1)

            # (64 x 64 x N/2)
            filter_count = self.first_filter//4
            dec3 = self._add_decoding_layer(filter_count, True, dec2)

            # (128 x 128 x N)
            filter_count = self.first_filter//2
            dec4 = self._add_decoding_layer(filter_count, True, dec3)

            # (256 x 256 x output_channel_count)
            decoder = Conv2D(self.output_channel, (3, 3),  activation='sigmoid', padding='same')(dec4)
            return decoder

    def add_new_decoder(self,  filter_count, sequence,):
        with tf.name_scope("dec_new"):
            # decoder
            # (16 x 16 x N/8)
            dec1 = self._add_decoding_layer(filter_count, True, sequence)

            # (32 x 32 x N/4)
            filter_count = self.first_filter//8
            dec2 = self._add_decoding_layer(filter_count, True, dec1)

            # (64 x 64 x N/2)
            filter_count = self.first_filter//4
            dec3 = self._add_decoding_layer(filter_count, True, dec2)

            # (128 x 128 x N)
            filter_count = self.first_filter//2
            dec4 = self._add_decoding_layer(filter_count, True, dec3)

            # (256 x 256 x output_channel_count)
            decoder = Conv2D(self.output_channel, (3, 3),  activation='sigmoid', padding='same')(dec4)
        return decoder

    def _add_encoding_layer(self, filter_count, n, sequence):
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE, activation='relu', padding='same',name="enc_conv2d_"+str(n))(sequence)
        new_sequence = MaxPooling2D((2, 2),  padding='same', name="enc_maxpool_"+str(n))(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE, activation='relu', padding='same')(sequence)
        new_sequence = UpSampling2D((2, 2))(new_sequence)
        return new_sequence

    def get_model(self, vgg_input=None, dec_new=False):
        return self.MODEL