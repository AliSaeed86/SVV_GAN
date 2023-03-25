"""The model definitions."""
from keras import objectives
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.models import Model

from keras.layers import Input, merge, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.core import Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
import numpy as np

import tensorflow as tf

from keras.layers.convolutional import Conv2D

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering('th')


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f, k, k, border_mode=border_mode, subsample=(s, s),
                         **kwargs)


def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    return Deconvolution2D(f, k, k, output_shape=output_shape,
                           subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(mode=mode, axis=axis, **kwargs)


def g_unet(in_ch, out_ch, nf, is_binary=False, name='unet'):
    """Define a U-Net.

    Input has shape in_ch x 256 x 256
    Parameters:
    - in_ch: the number of input channels;
    - out_ch: the number of output channels;
    - nf: the number of filters of the first layer;
    - is_binary: if is_binary is true, the last layer is followed by a sigmoid
    activation function, otherwise, a tanh is used.
    """
    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }

    # i = Input(shape=(in_ch, 256, 256))
    # in_ch+in_ch is for BV and OD we sent the combined masks to the model
    i = Input(shape=(2, 256, 256))

    # in_ch x 256 x 256
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 8 x 8

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 4 x 4

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 2 x 2

    conv8 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8, (None, nf*8, 2, 2), k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    x = merge([dconv1, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8, (None, nf*8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = merge([dconv2, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8, (None, nf*8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = merge([dconv3, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8, (None, nf*8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = merge([dconv4, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 4, (None, nf*4, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = merge([dconv5, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 32 x 32

    dconv6 = Deconvolution(nf * 2, (None, nf*2, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = merge([dconv6, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 64 x 64

    dconv7 = Deconvolution(nf, (None, nf, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = merge([dconv7, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 128 x 128

    dconv8 = Deconvolution(out_ch, (None, out_ch, 256, 256))(x)
    # dconv8 = Deconvolution(2, (None, 2, 256, 256))(x)

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv8)

    unet = Model(i, out, name=name)

    return unet


def g_vae(in_ch, out_ch, nf, latent_dim, is_binary=False, name='vae'):
    """
    Define a Variational Auto Encoder.

    Params:
    - in_ch: number of input channels.
    - out_ch: number of output channels.
    - nf: number of filters of the first layer.
    - latent_dim: the number of latent factors to use.
    - is_binary: whether the output is binary or not.
    # """
    # in_ch=1
    # out_ch=1
    # nf = 32
    # latent_dim=64
    # is_binary=True
    # name='vae'

    i = Input(shape=(in_ch, 256, 256))

    # in_ch x 256 x 256
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    conv1 = Convolution(nf, s=1)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    conv2 = Convolution(nf * 2, s=1)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    conv3 = Convolution(nf * 4, s=1)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    conv4 = Convolution(nf * 8, s=1)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    x = Dropout(0.5)(x)
    conv5 = Convolution(nf * 8, s=1)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    x = Dropout(0.5)(x)
    # nf*8 x 8 x 8

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    x = Dropout(0.5)(x)
    conv6 = Convolution(nf * 8, s=1)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    x = Dropout(0.5)(x)
    # nf*8 x 4 x 4

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    x = Dropout(0.5)(x)
    conv7 = Convolution(nf * 8, s=1)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    x = Dropout(0.5)(x)
    # nf*8 x 2 x 2

    conv8 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 1 x 1

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args

        if K.backend() == 'tensorflow':
            batch_size = K.shape(z_mean)[0]
        else:
            batch_size = z_mean.shape[0]

        epsilon = K.random_normal(
            shape=(batch_size, latent_dim), mean=0., std=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # encoder = Model(i, z, name='{0}_encoder'.format(name))    # ORIGINAL
    encoder = Model(i, z, name='{0}_encoder'.format(name))
# encoder.summary()

    ig = Input(shape=(latent_dim,))

    x = Dense(nf * 8)(ig)
    x = Reshape((nf * 8, 1, 1))(x)

    x = UpSampling2D(size=(2, 2))(x)
    dconv1 = Convolution(nf * 8, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    x = LeakyReLU(0.2)(dconv1)
    # nf*8 x 2 x 2

    x = UpSampling2D(size=(2, 2))(x)
    dconv2 = Convolution(nf * 8, s=1)(x)
    dconv2 = BatchNorm()(dconv2)
    x = LeakyReLU(0.2)(dconv2)
    x = Dropout(0.5)(x)
    # nf*8 x 4 x 4

    x = UpSampling2D(size=(2, 2))(x)
    dconv3 = Convolution(nf * 8, s=1)(x)
    dconv3 = BatchNorm()(dconv3)
    x = LeakyReLU(0.2)(dconv3)
    x = Dropout(0.5)(x)
    # nf*8 x 8 x 8

    x = UpSampling2D(size=(2, 2))(x)
    dconv4 = Convolution(nf * 8, s=1)(x)
    dconv4 = BatchNorm()(dconv4)
    x = LeakyReLU(0.2)(dconv4)
    x = Dropout(0.5)(x)
    # nf*8 x 16 x 16

    x = UpSampling2D(size=(2, 2))(x)
    dconv5 = Convolution(nf * 4, s=1)(x)
    dconv5 = BatchNorm()(dconv5)
    x = LeakyReLU(0.2)(dconv5)
    dconv5 = Convolution(nf * 4, s=1)(x)
    dconv5 = BatchNorm()(dconv5)
    x = LeakyReLU(0.2)(dconv5)
    # nf*4 x 32 x 32

    x = UpSampling2D(size=(2, 2))(x)
    dconv6 = Convolution(nf * 2, s=1)(x)
    dconv6 = BatchNorm()(dconv6)
    x = LeakyReLU(0.2)(dconv6)
    dconv6 = Convolution(nf * 2, s=1)(x)
    dconv6 = BatchNorm()(dconv6)
    x = LeakyReLU(0.2)(dconv6)
    # nf*2 x 64 x 64

    x = UpSampling2D(size=(2, 2))(x)
    dconv7 = Convolution(nf, s=1)(x)
    dconv7 = BatchNorm()(dconv7)
    x = LeakyReLU(0.2)(dconv7)
    dconv7 = Convolution(nf, s=1)(x)
    dconv7 = BatchNorm()(dconv7)
    x = LeakyReLU(0.2)(dconv7)
    # nf x 128 x 128

    x = UpSampling2D(size=(2, 2))(x)
    dconv8 = Convolution(nf, s=1)(x)
    dconv8 = BatchNorm()(dconv8)
    x = LeakyReLU(0.2)(dconv8)
    dconv8 = Convolution(nf, s=1)(x)
    dconv8 = BatchNorm()(dconv8)
    x = LeakyReLU(0.2)(dconv8)
    x = Convolution(out_ch, k=1, s=1)(x)

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(x)

    decoder = Model(ig, out, name='{0}_decoder'.format(name))
# decoder.summary()

    def vae_loss(a, ap):
        a_flat = K.batch_flatten(a)
        ap_flat = K.batch_flatten(ap)

        # reconstruction loss commented by ali
        L_atoa = objectives.binary_crossentropy(a_flat, ap_flat)

        # ADDED by ALI
        # KL divergence
        # kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # print(" 100 * K.mean(L_atoa + kl_loss) ")
        # print( 100 * K.mean(L_atoa + kl_loss) )
        # print(" 100 * K.mean(L_atoa + kl_loss) ")

        # return 100 * K.mean(L_atoa + kl_loss)

        return 100 * L_atoa          # ORIGINAL

    vae = Model(i, decoder(encoder(i)), name=name)
    vaeopt = Adam(lr=1e-4)
    vae.compile(optimizer=vaeopt, loss=vae_loss)

    return vae


def discriminator(a_ch, b_ch, nf, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    """Define the discriminator network.

    Parameters:
    - a_ch: the number of channels of the first image;
    - b_ch: the number of channels of the second image;
    - nf: the number of filters of the first layer.
    """
    # i = Input(shape=(a_ch + b_ch, 256, 256))      # ORIGINAL
    # One Channel is added for the input coz we merged OD image with BV images with real image which is 3 channels
    i = Input(shape=(a_ch + a_ch + b_ch, 256, 256))

    # (a_ch + b_ch) x 256 x 256
    conv1 = Convolution(nf)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf*2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(1)(x)
    out = Activation('sigmoid')(conv4)
    # 1 x 16 x 16

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(
            K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def code_discriminator(latent_dim, nf, opt=Adam(), name='code_d'):
    """Define the discriminator that validates the latent code.
    'nfd': 32,  # Number of filters of the first layer of the discriminator
    'latent_dim': 16,  # The dimension of the latent space. Necessary when training the VAE
    """
    z = Input(shape=(latent_dim,))

    h1 = Dense(nf)(z)
    x = LeakyReLU(0.2)(h1)

# ## added by Ali #################
    # h11 = Dense(nf*2)(x)
    # x = LeakyReLU(0.2)(h11)
# #################################
# ## added by Ali  ################
    # h111 = Dense(nf*4)(x)
    # x = LeakyReLU(0.2)(h111)
# ##################################
    h2 = Dense(1)(x)
    out = Activation('sigmoid')(h2)

    d = Model(z, out)

    d.compile(optimizer=opt, loss='binary_crossentropy')
    return d


def code_discriminator_OD(latent_dim, nf, opt=Adam(), name='code_d'):
    """Define the discriminator that validates the latent code.
    'nfd': 32,  # Number of filters of the first layer of the discriminator
    'latent_dim': 16,  # The dimension of the latent space. Necessary when training the VAE
    """
    z = Input(shape=(latent_dim,))

    h1 = Dense(nf)(z)
    x = LeakyReLU(0.2)(h1)

# ## added by Ali #################
    # h11 = Dense(nf*2)(x)
    # x = LeakyReLU(0.2)(h11)
# #################################
# ## added by Ali  ################
#     h111 = Dense(nf*4)(x)
#     x = LeakyReLU(0.2)(h111)
# ##################################
    h2 = Dense(1)(x)
    out = Activation('sigmoid')(h2)

    d = Model(z, out)

    d.compile(optimizer=opt, loss='binary_crossentropy')
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False,
            is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), name='pix2pix'):
    """Define the pix2pix network."""
    a = Input(shape=(a_ch, 256, 256))
    b = Input(shape=(b_ch, 256, 256))
    OD = Input(shape=(a_ch, 256, 256))

    BV_OD = merge([a, OD], mode='concat', concat_axis=1)

    # A -> B'
    bp = atob(BV_OD)
    # bp = atob(a)

    # Discriminator receives the pair of images
    # bp_OD = merge([bp, OD], mode='concat', concat_axis=1)
    d_in = merge([BV_OD, bp], mode='concat', concat_axis=1)
    # # d_in = merge([a, bp], mode='concat', concat_axis=1)
    # d_in = merge([BV_OD, bp], mode='concat', concat_axis=1)

    pix2pix = Model([a, b, OD], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        # Flatten the output of the discriminator. For some reason, applying
        # the loss direcly on the tensors was not working
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        # 'alpha': 100,  # The weight of the reconstruction loss of the atob model
        return L_adv + alpha*L_atob
        # 'beta': 100,  # The weight of the reconstruction loss of the atoa model

    # This network is used to train the generator. Freeze the discriminator
    # part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)

    return pix2pix


def pix2pix2pix(vae, vae_OD, atob, d, code_d, code_d_OD, a_ch, b_ch, alpha=100, beta=100, is_a_binary=False,
                is_OD_binary=False, is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5),
                name='pix2pix2pix'):
    """
    Define the pix2pix2pix network.

    Generator converts A -> A' -> B' and discriminator checks if A'/B' is a
    valid pair.

    The A -> A' transofrmation is performed by a VAE to make sure that
    the bottleneck can be sampled by a gaussian distribution.

    Then, it is possible to sample from z -> A' -> B' to get generated A'/B'
    pairs from z.

    Parameters:
    - vae: a model for a variational auto encoder. Needs to bee composed of
    3 models: a 'vae_encoder' that maps an image to the parameters of the
    distribution (mean, var); a 'vae_sampler' that samples from the previous
    distribution; and a 'vae_decoder' that maps a sample from a distribution
    to an image.
    - atob: a standard auto encoder.
    - d: the discriminator model. Must have the name 'd'.
    - alpha: the weight of the reconstruction term of the atob model in relation
    to the adversarial term. See the pix2pix paper.
    - beta: the weight of the reconstruction term of the atoa model in relation
    to the adversarial term.
    """
    a = Input(shape=(a_ch, 256, 256))
    b = Input(shape=(b_ch, 256, 256))
    OD = Input(shape=(a_ch, 256, 256))

    # A -> A'
    encoder = vae.get_layer('vae_encoder')
    decoder = vae.get_layer('vae_decoder')
    z = encoder(a)
    ap = decoder(z)

    encoder_OD = vae_OD.get_layer('vae_OD_encoder')
    decoder_OD = vae_OD.get_layer('vae_OD_decoder')
    z_OD = encoder_OD(OD)
    ap_OD = decoder_OD(z_OD)

    BV_OD = merge([ap, ap_OD], mode='concat', concat_axis=1)
    # print(BV_OD.shape)   # (?, 2, 256, 256)



    # MY CUSTOM LAYER for sharpeneing VAE output       Added by Ali
    # x = Conv2D(1, 9, 9, init=my_init4,border_mode='same', subsample=(1, 1))(ap)
    # x = Conv2D(1, 9, 9, init=my_init4,border_mode='same', subsample=(1, 1),activation='relu')(x)
    # x = BatchNorm()(x)
    # x = Activation('sigmoid')(x)
    # x = Lambda(lambda z: Binariziation_vessels(x))(x)
    ##################
    # x_OD = merge([x, ap_OD], mode='concat', concat_axis=1)
    # bp = atob(x_OD)
    



    # A' -> B'
    bp = atob(BV_OD)                                      # print(bp.shape)      # (?, 3, 256, 256)


    # Discriminator receives the two generated images
    # ap_ap_OD = merge([ap, ap_OD], mode='concat', concat_axis=1)
    # bp_ap_OD = merge([bp, ap_OD], mode='concat', concat_axis=1)

    # # d_in = merge([ap, bp], mode='concat', concat_axis=1)
    d_in = merge([BV_OD, bp], mode='concat', concat_axis=1)
    # d_in = merge([ap_ap_OD, bp_ap_OD], mode='concat', concat_axis=1)
    # # print(d_in.shape)      # (?, 5, 256, 256)

    # gan = Model([a, b], d(d_in), name=name)
    gan = Model([a, b, OD], d(d_in), name=name)

    def gan_loss(y_true, y_pred):
        # Flatten the output of the discriminator. For some reason, applying
        # the loss direcly on the tensors was not working
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to A loss
        a_flat = K.batch_flatten(a)
        ap_flat = K.batch_flatten(ap)
        if is_a_binary:
            L_atoa = objectives.binary_crossentropy(a_flat, ap_flat)
        else:
            L_atoa = K.mean(K.abs(a_flat - ap_flat))

        # A to A loss for OD
        a_flat_OD = K.batch_flatten(OD)
        ap_flat_OD = K.batch_flatten(ap_OD)
        if is_OD_binary:
            L_atoa_OD = objectives.binary_crossentropy(a_flat_OD, ap_flat_OD)
        else:
            L_atoa_OD = K.mean(K.abs(a_flat_OD - ap_flat_OD))

        # A to B loss
        # if bp:
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        L_code = objectives.binary_crossentropy(np.asarray(
            1).astype('float32').reshape((-1, 1)), code_d(z))
        L_code_OD = objectives.binary_crossentropy(np.asarray(
            1).astype('float32').reshape((-1, 1)), code_d_OD(z_OD))

        # return L_adv + beta*L_atoa + alpha*L_atob + L_code
        return L_adv + 100*L_atoa + 20*L_atoa_OD + alpha*L_atob + L_code + L_code_OD

    # This network is used to train the generator. Freeze the discriminator
    # part
    gan.get_layer('d').trainable = False

    gan.compile(optimizer=opt, loss=gan_loss)

    return gan


def conditional_generator(atoa, atoa_OD, atob, a_ch):
    """Merge the two models into one generator model that goes from a to b."""
    i = Input(shape=(a_ch, 256, 256))
    i_OD = Input(shape=(a_ch, 256, 256))

    BV = atoa(i)
    OD = atoa_OD(i_OD)

    BV_OD = merge([BV, OD], mode='concat', concat_axis=1)

    bp = atob(BV_OD)

    # BV_OD = merge([bp,OD], mode='concat', concat_axis=1)

    g = Model([i, i_OD], bp)

    # g = Model([i,i_OD], atob(BV_OD))
    # # g = Model(i, atob(atoa(i)))

    return g


# # def sharping(image, **kwargs):
# #     kernel = np.array([[-2, -2, -2],
# #                             [-2, 17, -2],
# #                             [-2, -2, -2]])
# #     kernel = tf.expand_dims(kernel, 0)
# #     kernel = tf.expand_dims(kernel, 0)
# #     kernel = tf.cast(kernel, tf.float32)
# #     return tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
# #     return Convolution2D(image, kernel, subsample=(1,1,1,1),nb_col=2, border_mode='same')

#  # return Model.add(Conv2D(32, 3, 3, input_shape=(3, 150, 150)))
# # padding == border_mode
# # subsample == strides

# # from keras import backend as K
# # from keras.engine.topology import Layer
# # import numpy as np

# # from keras import backend as K
# # from keras.engine.topology import Layer

# # ########################### ADDED by ALI
# # class Sharpen(Layer):
# #     def __init__(self, num_outputs, **kwargs):
# #         self.num_outputs = num_outputs
# #         super(Sharpen, self).__init__(**kwargs)

# #     def build(self, input_shape):
# #         self.kernel = np.array([[-2, -2, -2],
# #                                 [-2, 17, -2],
# #                                 [-2, -2, -2]])

# #         self.kernel = tf.expand_dims(self.kernel, 0)
# #         self.kernel = tf.expand_dims(self.kernel, 0)
# #         self.kernel = tf.cast(self.kernel, tf.float32)
# #         print('input_shape')
# #         print(input_shape)
# #         print(' self.kernel ')
# #         print( self.kernel )
# #         super(Sharpen, self).build(input_shape)

# #     def call(self, input_shape,mask=None):
# #         #return Conv2D((input_shape),3,3)# strides=(1, 1), dim_ordering='th', padding='SAME')
# #         # return Convolution2D((input_shape), 3, 3)
# #         return tf.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

# #         #Convolution2D(input_shape, self.kernel, subsample=(1,1,1,1), border_mode='same')
# #     def  get_output_shape_for(self, input_shape):
# #         return(input_shape[0],self.num_outputs)


# # class MyLayer(Layer):
# #     def __init__(self, num_outputs, **kwargs):
# #         self.num_outputs = num_outputs
# #         super(MyLayer, self).__init__(**kwargs)

# #     # custom filter
# #     # def my_filter(self):

# #     #     f = np.array([
# #     #             [[[-2]], [[-2]], [[-2]]],
# #     #             [[[-2]], [[17]], [[-2]]],
# #     #             [[[-2]], [[-2]], [[-2]]]
# #     #         ])
# #     #     assert f.shape == shape
# #     #     return K.variable(f, dtype='float32')

# #     def build(self, input_shape):
# #         ## Create a trainable weight variable for this layer.
# #         self.kernel = self.add_weight(shape=(input_shape[1], self.num_outputs),
# #                                   initializer= np.array([[[[-2]], [[-2]], [[-2]]],
# #                                                          [[[-2]], [[17]], [[-2]]],
# #                                                          [[[-2]], [[-2]], [[-2]]]])
# #                                   # print(self.kernel.shape)
# #         # self.kernel = np.array([[-2, -2, -2],
# #         #                         [-2, 17, -2],
# #         #                         [-2, -2, -2]])
# #         # self.kernel = tf.expand_dims(self.kernel, 0)
# #         # self.kernel = tf.expand_dims(self.kernel, 0)
# #         # self.kernel = tf.cast(self.kernel, tf.float32)

# #         super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

# #     def call(self, input_shape, mask=None):
# #         # return K.dot(x, self.W)
# #         return tf.nn.conv2d(input_shape, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

# #     def get_output_shape_for(self, input_shape):
# #         return (input_shape[0], self.num_outputs)

# ################################
'''##############################################'''
'''  HERE IS THE CORRECT SHARPENING LAYER   '''
'''##############################################'''
# # def my_init(shape, name="SHAPE", **kwargs):
# #     print(shape)
# #     # value = np.random.random(shape)
# #     a = np.array([-2, -2, -2, -2, 17, -2, -2, -2, -2])
# #     # a = np.array([0 , -1,  0, -1,  5, -1,  0, -1,  0])
# #     # a = np.array([-1 , -1 , -1, -1,  0, -1, -1, -1, -1])
# #     value = a.reshape(3, 3)
# #     value= np.expand_dims(value, axis=0)
# #     value= np.expand_dims(value, axis=0)
# #     print(value.shape)
# #     return K.variable(value, name=name)

# # def my_init2(shape, name="SHAPE", **kwargs):
# #     print(shape)
# #     # value = np.random.random(shape)
# #     # a = np.array([-2, -2, -2, -2, 30, -2, -2, -2, -2])
# #     # a = np.array([0 , -1,  0, -1,  4, -1,  0, -1,  0])
# #     a = np.array([-1 , -1 , -1, -1,  0, -1, -1, -1, -1])
# #     value = a.reshape(3, 3)
# #     value= np.expand_dims(value, axis=0)
# #     value= np.expand_dims(value, axis=0)
# #     print(value.shape)
# #     return K.variable(value, name=name)

def my_init3(shape, name="SHAPE", **kwargs):
    print(shape)
    # value = np.random.random(shape)
    # a = np.array([-2, -2, -2, -2, 30, -2, -2, -2, -2])
    # a = np.array([0 , -1,  0, -1,  4, -1,  0, -1,  0])
    a = np.array([-1, -1, -1, -1, 8, -1,-1, -1, 0])
    value = a.reshape(3, 3)
    value = 1/3 * value
    value= np.expand_dims(value, axis=0)
    value= np.expand_dims(value, axis=0)
    print(value.shape)
    return K.variable(value, name=name)


def my_init4(shape, name="SHAPE", **kwargs):
    # print(shape)
    # value = np.random.random(shape)
    # a = np.array([-2, -2, -2, -2, 30, -2, -2, -2, -2])
    # a = np.array([0 , -1,  0, -1,  4, -1,  0, -1,  0])
    # value = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1],
    #                   [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    #                   [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    #                   [-1,-1,-1,9,9,9,9,-1,-1],
    #                   [-1,-1,-1,9,9,9,9,-1,-1],
    #                   [-1,-1,-1,9,9,9,9,-1,-1],
    #                   [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    #                   [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    #                   [-1,-1,-1,-1,-1,-1,-1,-1,-1]])
    value = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                      [0, 2, 3, 5, 5, 5, 2, 2, 0],
                      [3, 3, 5, 3, 0, 3, 5, 3, 3],
                      [2, 5, 3, -12, -23, -12, 3, 5, 2],
                      [2, 5, 0, -23, -40, -23, 0, 5, 2],
                      [2, 5, 3, -12, -23, -12, 3, 5, 2],
                      [3, 3, 5, 3, 0, 2, 5, 3, 3],
                      [0, 2, 3, 5, 5, 5, 3, 2, 0],
                      [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    value = np.expand_dims(value, axis=0)
    value = np.expand_dims(value, axis=0)
    print(value.shape)
    return K.variable(value, name=name)

# # def sobel(shape, name="SHAPE", **kwargs):
# #     print(shape)

# #     value = np.array([[-5 ,-4  ,0 ,4  ,5 ],
# #                       [-8 ,-10 ,0 ,10 ,8 ],
# #                       [-10,-20 ,0 ,20 ,10],
# #                       [-8 ,-10 ,0 ,10 ,8],
# #                       [-5 ,-4  ,0 ,4  ,5]])

# #     value= np.expand_dims(value, axis=0)
# #     value= np.expand_dims(value, axis=0)
# #     print(value.shape)
# #     return K.variable(value, name=name)
# # def sobel2(shape, name="SHAPE", **kwargs):
# #     print(shape)

# #     value = np.array([[ 5 , 8  ,10 ,8   ,5 ],
# #                       [ 4 , 10 ,20 ,10  ,4 ],
# #                       [ 0 ,0   ,0  ,0   ,0 ],
# #                       [-4 ,-10 ,-20,-10 ,-4],
# #                       [-5 ,-8  ,-10 ,-8  ,-5]])

# #     value= np.expand_dims(value, axis=0)
# #     value= np.expand_dims(value, axis=0)
# #     print(value.shape)
# #     return K.variable(value, name=name)

# # def canny3(shape, name="SHAPE", **kwargs):
# #     value = np.array([[ 2 , 4 ,5 ,4 ,2 ],
# #                       [ 4 , 9 ,12,9 ,4 ],
# #                       [ 5 , 12,15,12,5 ],
# #                       [ 4 , 9 ,12,9 ,4],
# #                       [ 2 ,4  ,5 ,4 ,2]])
# #     value = 1/159 * value
# #     value= np.expand_dims(value, axis=0)
# #     value= np.expand_dims(value, axis=0)
# #     return K.variable(value, name=name)

# import cv2
# img = cv2.imread('771.tif',0)
# img.shape
# plt.imshow(img)
# image = np.expand_dims(np.expand_dims(np.array(img),0),0)
# image.shape


def add1(x, x2):
    return tf.math.add(x, x2)


def mult1(x, x2):
    return tf.math.multiply(x, x2)


# # img22 = models.atob.predict(b_gen2)
# # img23=img22[0,0,:,:]
# # img23.shape
# # plt.imshow(img23)
# # plt.show()

#  i = Input(shape=(32,))
#  i_OD = i

#  decoder = vae.get_layer('vae_decoder')
#  ap = decoder(i)

#  ap = Conv2D(1,9,9,init=my_init4, border_mode='same',subsample=(1, 1))(ap)
#  ap = BatchNorm()(ap)
#  ap = Activation('sigmoid')(ap)
#  ap = Lambda(lambda z: Binariziation_vessels(ap))(ap)

#  decoder_OD = vae_OD.get_layer('vae_OD_decoder')
#  ap_OD = decoder_OD(i_OD)

#  ap_apOD = Lambda(lambda z: add1(ap,ap_OD))(ap_OD)
#  ap_apOD = merge([ap_OD, ap], mode='concat', concat_axis=1)

#   bp = models.atob(ap_apOD) #atob(ap)

#  # ap_apOD = merge([ap, ap_apOD], mode='concat', concat_axis=1)   # we added this to get the output channels for ap_apOD equal to 3 not equal to 2, to avoid the problem we face later in the  PLOT THE A->A'->B' RESULTS when we call compose_imgs and convert_to_rgb function which needs image with either one channel or three channels

#    g = Model([i], [ap_apOD, bp])
#  g = Model(i, ap_apOD)


def Binariziation_vessels(x):
    x1 = x < 0.2
    x1 = tf.cast(x1, tf.float32)
    return x1


# i = Input(shape=(32,))
# i_OD = i
# decoder = vae.get_layer('vae_decoder')
# ap = decoder(i)
# x = Conv2D(1, 9, 9, init=my_init4, border_mode='same', subsample=(1, 1))(ap)
# x = BatchNorm()(x)
# x = Activation('sigmoid')(x)
# x = Lambda(lambda z: Binariziation_vessels(x))(x)

# decoder_OD = vae_OD.get_layer('vae_OD_decoder')
# ap_OD = decoder_OD(i_OD)

# ## ap_apOD = Lambda(lambda z: add1(ap,ap_OD))(ap_OD)
# x_OD = merge([x, ap_OD], mode='concat', concat_axis=1)

# bp = models.atob(x_OD)  # atob(ap)

# ## we added this to get the output channels for ap_apOD equal to 3 not equal to 2, to avoid the problem we face later in the  PLOT THE A->A'->B' RESULTS when we call compose_imgs and convert_to_rgb function which needs image with either one channel or three channels
# BV_OD = merge([ap, ap_OD], mode='concat', concat_axis=1)

# g = Model([i], [BV_OD, bp])
# ## g = Model(i, x)
# ################ prediction 
# z_sample = np.random.normal(loc=0.0, scale=1.0, size=(1, 32))   # this executed only once

# vessels,img = g.predict(z_sample, batch_size=1)
# vessels.shape
# img2 = img[0, 0, :, :]
# img2.shape
# plt.imshow(img2)

'''##############################################'''
'''##############################################'''

def generator(vae, vae_OD, atob, latent_dim):
    """Create a model that generates a pair of images."""
    i = Input(shape=(latent_dim,))
   # i_OD = Input(shape=(latent_dim,))           # اذا عمل سامبلنك مختلف الغي احد الادخالات واعتمد على واحد فقط
    i_OD = i
    decoder = vae.get_layer('vae_decoder')
    ap = decoder(i)
    # MY CUSTOM LAYER for sharpeneing VAE output       Added by Ali
    x = Conv2D(1, 9, 9, init=my_init4,border_mode='same', subsample=(1, 1))(ap)
    x = BatchNorm()(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda z: Binariziation_vessels(x))(x)
    ##################
    decoder_OD = vae_OD.get_layer('vae_OD_decoder')
    ap_OD = decoder_OD(i_OD)

    BV_OD = merge([ap, ap_OD], mode='concat', concat_axis=1)
    x_OD = merge([x, ap_OD], mode='concat', concat_axis=1)

    bp = atob(x_OD)  # atob(ap)

    # bp_OD = merge([bp, ap_OD], mode='concat', concat_axis=1)
    # ap_apOD = merge([ap, ap_OD,ap ], mode='concat', concat_axis=1)   # we added this to get the output channels for ap_apOD equal to 3 not equal to 2, to avoid the problem we face later in the  PLOT THE A->A'->B' RESULTS when we call compose_imgs and convert_to_rgb function which needs image with either one channel or three channels

    g = Model([i], [BV_OD, bp])

    # # ap_apOD = Lambda(lambda z: add1(ap,ap_OD))(ap_OD)
    # x_OD = merge([x, ap_OD], mode='concat', concat_axis=1)
    # bp = atob(x_OD) #atob(ap)
    # ap_apOD = merge([ap, ap, ap_OD], mode='concat', concat_axis=1)   # we added this to get the output channels for ap_apOD equal to 3 not equal to 2, to avoid the problem we face later in the  PLOT THE A->A'->B' RESULTS when we call compose_imgs and convert_to_rgb function which needs image with either one channel or three channels
    # g = Model([i], [ap_apOD, bp])
    # #g = Model(i, [ap, bp])
    return g


def generator_from_conditional_generator(g, latent_dim):
    """Create a generator from a conditional generator."""
    g.summary()
    vae = g.layers[2]

    vae_OD = g.layers[3]
    atob = g.layers[5]

    # Original
    # vae = g.layers[1]
    # atob = g.layers[2]
    # return generator(vae, atob, latent_dim)
    return generator(vae, vae_OD, atob, latent_dim)
