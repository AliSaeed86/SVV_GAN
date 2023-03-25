"""Auxiliary methods."""
import os
import json
from errno import EEXIST
# import models as m

import numpy as np
import seaborn as sns
import pickle as pickle     # instead of -->   import cPickle as pickle
import matplotlib.pyplot as plt

sns.set()

DEFAULT_LOG_DIR = 'log'
ATOA_WEIGHTS_FILE = 'atoa_weights.h5'
ATOA_OD_WEIGHTS_FILE = 'atoa_OD_weights.h5'
ATOB_WEIGHTS_FILE = 'atob_weights.h5'
D_WEIGHTS_FILE = 'd_weights.h5'


class MyDict(dict):
    """
    Dictionary that allows to access elements with dot notation.

    ex:
        >> d = MyDict({'key': 'val'})
        >> d.key
        'val'
        >> d.key2 = 'val2'
        >> d
        {'key2': 'val2', 'key': 'val'}
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def convert_to_rgb(img, is_binary=False):
    """Given an image, make sure it has 3 channels."""
    img_shape = img.shape
    if len(img_shape) != 3:
        raise Exception("""Image must have 3 dimensions. """
                        """Given {0}""".format(len(img_shape)))
    img_ch, _, _ = img_shape

    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)
    elif img_ch != 3:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))

    if not is_binary:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.

    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)


def compose_imgs(a, b, is_a_binary=False, is_OD_binary=False, is_b_binary=False):
    """Place a and b side by side to be plotted."""
    ap = convert_to_rgb(a, is_binary=is_a_binary)
    bp = convert_to_rgb(b, is_binary=is_b_binary)
    #OD = convert_to_rgb(OD, is_binary=is_OD_binary)

    if ap.shape != bp.shape:
        raise Exception("""A and B must have the same size. """
                        """{0} != {1}""".format(ap.shape, bp.shape))

    # ap.shape and bp.shape must be the same here
    h, w, ch = ap.shape
    composed = np.zeros((h, 2*w, ch))
    composed[:, :w, :] = ap
    composed[:, w:, :] = bp
    # composed = np.zeros((h, 3*w, ch))
    # composed[:, :w, :] = ap
    # composed[:, w:w*2, :] = OD
    # composed[:, w*2:, :] = bp
    
    return composed


def mkdir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise


def create_expt_dir(params):
    """Create the experiment directory and returns it."""
    expt_dir = get_log_dir(params.log_dir, params.expt_name)

    # Create directories if they do not exist
    mkdir(params.log_dir)
    mkdir(expt_dir)

    # Save the parameters
    # json.dump(params, open(os.path.join(expt_dir, 'params.json'), 'wb'),
    #            indent=4, sort_keys=True)
    json.dump(params, open(os.path.join(expt_dir, 'params.json'), 'w'),
              indent=4, sort_keys=True)

    return expt_dir


def load_params(params):
    """
    Load the parameters of an experiment and return them.

    The params passed as argument will be merged with the new params dict.
    If there is a conflict with a key, the params passed as argument prevails.
    """
    expt_dir = get_log_dir(params.log_dir, params.expt_name)

    # expt_params = json.load(open(os.path.join(expt_dir, 'params.json'), 'rb'))
    expt_params = json.load(open(os.path.join(expt_dir, 'params.json'), 'r'))

    # Update the loaded parameters with the current parameters. This will
    # override conflicting keys as expected.
    expt_params.update(params)

    return expt_params


def get_log_dir(log_dir, expt_name):
    """Compose the log_dir with the experiment name."""
    if log_dir is None:
        raise Exception('log_dir can not be None.')

    if expt_name is not None:
        return os.path.join(log_dir, expt_name)
    return log_dir


def plot_loss(loss, label, filename, log_dir):
    """Plot a loss function and save it in a file."""
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    plt.legend()
    plt.savefig(os.path.join(log_dir, filename))
    plt.clf()


def log(losses, atob, params, it_val, epoch_count ,sampler=None, latent_dim=None, N=4,
        log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Log how the train is going."""
    log_dir = get_log_dir(log_dir, expt_name)
    epoch_count=epoch_count+1
    # Save the losses for further 
    if epoch_count % 100 == 0:
        pickle.dump(losses, open(os.path.join(log_dir, str(epoch_count)+'_losses.pkl'), 'wb'))

    ###########################################################################
    #                             PLOT THE LOSSES                             #
    ###########################################################################
    plot_loss(losses['d'], 'discriminator', 'd_loss.png', log_dir)
    plot_loss(losses['d_val'], 'discriminator validation', 'd_val_loss.png', log_dir)

    plot_loss(losses['p2p'], 'Pix2Pix', 'p2p_loss.png', log_dir)
    plot_loss(losses['p2p_val'], 'Pix2Pix validation', 'p2p_val_loss.png', log_dir)

    if len(losses['p2p2p']) > 0:
        plot_loss(losses['p2p2p'], 'Pix2Pix2Pix', 'p2p2p_loss.png', log_dir)
        plot_loss(losses['p2p2p_val'], 'Pix2Pix2Pix validation', 'p2p2p_val_loss.png', log_dir)

    if len(losses['code_d']) > 0:
        plot_loss(losses['code_d'], 'code discriminator', 'code_d_loss.png', log_dir)
        plot_loss(losses['code_d_val'], 'code discriminator validation', 'code_d_val_loss.png', log_dir)
    
    if len(losses['code_d_OD']) > 0:
        plot_loss(losses['code_d_OD'], 'code discriminator_OD', 'code_d_OD_loss.png', log_dir)
        plot_loss(losses['code_d_val_OD'], 'code discriminator_OD validation', 'code_d_val_OD.png', log_dir)

    ###########################################################################
    #                          PLOT THE A->B RESULTS                          #
    ###########################################################################
    plt.figure(figsize=(10, 6))
    for i in range(N*N):
        a, _, OD = next(it_val)
         
        BV_OD = np.concatenate((a, OD), axis=1)   # shape of BV_OD is --> (1, 2, 256, 256)
        
        bp = atob.predict(BV_OD)     # ORIGINAL
        # bp = atob.predict(BV_OD)
        
        # bp= np.concatenate((bp, bp[:,:1,:,:]), axis=1)   
        BV_OD = np.concatenate((BV_OD, a), axis=1) # add another channel to BV_OD to make them three channel (1, 3, 256, 256) instead of two (1, 2, 256, 256) to avoid the problem when calling image convert_to_rgb function which requires either 1 channel image of 3 channel images, and if recieved only one channel image, it will automatically convert it to three channel so we are doing that before hand calling the function
               
        img = compose_imgs(BV_OD[0], bp[0], is_a_binary=params.is_a_binary, is_OD_binary=params.is_OD_binary,
                           is_b_binary=params.is_b_binary)

        plt.subplot(N, N, i+1)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(os.path.join(log_dir, 'atob_'+str(epoch_count+1)+'.png'))
    plt.clf()

    ###########################################################################
    #                        PLOT THE A->A'->B' RESULTS                       #
    ###########################################################################
    if sampler is not None:
        z_sample = np.random.normal(loc=0.0, scale=1.0, size=(N*N, latent_dim))
        # a_gen, b_gen = sampler.predict(z_sample, batch_size=1)   # Original
        ''' the output of the sampler is from the generator function'''
        a_gen, b_gen = sampler.predict(z_sample, batch_size=1)     #you have to create two samplers
        
        # print('\n a_gen.shape...a_gen.shape....a_gen.shape')
        # print(a_gen.shape)        # shape for a_gen is (?, 3, 256, 256)
        # print('\n b_gen.shape...b_gen.shape....b_gen.shape')
        # print(b_gen.shape)       # shape for b_gen is (?, 3, 256, 256)
        
        a_gen = np.concatenate((a_gen,a_gen[:,:1,:,:]), axis=1)
        # print('\n a_gen.shape...a_gen.shape....a_gen.shape')
        # print(a_gen.shape)        # shape for a_gen is (?, 3, 256, 256)
        # print('\n b_gen.shape...b_gen.shape....b_gen.shape')
        # print(b_gen.shape)       # shape for b_gen is (?, 3, 256, 256)
        plt.figure(figsize=(10, 6))
        for i in range(N*N):
            img = compose_imgs(a_gen[i], b_gen[i], is_a_binary=params.is_a_binary, is_OD_binary=params.is_OD_binary,
                                is_b_binary=params.is_b_binary)

            plt.subplot(N, N, i+1)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig(os.path.join(log_dir, 'results_'+str(epoch_count+1)+'.png'))
        plt.clf()

    # Make sure all the figures are closed.
    plt.close('all')


def save_weights(models,e, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Save the weights of the models into a file."""
    log_dir = get_log_dir(log_dir, expt_name)
    e=e+1
    models.atob.save_weights(os.path.join(log_dir, str(e)+ "_" + ATOB_WEIGHTS_FILE), overwrite=True)
    models.d.save_weights(os.path.join(log_dir, str(e)+ "_" + D_WEIGHTS_FILE), overwrite=True)
    if models.atoa is not None:
        models.atoa.save_weights(os.path.join(log_dir, str(e)+ "_" + ATOA_WEIGHTS_FILE), overwrite=True)
    if models.atoa_OD is not None:
        models.atoa_OD.save_weights(os.path.join(log_dir, str(e)+ "_" + ATOA_OD_WEIGHTS_FILE), overwrite=True)


def load_weights(atoa, atoa_OD, atob, d, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Load the weights into the corresponding models."""
    log_dir = get_log_dir(log_dir, expt_name)

    # atoa.load_weights(os.path.join(log_dir, 'vae_weights.h5'))     ## Added by ALI
    # atoa_OD.load_weights(os.path.join(log_dir, 'vae_OD_weights.h5'))     ## Added by ALI
    atoa.load_weights(os.path.join(log_dir, 'atoa_weights.h5'))     ## Added by ALI
    atoa_OD.load_weights(os.path.join(log_dir, 'atoa_OD_weights.h5'))     ## Added by ALI
    atob.load_weights(os.path.join(log_dir, ATOB_WEIGHTS_FILE))
    d.load_weights(os.path.join(log_dir, D_WEIGHTS_FILE))
    #if atoa:                    ## ORIGINAL
    #    atoa.load_weights(os.path.join(log_dir, ATOA_WEIGHTS_FILE))    ## ORIGINAL


def load_weights_of(m, weights_file, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Load the weights of the model m."""
    log_dir = get_log_dir(log_dir, expt_name)

    m.load_weights(os.path.join(log_dir, weights_file))


def load_losses(log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Load the losses of the given experiment."""
    log_dir = get_log_dir(log_dir, expt_name)
    losses = pickle.load(open(os.path.join(log_dir, 'losses.pkl'), 'rb'))
    return losses


def plot_loss_VAE(loss, label, filename, log_dir):
    """Plot a loss function and save it in a file."""
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    plt.legend()
    plt.savefig(os.path.join(log_dir, filename))
    plt.clf()     
    
def convert_to_rgb_VAE(img, is_binary=False):
    """Given an image, make sure it has 3 channels."""
    img_shape = img.shape
    if len(img_shape) != 3:
        raise Exception("""Image must have 3 dimensions. """
                        """Given {0}""".format(len(img_shape)))
    img_ch, _, _ = img_shape
    
    imgp = img
    # if img_ch == 1:
    #     imgp = np.repeat(img, 3, axis=0)
    # elif img_ch != 3:
    #     raise Exception("""Unsupported number of channels. """
    #                     """Must be 1 or 3, given {0}.""".format(img_ch))

    # if not is_binary:
    #     imgp = imgp * 127.5 + 127.5
    #     imgp /= 255.


    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)


def compose_imgs_VAE(a, b, is_a_binary=False, is_b_binary=False):
    """Place a and b side by side to be plotted."""
    ap = convert_to_rgb_VAE(a, is_binary=is_a_binary)
    bp = convert_to_rgb_VAE(b, is_binary=is_b_binary)

    if ap.shape != bp.shape:
        raise Exception("""A and B must have the same size. """
                        """{0} != {1}""".format(ap.shape, bp.shape))

    # ap.shape and bp.shape must be the same here
    h, w, ch = ap.shape
    composed = np.zeros((h, 2*w, ch))
    composed[:, :w, :] = ap
    composed[:, w:, :] = bp

    return composed    


# def save_weights_VAE(vae,e):
#     """Save the weights of the models into a file."""
#     #log_dir = get_log_dir('log',)
#     vae.save_weights(os.path.join('log/expt_name', str(e+1)+ "_" + 'vae_weights.h5'), overwrite=True)
#     vae.save_weights(os.path.join('log/expt_name', str(e+1)+ "_" + 'vae_OD_weights.h5'), overwrite=True)

