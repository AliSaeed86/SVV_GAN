# Synthesizing Retinal Fundus Images by Sharpening and Varying Vessels Abundance Using an End-To-End VAEs-GAN Pipeline-Based Intermediate Layer

To synthesize eye fundus images directly from data, we utilize an image-to-image translation technique that employs multiple adversarial learning architectures like Generative Adversarial Networks (GAN) and Variational Autoencoder (VAE). Our method involves pairing the reconstructed vessels tree and optic disc, feed them to an intermediate sharpening and varying vessels (SVV) layer, and then apply image-to-image translation to generate the fake retinal image. By using reconstructed pairs and the intermediate layer, we train a model to learn how to map the combined masks into a new retinal image.

![generated images with different SVV value](https://user-images.githubusercontent.com/68149304/227696582-22154243-e2c9-4b0b-9c71-247d81d47606.png)

# Dataset
The dataset used in this work is the Messidor-DB1 dataset, which is publicly available on the following link: https://www.adcis.net/en/third-party/messidor/ 
The DRIVE dataset is also use to test the efficiency of the generated images through utilizing downstream segmentation task and test the trained model on DRIVE testset. The DRIVE dataset can be accessed on this link: DRIVE Dataset | Papers With Code

# Prerequest 
Python 3.8.10 
Keras library and TensorFlow 2.6.0. 

#Train the model
To train the model run:

    'python train.py [--help]'
    
By default the model will be saved to a folder named 'log'.
