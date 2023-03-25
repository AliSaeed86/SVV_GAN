# Synthesizing Retinal Fundus Images by Sharpening and Varying Vessels Abundance Using an End-To-End VAEs-GAN Pipeline-Based Intermediate Layer

To synthesize eye fundus images directly from data, we utilize an image-to-image translation technique that employs multiple adversarial learning architectures like Generative Adversarial Netwroks (GAN) and Variational AutoEnconder (VAE). Our method involves pairing the reconstructed vessels tree and optic disc, feed the to and intermediate sharpening and varying vessels (SVV) layer, and then apply image-to-image translation to generate the fake retinal image. By using these paires data, we train a model to learn how to map the combined vessel tree and optic disc masks into a new retinal image.

![generated images with different SVV value](https://user-images.githubusercontent.com/68149304/227696582-22154243-e2c9-4b0b-9c71-247d81d47606.png)
