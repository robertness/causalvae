
## Deep Causal Varitional Autoencoder

To train a supervised variational autoencoder using Deepmind's [dSprites](https://github.com/deepmind/dsprites-dataset) dataset.

dSprites is a dataset of sprites, which are 2D shapes procedurally generated from 5 ground truth independent "factors." These factors are color, shape, scale, rotation, x and y positions of a sprite.

All possible combinations of these variables are present exactly once, generating N = 737280 total images.

Factors and their values:

* Shape: 3 values {square, ellipse, heart}
* Scale: 6 values linearly spaced in (0.5, 1)
* Orientation: 40 values in (0, 2$\pi$)
* Position X: 32 values in (0, 1)
* Position Y: 32 values in (0, 1)

There is a sixth factor for color, but it is white for every image in this dataset.

The purpose of this dataset was to evaluate the ability of disentanglement methods.  In these methods, you treat these factors as latent and then try to "disentangle" them in the latent representation.

However, in this project, these factors are not treated as latent, but are included as labels in the model training.  Further, a causal story is invented that relates these factors and the images in a DAG

![vae_dag](dag.png) 

$shape =  f_{shape}(N_{shape})$

$orientation = f_{orientation}({shape, N_{orientation}})$

$scale = f_{scale}(shape, N_{{scale}})$

$X = f_{{X}}({orientation}, N_{{X}})$

${Y} = f_{{Y}}({scale}, N_{{Y}})$

${image} = g({orientation}, {scale}, {X}, {Y},{image}, N_{{image}})$
