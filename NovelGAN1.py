# Novel GAN

import numpy as np
from numpy import random
from typing import Tuple
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Conv3DTranspose, Dropout, Input, Layer, InputSpec
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU, BatchNormalization, Reshape, LayerNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import InputSpec
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import Model
import math
import numpy as np
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure, Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet
import os

def load_real_samples(data_path: str) -> np.ndarray:
    """ Loads in the tensor of real samples.

      Args:
        data_path: Path to the dataset.
      Returns:
        Tensor of real samples. Shape = (num_samples, 64, 64, 4).
    """
    data_tensor = np.load(data_path)
    return np.reshape(data_tensor, (data_tensor.shape[0], 64, 64, 4))

def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in
        model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes

def conv_norm(x: tf.Tensor, units: int,
              filter: Tuple[int, int], stride: Tuple[int, int],
              discriminator: bool = True
              ) -> tf.Tensor:
  """Applies either a convolution or transposed convolution, normalization, and
     an activation function.

  Args:
    x: The previous keras tensor passed into the
    convolutional/normalization layer
    units: The number of channels in the convolutional layer
    filter: The filter size of the convolutional layer
    stride: The stride size of the convolutional layer
    discriminator: Whether conv_norm is present in the discriminator. If true,
    a convolution (as opposed to a transposed convolution) will be applied.

  Returns:
    The keras tensor after the convolution, normalization, and activation
    function.
  """
  if discriminator:
    conv = Conv3D(units, filter, strides = stride, padding = 'valid')
  else:
    conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
  x = tfa.layers.SpectralNormalization(conv)(x)
  x = LayerNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  return x

def dense_norm(x: tf.Tensor, units: int
               ) -> tf.Tensor:
  """Applies a dense layer, normalization, and an activation function.

  Args:
    x: The previous keras tensor passed into the dense/normalization layer
    units: The number of units in the dense layer


  Returns:
    The keras tensor after the dense layer, normalization, and activation
    function.
  """
  x = tfa.layers.SpectralNormalization(Dense(units))(x)
  x = LayerNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  return x

class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=512, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        return self

    def __next__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size, shape)

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)

class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x,y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1],0,1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights,-1)
        rw = x*weights + y*(1-weights)
        return rw

class Nontrainable(object):

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.trainable_status = self.model.trainable
        self.model.trainable = False
        return self.model

    def __exit__(self, type, value, traceback):
        self.model.trainable = self.trainable_status

class GradientPenalty(Layer):
    def call(self, inputs):
        real_image, generated_image, disc = inputs
        avg_image = RandomWeightedAverage()(
        [real_image, generated_image]
        )
        with tf.GradientTape() as tape:
          tape.watch(avg_image)
          disc_avg = disc(avg_image)

        grad = tape.gradient(disc_avg,[avg_image])[0]
        GP = K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1
        return GP

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

def generate_real_samples(dataset: np.ndarray, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly selects a number of samples from the real dataset.

    Args:
      dataset: The dataset of real samples.
      n_samples: The number of samples requested.

    Returns:
      The randomly selected samples from the dataset and a vector of ones
      to indicate that the samples are "real."
    """
    ix = random.randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]
    y = np.ones((n_samples,1))
    return X,y


def generate_latent_points(latent_dim: int, n_samples:int) -> np.ndarray:
    """Generates a random array to be used by the generator.

    Args:
      latent_dim: The dimension of the latent space.
      n_samples: The number of samples requested.

    Returns:
      An array of random latent variables. Shape = (n_samples, latent_dim).
    """
    x_input = random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input

def generate_fake_samples(generator: tf.Tensor,
                          latent_dim: int, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Generates fake samples from the generator.

    Args:
      generator: The generator model.
      latent_dim: The dimension of latent space.
      n_samples: The number of samples requested.

    Returns:
      The generated samples and a vector of zeros to indicate that the samples
      are "fake."
    """
    x_input = generate_latent_points(latent_dim,n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y

ef define_critic(in_shape = (64, 64, 4, 1)
) -> tf.Tensor:
    """Constructs the critic in the WGAN. Uses a combination of dense
    and convolutional layers.

    Args:
      in_shape: The shape of the input keras tensor.

    Returns:
      The critic model.
    """
    tens_in = Input(shape=in_shape, name="input")

    x = conv_norm(tens_in, 16, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 16, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (5,5,2), (5,5,1), True)
    x = conv_norm(x, 128, (2,2,2), (2,2,2), True)

    x = Flatten()(x)
    x = Dropout(0.25)(x)

    disc_out = tfa.layers.SpectralNormalization(Dense(1, activation = "linear"))(x)
    model = Model(inputs=tens_in, outputs=disc_out)

    return model

def define_generator(latent_dim):
    n_nodes = 16 * 16 * 4

    noise_in = Input(shape=(latent_dim,), name="noise_input")

    x = dense_norm(noise_in, n_nodes)
    x = Reshape((16,16, 4, 1))(x)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (2,2,2), (2,2,2), False)

    outMat = tfa.layers.SpectralNormalization(Conv3D(1,(1,1,10), activation = 'sigmoid', strides = (1,1,10), padding = 'valid'))(x)

    model = Model(inputs=noise_in, outputs=outMat)
    return model

class WGANGP(object):
    def __init__(self, gen, disc, model_path, api_key, output_path, lr_gen=0.0001, lr_disc=0.0001):
        # Initialize generator, discriminator, and optimizers
        self.gen = gen
        self.disc = disc
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.model_path = model_path
        self.api_key = api_key
        self.output_path = output_path
        self.m3gnet_model = M3GNet.from_dir(model_path)  # Load M3GNet model from specified path
        self.build()

    def build(self):
        tens_shape = (64, 64, 4, 1)
        
        self.opt_disc = Adam(self.lr_disc, beta_1=0.0, beta_2=0.9)
        self.opt_gen = Adam(self.lr_gen, beta_1=0.0, beta_2=0.9)

        # Discriminator and generator trainers setup omitted for brevity

    def predict_ehull(self, generated_samples):
        ehull_list = []

        for sample in generated_samples:
            try:
                crystal = Structure.from_sites(sample)  # Convert to pymatgen Structure object
                e_form_predict = self.m3gnet_model.predict_structure(crystal)
                elements = ''.join([i for i in crystal.formula if not i.isdigit()]).split(" ")

                mpr = MPRester(self.api_key)
                all_compounds = mpr.summary.search(elements=elements)

                # Prepare phase diagram entries
                pde_list = [ComputedEntry(str(comp.composition.reduced_composition).replace(" ", ""),
                                          comp.formation_energy_per_atom)
                            for comp in all_compounds
                            if hasattr(comp, 'composition') and hasattr(comp, 'formation_energy_per_atom')]

                if not pde_list:
                    ehull_list.append(float('inf'))
                    continue

                diagram = PhaseDiagram(pde_list)
                _, pmg_ehull = diagram.get_decomp_and_e_above_hull(
                    ComputedEntry(Composition(crystal.formula.replace(" ", "")), e_form_predict[0][0].numpy())
                )
                ehull_list.append(pmg_ehull)
            except Exception as e:
                ehull_list.append(float('inf'))  # Use inf as a high penalty for failures
        
        return np.mean(ehull_list)  # Average energy above hull for the batch

    def combined_loss(self, y_true, y_pred):
        # Compute GAN loss
        gan_loss = wasserstein_loss(y_true, y_pred)
        
        # Compute energy above hull loss
        generated_samples = K.get_value(y_pred)  # Assumes y_pred holds the generator output
        ehull_loss = self.predict_ehull(generated_samples)
        
        # Combine the losses, adjust the weight for ehull loss as needed
        return gan_loss + 0.1 * ehull_loss
        def fit_generator(self, noise_gen, dataset, latent_dim, n_epochs=10, n_batch=256, n_critic=5, model_name=None):
        for epoch in range(n_epochs):
            for _ in range(n_critic):
                # Train the discriminator (critic)
                real_samples = next(dataset)
                noise = next(noise_gen)
                d_loss = self.disc_trainer.train_on_batch([real_samples, noise], [real_labels, fake_labels, None])

            # Train the generator
            noise = next(noise_gen)
            g_loss = self.gen_trainer.train_on_batch(noise, fake_target)
            
            # Monitor training progress
            print(f"Epoch {epoch+1}, g_loss: {g_loss}, d_loss: {d_loss}")

import os
batch_size = 1

data_path = './'
os.chdir(data_path)

file_path = 'test.npy'
data = np.load(file_path)

n_epochs = 10
n_critic = 5
model_path = './model'

def main():
 #   args = parser.parse_args()
    noise_dim = 128
    critic = define_critic()
    generator = define_generator(noise_dim)
    gan_model = WGANGP(generator, critic)
    noise_gen = NoiseGenerator([noise_dim,], batch_size = batch_size)
    dataset = load_real_samples(file_path)
    gan_model.fit_generator(noise_gen, dataset, noise_dim, n_epochs, batch_size,n_critic, model_path)

if __name__ == "__main__":
    main()
