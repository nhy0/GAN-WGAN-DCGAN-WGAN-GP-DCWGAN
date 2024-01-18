import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization, Reshape, Flatten, Lambda
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tensorflow as tf
np.random.seed(1000)
dLosses = []
gLosses = []
# Define the Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Define the gradient penalty loss
def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

# Build the generator model
def build_generator(random_dim):
    input_noise = Input(shape=(random_dim,))
    x = Dense(128)(input_noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    output = Dense(784, activation='tanh')(x)

    generator = Model(input_noise, output)
    return generator

# Build the critic (discriminator) model
def build_critic(random_dim):
    input_data = Input(shape=(784,))
    x = Dense(1024)(input_data)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)

    critic = Model(input_data, output)
    return critic

# Combine the generator and critic into a GAN model
def build_gan(generator, critic):
    critic.trainable = False
    noise = Input(shape=(random_dim,))
    generated_data = generator(noise)
    validity = critic(generated_data)
    gan = Model(noise, validity)
    return gan

# Generate random weighted average between real and generated samples
def random_weighted_average(inputs):
    alpha = K.random_uniform((batch_size, 1))
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

# Load and preprocess the MNIST dataset
def load_mnist():
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 784))
    return X_train

# Clip the weights of the critic
def clip_critic_weights(critic, clip_value):
    for l in critic.layers:
        weights = l.get_weights()
        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
        l.set_weights(weights)

# Plot generated images
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=(examples, random_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    save_dir = 'images/WGAN-GP'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'epoch_%d.png' % epoch))
    plt.close()

def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Dsicriminiative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/WGAN-GP_Dloss_epoch_%d.png' % epoch)

    plt.figure(figsize=(10, 8))
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/WGAN-GP_Gloss_epoch_%d.png' % epoch)

    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/WGAN-GP_loss_epoch_%d.png' % epoch)

# Train the WGAN-GP model
def train_wgan_gp(epochs=100, batch_size=64, random_dim=100, clip_value=0.01, n_critic=5):
    X_train = load_mnist()

    # Build and compile the critic
    critic = build_critic(random_dim)
    critic.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.9), metrics=['accuracy'])

    # Build the generator
    generator = build_generator(random_dim)

    # Build and compile the WGAN-GP model
    critic.trainable = False
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = critic(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.9), metrics=['accuracy'])

    # Training loop
    for epoch in range(1, epochs + 1):
        for _ in tqdm(range(X_train.shape[0] // batch_size)):
            for _ in range(n_critic):
                noise = np.random.normal(0, 1, size=(batch_size, random_dim))
                generated_data = generator.predict(noise)

                real_data = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
                averaged_samples = Lambda(random_weighted_average, output_shape=(784,))([real_data, generated_data])

                valid = -np.ones((batch_size, 1))
                fake = np.ones((batch_size, 1))
                dummy_valid = -np.ones((batch_size, 1))

                d_loss_real = critic.train_on_batch(real_data, valid)
                d_loss_fake = critic.train_on_batch(generated_data, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                clip_critic_weights(critic, clip_value)

            noise = np.random.normal(0, 1, size=(batch_size, random_dim))
            valid = -np.ones((batch_size, 1))

            g_loss = gan.train_on_batch(noise, valid)

        print(f'Epoch {epoch}, [D loss: {d_loss[0]}] [G loss: {g_loss[0]}]')
        dLosses.append(d_loss[0])
        gLosses.append(g_loss[0])
        #if epoch % 10 == 0:
        plot_generated_images(epoch, generator)

    plot_loss(epoch)

if __name__ == '__main__':
    batch_size = 128
    random_dim = 100
    train_wgan_gp(epochs=50, batch_size=batch_size, random_dim=random_dim)
