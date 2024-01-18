import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, Reshape, Flatten
from keras.optimizers import RMSprop
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tensorflow as tf

def initNormal(shape, dtype=None, name=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.2, dtype=dtype)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(Dloss, label='Discriminative loss')
    plt.plot(Gloss, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('CIFAR10_images/WGAN_loss_epoch_%d.png' % epoch)

# Create a wall of generated CIFAR-10 images
def plotGeneratedImages(epoch, example=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=(example, randomDim))
    generatedImage = generator.predict(noise)
    generatedImage = generatedImage.reshape(example, 32, 32, 3)
    generatedImage = np.clip(generatedImage, 0, 1)
    plt.figure(figsize=figsize)

    for i in range(example):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImage[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('CIFAR10_images'):
        os.mkdir('CIFAR10_images')
    if not os.path.exists('CIFAR10_images/WGAN'):
        os.mkdir('CIFAR10_images/WGAN')
    plt.savefig('CIFAR10_images/WGAN/epoch_%d.png' % epoch)

def saveModels(epoch):
    if not os.path.exists('CIFAR10_Model'):
        os.mkdir('CIFAR10_Model')
    if not os.path.exists('CIFAR10_Model/WGAN'):
        os.mkdir('CIFAR10_Model/WGAN')
    generator.save('CIFAR10_Model/WGAN/generated_epoch_%d.h5' % epoch)
    discriminator.save('CIFAR10_Model/WGAN/discriminated_epoch_%d.h5' % epoch)

def train(epochs=1, batchsize=128):
    batchCount = X_train.shape[0] // batchsize  # Use integer division
    print('Epochs', epochs)
    print('Batch_size', batchsize)
    print('Batches per epoch', batchCount)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(int(batchCount))):
            noise = np.random.normal(0, 1, size=[batchsize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchsize)]

            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            yDis = np.ones(2 * batchsize)
            yDis[:batchsize] = -1

            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            noise = np.random.normal(0, 1, size=[batchsize, randomDim])
            yGen = np.ones(batchsize) * -1
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        Dloss.append(dloss)
        Gloss.append(gloss)
        plotGeneratedImages(e)
        if e == 1 or e % 5 == 0:
            saveModels(e)

    plot_loss(e)

np.random.seed(1000)
print("Building Generative Model...")

randomDim = 100

# Load CIFAR-10 data
(X_train, _), (_, _) = cifar10.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], -1)

adam = RMSprop(lr=0.0001)

g_input = Input(shape=(randomDim,))
H = Dense(256, kernel_initializer=initNormal)(g_input)
H = LeakyReLU(0.2)(H)

H = Dense(512)(H)
H = LeakyReLU(0.2)(H)

H = Dense(1024)(H)
H = LeakyReLU(0.2)(H)

g_output = Dense(X_train.shape[1], activation='tanh')(H)
generator = Model(g_input, g_output)

d_input = Input(shape=(X_train.shape[1],))
D = Dense(1024, kernel_initializer=initNormal)(d_input)
D = LeakyReLU(0.2)(D)
D = Dropout(0.3)(D)

D = Dense(512)(D)
D = LeakyReLU(0.2)(D)
D = Dropout(0.3)(D)

D = Dense(256)(D)
D = LeakyReLU(0.2)(D)
D = Dropout(0.3)(D)

d_output = Dense(1, activation='linear')(D)
discriminator = Model(d_input, d_output)

discriminator.trainable = False
gan_input = Input((randomDim,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = Model(gan_input, gan_output)

gan.compile(loss=wasserstein_loss, optimizer=adam)

Dloss = []
Gloss = []

if __name__ == '__main__':
    generator.compile(loss=wasserstein_loss, optimizer=adam)
    discriminator.compile(loss=wasserstein_loss, optimizer=adam)
    train(100, 128)
