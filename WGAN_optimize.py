import keras.backend as K
from keras.activations import relu
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, PReLU, LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tensorflow as tf

def initNormal(shape, dtype=None, name=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.2, dtype=dtype)

def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein distance for GAN
    author use:
    g_loss = mean(-fake_logit)
    c_loss = mean(fake_logit - true_logit)
    logit just denote result of discrimiantor without activated
    """
    return K.mean(y_true * y_pred)

def plot_loss(epoch,change):
    plt.figure(figsize=(10, 8))
    plt.plot(Dloss, label='Dsicriminiative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'images/WGAN_{change}/Dloss_epoch_{epoch}_{change}.png')

    plt.figure(figsize=(10, 8))
    plt.plot(Gloss, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'images/WGAN_{change}/Gloss_epoch_{epoch}_{change}.png')

    plt.figure(figsize=(10, 8))
    plt.plot(Dloss, label='Discriminitive loss')
    plt.plot(Gloss, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'images/WGAN_{change}/loss_epoch_{epoch}_{change}.png')

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, change,example=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=(example, randomDim))
    generatedImage = generator.predict(noise)
    generatedImage = generatedImage.reshape(example, 28, 28)

    plt.figure(figsize=figsize)

    for i in range(example):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImage[i], interpolation='nearest', cmap='gray')
        '''drop the x and y axis'''
        plt.axis('off')
    plt.tight_layout()

    if not os.path.exists(f'images/WGAN_{change}'):
        os.mkdir(f'images/WGAN_{change}')
    plt.savefig(f'images/WGAN_{change}/epoch_{epoch}.png')


def saveModels(epoch,change):
    if not os.path.exists(f'Model/WGAN_{change}'):
        os.mkdir(f'Model/WGAN_{change}')
    generator.save(f'Model/WGAN_{change}/generated_epoch_{epoch}.h5' )
    discriminator.save(f'Model/WGAN_{change}/discriminated_epoch_{epoch}.h5' )


def clip_weight(weight, lower, upper):
    weight_clip = []
    for w in weight:
        w = np.clip(w, lower, upper)
        weight_clip.append(w)
    return weight_clip


def train(epochs=1, batchsize=128):
    batchCount = X_train.shape[0] / batchsize
    print('Epochs', epochs)
    print('Bathc_size', batchsize)
    print('Batches per epoch', batchCount)
    # range ande xrange the different is a list and a generator
    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(int(batchCount))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchsize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchsize)]

            # generate fake MNIST images
            generatedImages = generator.predict(noise)

            # Default is axis=0, equal to vstack  is concate up and down
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.ones(2 * batchsize)

            # one-sided label smoothing
            yDis[:batchsize] = -1

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchsize, randomDim])
            yGen = np.ones(batchsize) * -1
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)


            d_weight = discriminator.get_weights()
            d_weight = clip_weight(d_weight,-0.01,0.01)
            discriminator.set_weights(d_weight)


        # Store loss of most recent batch from this epoch
        Dloss.append(dloss)
        Gloss.append(gloss)

        plotGeneratedImages(e,change)
        if e == 1 or e % 5 == 0:
            saveModels(e,change)

    plot_loss(e,change)


lr = 0.0002
np.random.seed(1000)
print("Building Generative Model...")

randomDim = 100

# load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(60000, 784)


# Function for initializing network weights


adam = Adam(lr=lr, beta_1=0.5)

# Build Generative Model
g_input = Input(shape=(randomDim,))
# here the initNormal can equal to the 'normal',init can receive funciton
H = Dense(256, kernel_initializer=initNormal)(g_input)
H = LeakyReLU(0.2)(H)

H = Dense(512)(H)
H = LeakyReLU(0.2)(H)

H = Dense(1024)(H)
H = LeakyReLU(0.2)(H)

# Because train data have normalized to [-1,1] ,tanh can be fit
g_output = Dense(784, activation='tanh')(H)

generator = Model(g_input, g_output)
# generator.compile(loss=wasserstein_loss,optimizer='RMSprop')


# Build Discriminative Model
d_input = Input(shape=(784,))

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
discriminator.compile(loss=wasserstein_loss, optimizer='RMSprop')

# Combine the two networks
discriminator.trainable = False
gan_input = Input((randomDim,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = Model(gan_input, gan_output)
gan.compile(loss=wasserstein_loss, optimizer='RMSprop')

Dloss = []
Gloss = []


# Plot the loss from each epoch

change = "clip"
if __name__ == '__main__':
    train(100, 128)
'''
lr_s = [0.0001,0.0003]

for lr in lr_s:
    np.random.seed(1000)
    print("Building Generative Model...")

    randomDim = 100

    # load mnist data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape(60000, 784)


    # Function for initializing network weights


    adam = Adam(lr=lr, beta_1=0.5)

    # Build Generative Model
    g_input = Input(shape=(randomDim,))
    # here the initNormal can equal to the 'normal',init can receive funciton
    H = Dense(256, kernel_initializer=initNormal)(g_input)
    H = LeakyReLU(0.2)(H)

    H = Dense(512)(H)
    H = LeakyReLU(0.2)(H)

    H = Dense(1024)(H)
    H = LeakyReLU(0.2)(H)

    # Because train data have normalized to [-1,1] ,tanh can be fit
    g_output = Dense(784, activation='tanh')(H)

    generator = Model(g_input, g_output)
    # generator.compile(loss=wasserstein_loss,optimizer='RMSprop')


    # Build Discriminative Model
    d_input = Input(shape=(784,))

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
    discriminator.compile(loss=wasserstein_loss, optimizer='RMSprop')

    # Combine the two networks
    discriminator.trainable = False
    gan_input = Input((randomDim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = Model(gan_input, gan_output)
    gan.compile(loss=wasserstein_loss, optimizer='RMSprop')

    Dloss = []
    Gloss = []


    # Plot the loss from each epoch

    change = lr
    if __name__ == '__main__':
        train(100, 128)

batchsize_s = [64,256]
for batchsize in batchsize_s:
    np.random.seed(1000)
    print("Building Generative Model...")

    randomDim = 100

    # load mnist data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape(60000, 784)

    # Function for initializing network weights

    adam = Adam(lr=0.0002, beta_1=0.5)

    # Build Generative Model
    g_input = Input(shape=(randomDim,))
    #here the initNormal can equal to the 'normal',init can receive funciton
    H = Dense(256, kernel_initializer=initNormal)(g_input)
    H = LeakyReLU(0.2)(H)

    H = Dense(512)(H)
    H = LeakyReLU(0.2)(H)

    H = Dense(1024)(H)
    H = LeakyReLU(0.2)(H)

    # Because train data have normalized to [-1,1] ,tanh can be fit
    g_output = Dense(784, activation='tanh')(H)

    generator = Model(g_input, g_output)
    # generator.compile(loss=wasserstein_loss,optimizer='RMSprop')

    # Build Discriminative Model
    d_input = Input(shape=(784,))

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
    discriminator.compile(loss=wasserstein_loss, optimizer='RMSprop')

    # Combine the two networks
    discriminator.trainable = False
    gan_input = Input((randomDim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = Model(gan_input, gan_output)
    gan.compile(loss=wasserstein_loss, optimizer='RMSprop')

    Dloss = []
    Gloss = []

    # Plot the loss from each epoch
    change  = batchsize
    if __name__ == '__main__':
        train(100, batchsize)'''