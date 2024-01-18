import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import sqrtm
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


# 加载 CIFAR-10 数据集
(X_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# 将MNIST图像调整为InceptionV3期望的尺寸
def preprocess_mnist_images(images):
    return images / 255.0


X_train_inception = preprocess_mnist_images(X_train)
INCEPTION_SHAPE = (299, 299, 3)  # 根据您的实际需要调整

inception = InceptionV3(include_top=False, pooling='avg', input_shape=INCEPTION_SHAPE)
inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def calculate_is(model, images):
    logits = model.predict(images)
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    is_score = np.exp(np.mean(entropy))

    return is_score

# 从本地加载图像
def load_images_from_epoch(epoch):
    images = []
    image_path = f"CIFAR10_images/WGAN/epoch_{epoch}.png"
    if os.path.exists(image_path):
        img = load_img(image_path, color_mode='rgb', target_size=(299, 299))
        img_array = img_to_array(img)
        # No need to squeeze since we are loading in 'rgb' mode
        images.append(img_array)
    return np.array(images) / 255.0

# 计算FID和IS
def calculate_metrics(model, images1, images2):
    # 将图像调整为 Inception 模型的输入尺寸
    images1_resized = tf.image.resize(images1, (299, 299))
    images2_resized = tf.image.resize(images2, (299, 299))

    # 计算FID
    fid_score = calculate_fid(model, images1_resized, images2_resized)
    # 计算IS
    is_score = calculate_is(model, images2_resized)

    return fid_score, is_score


# 绘制折线图
def plot_metrics(epoch_list, fid_scores, is_scores):
    if not os.path.exists('images/FID_IS'):
        os.mkdir('images/FID_IS')

    plt.figure(figsize=(10, 8))
    plt.plot(epoch_list, fid_scores, label='FID Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('images/FID_IS/CIFAR_FID.png')

    plt.figure(figsize=(10, 8))
    plt.plot(epoch_list, is_scores, label='IS Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('images/FID_IS/CIFAR_IS.png')

    plt.figure(figsize=(10, 8))
    plt.plot(epoch_list, is_scores, label='IS Score')
    plt.plot(epoch_list, fid_scores, label='FID Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('images/FID_IS/CIFAR_FID_IS.png')
'''
method = ['WGAN','WGAN_0.0001']  在保存图像的地方多一个参数
for a in method:'''
num_epochs = 100 # 总共的epoch数量

epoch_list = []
fid_scores = []
is_scores = []

for epoch in range(1, num_epochs + 1):
    loaded_images = load_images_from_epoch(epoch)

    # 将MNIST图像调整为InceptionV3期望的尺寸
    loaded_images_inception = preprocess_mnist_images(loaded_images)
    # 计算FID和IS
    fid, is_score = calculate_metrics(inception, X_train_inception[:100], loaded_images_inception)
    # 保存结果
    epoch_list.append(epoch)
    fid_scores.append(fid)
    is_scores.append(is_score)
    print("fid =",fid)
    print("IS =", is_score)
# 绘制折线图
plot_metrics(epoch_list, fid_scores, is_scores)