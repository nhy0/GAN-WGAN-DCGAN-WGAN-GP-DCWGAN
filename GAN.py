import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

output_path = "./images/GAN/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

## 对数据做归一化 （-1  1）
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-1 ： channel,high ,witch
    transforms.Normalize(0.5,0.5)
])

train_ds = torchvision.datasets.MNIST('data',train=True,
                                      transform=transform,
                                      download=True)

dataloader = torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)
# 生成器  使用噪声来进行输入
# 输入为长度为100的 噪声 （正态分布随机数） 生成器输出为（1，28，28）的图片
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100,256),nn.ReLU(),
            nn.Linear(256,512),nn.ReLU(),
            nn.Linear(512,28*28),nn.Tanh()
        )

    def forward(self,x):   # x 表示为长度为100 的噪声
        img = self.main(x)
        img = img.view(-1,28,28)
        return img

# 判别器的实现 输入为一张（1，28，28）图片  输出为二分类的概率值，输出使用sigmoid激活 0-1#
# 是用BCELoss损失函数
# 判别器一般使用 LeakyReLu 激活函数

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28,512),nn.LeakyReLU(),
            nn.Linear(512,256),nn.LeakyReLU(),
            nn.Linear(256,1),nn.Sigmoid()
        )

    def forward(self,x): # x 为一张图片
        x = x.view(-1,28*28)
        x = self.main(x)
        return x

epochs = 100
lr = 0.0001

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = Generator().to(device)
discriminator = Discriminator().to(device)
# 优化器（梯度下降）
g_optim = torch.optim.Adam(generator.parameters(),lr=lr)
d_optim = torch.optim.Adam(discriminator.parameters(),lr=lr)

loss_fn = torch.nn.BCELoss()

# 绘图函数

def gen_img_plot(model,test_input,epoch):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    if not os.path.exists('images/GAN'):
        os.mkdir('images/GAN')
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i] + 1 )/2)
        plt.axis('off')
    plt.savefig(os.path.join(output_path, f'epoch_{epoch}.png'))

test_input = torch.randn(16,100,device=device)
# GAN训练

D_loss = list()
G_loss = list()

for epoch in range(1,1+epochs):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step , (img,_) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size,100,device=device)

        # 真实图片上的损失
        d_optim.zero_grad()
        real_output = discriminator(img) # 对判别器输入真实的图片，real_output 对真实图片预测的结果
        # 判别器在真实图像上的损失
        d_real_loss = loss_fn(real_output,torch.ones_like(real_output))
        d_real_loss.backward()

        # 生成图片上的损失
        gen_img = generator(random_noise)
        fake_output = discriminator(gen_img.detach())  # 判别器输入生成的图片，对生成图片的预测
        # 得到判别器在生成图像上的损失
        d_fake_loss = loss_fn(fake_output,torch.zeros_like(fake_output))
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        # 生成器
        g_optim.zero_grad()
        fake_output = discriminator(gen_img)
        # 生成器的损失
        g_loss = loss_fn(fake_output,torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:',epoch)
        gen_img_plot(generator,test_input,epoch)
        print('D_loss:',d_epoch_loss)
        print('G_loss:',g_epoch_loss)

