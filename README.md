# 实验环境：

tensorflow 2.15.0

keras 2.15.0

python == 3.6 

pytorch 2.1.0

# 数据集下载：

MNIST数据集 从keras.datasets的数据库中加载，无需下载 
CIFAR-10数据集 从keras.datasets的数据库中加载，无需下载 
LFW数据集 http://vis-www.cs.umass.edu/lfw/

# 运行方式：

python file_name.py 
创建对应的子目录及保存结果

# 实验结果：

CIFAR10_images 保存利用WGAN在CIFAR-10数据集上进行图像生成的结果以及收敛图像
CIFAR10_Model 保存利用WGAN在CIFAR-10数据集上进行图像生成时训练好的生成器与判别器
data 储存MNIST和LFW数据集
LFW_images 与LFW_Model与CIFAR10对应的文件夹相同，保存在LFW数据集上的结果和模型
Model 保存在MNIST数据集上的各种优化和改进的生成器与判别器模型

images中：
各子目录分别保存不同的GAN模型、参数、改进的生成图像及收敛图像
FID_IS中保存使用不同的方法得到的生成图像的FID与IS的值及其变化
以及部分模型的损失图像

### 各python文件的功能与其命名相同。