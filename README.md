# 实验环境：

tensorflow 2.15.0

keras 2.15.0

python == 3.6 

pytorch 2.1.0

# 数据集下载：

MNIST数据集 从keras.datasets的数据库中加载，在代码中直接体现，无需下载 
CIFAR-10数据集 从keras.datasets的数据库中加载，在代码中直接体现，无需下载 
LFW数据集 http://vis-www.cs.umass.edu/lfw/



# 运行方式：

python file_name.py 
代码会自动创建对应的子目录并保存生成的图像结果，以及绘制的曲线图像。

**注：各python文件的功能与其命名相同。**



# 实验结果：

运行python文件后，会在代码的同级目录生成如图所示的文件夹。

![image-20240118124128351](C:\Users\风起\AppData\Roaming\Typora\typora-user-images\image-20240118124128351.png)

其中，data为数据集，其中有LFW数据集的数据，

images中存放在MNIST数据集上运行的图像结果，其中有各种对抗网络生成的图像、绘制的收敛曲线和FID与IS的指标值，如图所示。

其中的各子文件夹中存放了对应的实验中所使用的模型及相应的参数所得到的生成图像，DCGAN与WGAN结合后的结果放于DCWGAN中，FID_IS中存放的是所有图像判断生成效果的FID和IS的指标。

![image-20240118124515792](C:\Users\风起\AppData\Roaming\Typora\typora-user-images\image-20240118124515792.png)

Model文件夹中存放各模型运行不同轮次时训练的生成器与判别器，存放于各子文件夹中。

![image-20240118124920715](C:\Users\风起\AppData\Roaming\Typora\typora-user-images\image-20240118124920715.png)

CIFAR10_images 保存利用WGAN在CIFAR-10数据集上进行图像生成的结果以及收敛图像
CIFAR10_Model 保存利用WGAN在CIFAR-10数据集上进行图像生成时训练好的生成器与判别器
data 储存MNIST和LFW数据集
LFW_images 与LFW_Model与CIFAR10对应的文件夹相同，保存在LFW数据集上的结果和模型



