## one pixel attack on cifar10 by pytorch

目前攻击默认采用的是三像素攻击，有些模型采用了五像素。默认攻击迭代次数是100次。

### 目前已有模型：

- VGG11

- VGG13

- VGG16

- VGG19

- LeNet

- MobileNet

- ResNet18

- ResNet34

- ResNet101

数据集：cifar10

[模型参考](https://github.com/kuangliu/pytorch-cifar)

### 环境

torch==1.1.0

torchvision==0.3.0

pillow<7.0.0

tqdm

### 运行须知

- 运行前请先配置`config.py`文件中内容（只需配置此内容即可），config中加载的模型名必须与models/__init__.py中的名字一致

- 运行`main.py`即可，此方法需要配置config文件。或使用notebook运行`main.ipynb`，此方法在ipynb里面配置即可

- 若`./ckps`文件夹下无预训练模型，则需要先训练模型

- 先训练模型(`main.py`中的train()函数，若有预训练模型，可以跳过)，模型训练完并保存在ckps文件夹后，修改config中model_path为预训练路径

- 实施攻击(`main.py`中attack_model()函数)，之后会保存所有内容在`log.txt`文件夹下，保存的内容有
       
       - 模型名
       - 准确率
       - 攻击成功率
       - 保存日志时间

### 结果

| 模型名 | 准确率 | 攻击成功率 |
| :------: | :------: | :------: |
| Lenet | 0.57 | 0.9966 |
| MobileNet | 0.834 | 0.6309 |
| ResNet18 | 0.827 | 0.6591 |
| MobileNet | 0.834 | 0.6309 |
| ResNet34 | 0.811 | 0.6023 |
| ResNet101 | 0.852 | 0.6146 |
| ResNet18 | 0.827 | 0.6591 |
| VGG11 | 0.823 | 0.7803 |
| VGG13 | 0.842 | 0.8034 |
| VGG16 | 0.852 | 0.6290 |
| VGG19(500次迭代) | 0.834 | 0.0 |
| VGG19(5像素) | 0.859 | 0.0 |