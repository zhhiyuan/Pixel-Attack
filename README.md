## one pixel attack on cifar10 by pytorch


### 目前已有模型：

- vgg，包含vgg11,vgg13,vgg16,vgg19

- Lenet5

- mobilenet

数据集：cifar10

### 环境

torch==1.1.0

torchvision==0.3.0

pillow<7.0.0

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
