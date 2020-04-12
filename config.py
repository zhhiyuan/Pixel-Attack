import torch as t
class Config():
    #一般只修改以下三项
    model_path = './ckps/Lenet5_04_12_20_01.pth'  # 预训练模型，None表示重新训练
    model = 'LeNet5'#加载的模型，模型名必须与models/__init__.py中的名字一致
    use_gpu=True    #是否使用gpu

    attack_num = 1000  #选择攻击的样本数量

    train_epoch = 10 #将数据集训练多少次
    batch_size = 128 #每次喂入多少数据

    print_freq = 500    #每训练多少批次就打印一次

    def _parese(self):
        self.device = t.device('cuda') if self.use_gpu else t.device('cpu')
        print('Caculate on {}'.format(self.device))
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = Config()