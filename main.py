import os
import torchvision as tv
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import torch as t
import numpy as np
from time import strftime

from config import opt
import models
from attack.PixelAttack import attack_all
from attack.PGDAttack import LinfPGDAttack

DOWNLOAD_CIFAR10=False   #是否需要下载数据集


def train():
    '''
    训练神经网络
    :return:
    '''
    #1.加载配置
    opt._parese()
    global DOWNLOAD_CIFAR10

    #1a.加载模型
    model = getattr(models,opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device)

    #2.定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)


    #3.加载数据
    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=True,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )



    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8
                              )

    #训练模型
    for epoch in range(opt.train_epoch):
        for ii,(data,label) in tqdm(enumerate(train_loader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            if (ii+1)%opt.print_freq ==0:
                print('loss:%.2f'%loss.cpu().data.numpy())
    model.save()

@t.no_grad()
def test_acc():
    '''
    测试模型准确率
    :return:
    '''
    # 1.加载配置
    opt._parese()
    global DOWNLOAD_CIFAR10

    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device)

    #2.加载数据
    transform = tv.transforms.Compose(        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=False,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )

    test_loader = DataLoader(test_data,batch_size=1000,shuffle=True,num_workers=8)

    dataiter = iter(test_loader)
    test_x, test_y = next(dataiter)
    test_x = test_x.to(opt.device)

    test_score = model(test_x)
    accuracy = np.mean((t.argmax(test_score.to('cpu'),1)==test_y).numpy())
    print('test accuracy:%.2f' % accuracy)
    return accuracy


def attack_model_pixel():
    '''
    pixel攻击模型
    :return:
    '''
    accuracy = test_acc()
    # 1.加载配置
    opt._parese()
    global DOWNLOAD_CIFAR10
    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device).eval()


    # 2.加载数据
    transform = tv.transforms.Compose([
           tv.transforms.ToTensor(),
           tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])

    test_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=False,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)



    success_rate = attack_all(model,test_loader,pixels=1,targeted=False,maxiter=400,popsize=400,verbose=False,device=opt.device,sample=opt.attack_num)
    string = 'model name:{} | accuracy:{} | success rate:{}| time: {}\n'.format(opt.model,accuracy,success_rate, strftime('%m_%d_%H_%M_%S'))
    open('log.txt','a').write(string)

def attack_model_PGD():
    '''
    PGD攻击模型
    :return:
    '''
    accuracy = test_acc()
    # 1.加载配置
    opt._parese()
    global DOWNLOAD_CIFAR10
    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device).eval()


    # 2.加载数据
    transform = tv.transforms.Compose([
           tv.transforms.ToTensor(),
           tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])

    test_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=False,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    success_num = 0
    attack = LinfPGDAttack(model)
    for ii, (data, label) in enumerate(test_loader):
        if ii>=opt.attack_num:
            break
        data,label = data.to(opt.device),label.to(opt.device)
        test_score = model(data)
        if t.argmax(test_score.to('cpu'), 1) == label:
            continue
        perturb_x =attack.perturb(data,label)
        test_score = model(t.FloatTensor(perturb_x).to(opt.device))
        if t.argmax(test_score.to('cpu'), 1) != label:
            success_num+=1


    success_rate = success_num/ii

    string = 'model name:{} | accuracy:{} | success rate:{}| time: {}\n'.format(opt.model,accuracy,success_rate, strftime('%m_%d_%H_%M_%S'))
    open('log.txt','a').write(string)


if __name__ == '__main__':
    test_acc()
    # train()

    # attack_model_pixel()