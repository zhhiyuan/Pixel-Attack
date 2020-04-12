#coding:utf-8
import torch as t
from time import strftime
class BasicModule(t.nn.Module):
    """
    封装了nn.Module，主要提供了save和load两个方法
    """
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))#默认名字

    def load(self,path):
        '''
        加载指定路径的模型
        :param path: 路径
        :return:
        '''
        self.load_state_dict(t.load(path))

    def save(self,name=None):
        if name is None:
            prefix = './ckps/' + self.model_name + '_'
            name = strftime(prefix + '%m_%d_%H_%M.pth')
        t.save(self.state_dict(), name)
        print('model saved at {}'.format(name))




