from torchvision import models
import torch.nn as nn

all=['vgg11','vgg13','vgg16','vgg19']
class vgg16:
    def __init__(self):
        super(vgg16,self).__init__()
        self.model_name = 'vgg16'
        self.model = models.vgg16(pretrained=True)
        self.classifier=nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 10),
        )

    def forward(self,input):
        out = self.model(input)
        out = self.classifier(out)
        return out

class vgg11:
    def __init__(self):
        super(vgg11,self).__init__()
        self.model_name = 'vgg11'
        self.model = models.vgg11(pretrained=True)
        self.classifier=nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 10),
        )

    def forward(self,input):
        out = self.model(input)
        out = self.classifier(out)
        return out

class vgg13:
    def __init__(self):
        super(vgg13,self).__init__()
        self.model_name = 'vgg13'
        self.model = models.vgg13(pretrained=True)
        self.classifier=nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 10),
        )

    def forward(self,input):
        out = self.model(input)
        out = self.classifier(out)
        return out

class vgg19:
    def __init__(self):
        super(vgg19,self).__init__()
        self.model_name = 'vgg19'
        self.model = models.vgg19(pretrained=True)
        self.classifier=nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 10),
        )

    def forward(self,input):
        out = self.model(input)
        out = self.classifier(out)
        return out
