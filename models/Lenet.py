import torch.nn as nn
from models.basic_module import BasicModule
class LeNet5(BasicModule):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.model_name = 'Lenet5'

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 16, 5, 5]
            nn.Conv2d(3, 6, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )




    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return:
        """
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)

        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(-1, 16 * 5 * 5)

        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)
        # [b, 10]
        return logits