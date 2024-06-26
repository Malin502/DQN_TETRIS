import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(64, 1))


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
