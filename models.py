import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 이 models dict와 models.py 는 딥러닝 repository와 동일하게 유지되어야 함 
# class 가독성을 놎이기 위해 network 부분만 따로 분리
class CFLinear(nn.Module):
    def __init__(self, state_size=6*7, action_size=7, hidden_size=64):
        super(CFLinear,self).__init__()
        self.model_type = 'Linear'
        
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)

        self.layers = [self.linear1, self.linear2, self.linear3]
        for layer in self.layers:
            if type(layer) in [nn.Conv2d, nn.Linear]:
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            layer = layer.to(device)
        
    def forward(self, x):
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        y = self.linear3(y)
        return y

# class 가독성을 높이기 위해 network 부분만 따로 분리 
class CFCNN(nn.Module):
    def __init__(self, action_size=7):
        super(CFCNN,self).__init__()
        self.model_type = 'CNN'
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,2), stride=1,padding=1)
        # self.conv2 = nn.Conv2d(32,64,(2,2), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64,64,(2,2), stride=1, padding=1)
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4), stride=1,padding=2)
        # self.conv2 = nn.Conv2d(32,64,(4,4), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64,64,(4,4), stride=1, padding=1)
        # self.linear1 = nn.Linear(64*3*4, 64)
        # self.linear2 = nn.Linear(64, action_size)

        self.conv1 = nn.Conv2d(1,42,(4,4), stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear1 = nn.Linear(42*3*4, 42)
        self.linear2 = nn.Linear(42, action_size)
        
        self.layers = [
            self.conv1,
            # self.conv2,
            # self.conv3,
            self.maxpool1,
            self.linear1,
            self.linear2
        ]

        # relu activation 함수를 사용하므로, He 가중치 사용
        for layer in self.layers:
            if type(layer) in [nn.Conv2d, nn.Linear]:
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            layer = layer.to(device)


    def forward(self, x):
        y = F.relu(self.conv1(x))  # (N, 42, 6, 7)
        y = self.maxpool1(y)  # (N, 42, 3, 4)
        y = y.flatten(start_dim=2)  # (N, 42, 12)
        # view로 채널 차원을 마지막으로 빼줌
        # 정확한 이유는 나중에 알아봐야 할듯? 
        y = y.view(y.shape[0], -1, 42)  # (N, 12, 42)
        y = y.flatten(start_dim=1)  # (N, 12*42)
        y = F.relu(self.linear1(y))
        y = self.linear2(y)
        return y

    # def forward(self,x):
    #     # (N, 1, 6,7)
    #     y = F.relu(self.conv1(x))
    #     # (N, 32, 7,8)
    #     y = F.relu(self.conv2(y))
    #     # (N, 64, 8,9)
    #     y = F.relu(self.conv3(y))
    #     # (N, 64, 9,10)
    #     #print("shape x after conv:",y.shape)
    #     y = y.flatten(start_dim=2)
    #     # (N, 64, 90)
    #     #print("shape x after flatten:",y.shape)
    #     y = y.view(y.shape[0], -1, 64)
    #     # (N, 90, 64)
    #     #print("shape x after view:",y.shape)
    #     y = y.flatten(start_dim=1)
    #     # (N, 90*64)
    #     y = F.relu(self.linear1(y))
    #     # (N, 64)
    #     y = self.linear2(y) # size N, 12
    #     # (N, 12)
    #     return y.cuda()

# heuristic model을 이용하기 위한 껍데기
# action을 선택할 때 2차원 배열을 그대로 이용하므로 'cnn'으로 둠
class HeuristicModel():
    def __init__(self):
        self.model_type = 'CNN'