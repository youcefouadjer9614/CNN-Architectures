import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_5(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        
        self.pool = nn.AvgPool2d((2,2), stride=(2,2))
        
        self.fc1 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)
        
    def forward(self,x):
        
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = torch.tanh(x)
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc1(x)
        x = torch.tanh(x)
        
        x = self.out(x)
        
        return x
    
