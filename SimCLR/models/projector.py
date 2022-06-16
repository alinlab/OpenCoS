import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, expansion=4):
        super(Projector, self).__init__()

        if expansion == 0:
            self.linear_1 = nn.Linear(128, 128)
            self.linear_2 = nn.Linear(128, 128)
        else:
            self.linear_1 = nn.Linear(512*expansion, 2048)
            self.linear_2 = nn.Linear(2048, 2048)

    def forward(self, x, internal_output_list=False):
            
        #output_list = []

        output = self.linear_1(x)
        output = F.relu(output)
        #output_list.append(output)

        output = self.linear_2(output)

        #output_list.append(output)


        return output 

