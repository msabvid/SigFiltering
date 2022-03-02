import torch
import torch.nn as nn




class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        layers.append(nn.BatchNorm1d(sizes[0]))
        for j in range(1,len(sizes)):
            layers.append(nn.Linear(sizes[j-1], sizes[j]))
            if j<(len(sizes)-1):
                layers.append(nn.BatchNorm1d(sizes[j]))
                layers.append(activation())
            else:
                #layers.append(nn.BatchNorm1d(sizes[j]))
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, *args):
        x = torch.cat(args, -1)

        return self.net(x)
