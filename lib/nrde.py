import torch
import torch.nn as nn
import signatory
import torchcde
from typing import List

from lib.networks import FFN



class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.net = FFN(sizes=[hidden_channels + 1, 128, input_channels * hidden_channels], output_activation=nn.Tanh)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        # time is necessary because we are solving the filtering problem
        t = t * torch.ones(z.shape[0], 1, device=z.device)
        return self.net(t,z).view(z.shape[0], self.hidden_channels, self.input_channels)
        #z = self.linear1(z)
        #z = z.relu()
        #z = self.linear2(z)
        #z = z.tanh()
        #z = z.view(z.size(0), self.hidden_channels, self.input_channels) # (batch_size, hidden_channels, input_channels)
        #return z


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic", gen=False):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.gen = gen
        #if gen:
        #    self.initial = FFN(sizes=[input_channels+hidden_channels, 20, hidden_channels], output_activation=nn.Tanh)
        #else:
        self.initial = FFN(sizes=[input_channels, 20, hidden_channels], output_activation=nn.Tanh)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs, t: str, **kwargs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        # Initial hidden state should be a function of the first observation.
        X0 = X.evaluate(X.interval[0])
        if self.gen:
            #noise = torch.randn(X0.shape[0],1,device=X0.device)
            #hidden_state = kwargs['z']
            #z0 = self.initial(X0, hidden_state)#, noise)
            z0 = kwargs['z']# + self.initial(torch.randn_like(X0))
        else:
            z0 = self.initial(X0)

        # Solve the CDE.
        z = torchcde.cdeint(X=X, z0=z0, func=self.func, t=getattr(X, t), adjoint=False)
        pred_y = self.readout(z)
        return z, pred_y






