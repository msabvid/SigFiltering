import torch
import torch.nn as nn
import signatory
import torchcde
import torchdiffeq
from functools import partial


from typing import List
from lib.networks import FFN



class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.net = FFN(sizes=[hidden_channels+1, 128, input_channels * hidden_channels], output_activation=nn.Tanh)

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


class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = FFN(sizes=[input_channels, 20, hidden_channels], output_activation=nn.Tanh)
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
        z0 = self.initial(X0)

        # Solve the CDE.
        z = torchcde.cdeint(X=X, z0=z0, func=self.func, t=getattr(X, t), adjoint=False)
        return z



class ODEFunc(nn.Module):
    def __init__(self, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.net = FFN(sizes=[hidden_channels + 1, 128, hidden_channels])
        #self.net = FFN(sizes=[1, 128, hidden_channels])
        self._z0 = None

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, new_z0):
        self._z0 = new_z0

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        t = t * torch.ones(z.shape[0], 1, device=z.device)
        return self.net(t,self._z0)
        #return self.net(t, z)





class NeuralODE(nn.Module):
    def __init__(self, hidden_channels, output_channels, gen: bool=True):
        super().__init__()

        if gen:
            sizes = [hidden_channels+1, 20, output_channels]
        else:
            sizes = [hidden_channels, 20, output_channels]
        self.gen = gen
        
        self.initial = FFN(sizes=sizes)# +1 is because of input noise
        self.func = ODEFunc(output_channels)
        #self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, z0, t: torch.Tensor):
        # Solve the ODE.
        batch_size = z0.shape[0]
        if self.gen:
            noise = torch.randn(batch_size, 1, device=z0.device)
            y0 = self.initial(z0, noise)
        else:
            y0 = self.initial(z0)
        self.func.z0 = y0
        y = torchdiffeq.odeint(self.func, y0, t)# , z0=z0, func=self.func, t=getattr(X, t), adjoint=False)
        y = y.permute((1,0,2)) # (batch_size, L, dim)
        #pred_y = self.readout(z)
        return y#, pred_y



