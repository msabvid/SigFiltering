import os
import pickle
import torch
import torch.nn as nn
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


class SDE():

    """
    dXt = b(Xt,Yt)dt + sigma dWt
    dYt = B(Xt,Yt)dt + dZt

    d<Wt, Zt> = rho dt
    """

    def __init__(self, rho, sigma):
        
        self.rho = rho
        self.sigma = sigma
    
    @abstractmethod
    def b(self, x, y):
        """
        Drift of state process Xt
        """
        ...

    @abstractmethod
    def B(self, x, y):
        """
        Drift of observation process Yt
        """
        ...

    def sdeint(self, x0, y0, ts):
        """
        SDE solver using Euler scheme
        Parameters
        ----------
        x0: torch.Tensor
            Initial x0 state vector. Tensor of shape (batch_size)
        y0: torch.Tensor
            Initial y0 observation vector. Tensor of shape (batch_size)
        ts: torch.Tensor
            Time discretisation

        Returns
        -------
        xy: torch.Tensor
            Process (x,y). Tensor of shape (batch_size, L, 2)
        """
        device = x0.device
        batch_size = x0.shape[0]
        xy = torch.stack((x0,y0), 1).unsqueeze(1) 
        x, y = x0, y0
        for i, t in enumerate(ts[:-1]):
            h = ts[i+1] - ts[i]
            z = torch.randn(batch_size, 2).to(x0.device)
            dW = torch.sqrt(h) * z
            dW[:,0] = self.rho**2 * dW[:,0] + (1-self.rho**2) * dW[:,1]

            x = x + self.b(x,y)*h + self.sigma * dW[:,0]
            y = y + self.B(x,y)*h + dW[:,1]

            xy = torch.cat([xy, torch.stack((x,y),1).unsqueeze(1)],1)
        return xy


class SDE_Linear(SDE):

    def __init__(self, rho, sigma):
        super().__init__(rho=rho, sigma=sigma)

    def b(self, x, y):
        return 0.1*x
    
    def B(self, x, y):
        return 0.2*x
