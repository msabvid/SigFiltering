import os
import math
import pickle
import torch
import torch.nn as nn
from collections.abc import Callable

from scipy.integrate import solve_ivp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from lib.utils import to_numpy




def F(t: float):
    return 0.1

def G(t: float):
    return 0.2

def C(t: float):
    return 1

def D(t: float):
    return 1


def linear_sdeint(x0, y0, ts, F: Callable, C: Callable, G: Callable, D: Callable):
    """
    Linear SDE solver using Euler scheme
    
    dXt = F(t)Xt dt + C(t)dWt
    dYt = G(t)Xt dt + D(t)dZt 

    Z, W independent brownian motions

    Parameters
    ----------
    x0: torch.Tensor
        Initial x0 state vector. Tensor of shape (batch_size)
    y0: torch.Tensor
        Initial y0 observation vector. Tensor of shape (batch_size)
    ts: torch.Tensor
        Time discretisation
    F,C,G,D: Callable[[float], float]

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

        x = x + F(t) * x * h + C(t) * dW[:,0]
        y = y + G(t) * x * h + D(t) * dW[:,1]

        xy = torch.cat([xy, torch.stack((x,y),1).unsqueeze(1)],1)
    return xy


def func_riccati(t, y, F: Callable, C: Callable, G: Callable, D: Callable):
    """
    Function of the Ricatti equation for S(t) --> See Oksendal book Theorem 6.2.8
    d S(t) / dt  = 2F(t)S(t) - G^2(t)/D^2(t) S^2(t) + C^2(t)
    """
    return 2*F(t)*y - (G(t)**2 / D(t)**2) * y**2 + C(t)**2



def kalman_filter(obs: torch.Tensor, x0: torch.Tensor, ts: torch.Tensor, F: Callable, C: Callable, G: Callable, D: Callable):
    """
    Kalman filter: Let \hat X_t := E(X_t | F_t^Y) where X_t is the state and Y_t is the observation 
    d \hat_X = (F(t) - (G^2(t)S(t))/D^2(t) ) * \hat X_t dt + (G(t)S(t))/D^2(t) dY_t ; \hat X_0 = E(X_0)
    Parameters
    ----------
    x0: torch.Tensor
        Initial x0 state vector. Tensor of shape (batch_size)
    obs: torch.Tensor
        Observation process. Tensor of shape (batch_size, L, 1)
    ts: torch.Tensor
        Time discretisation
    F,C,G,D: Callable[[float], float]

    Returns
    -------
    x_ce : torch.Tensor
        Conditional expectation E(X | F_t^Y). Tensor of shape (batch_size, L, 1)
    """
    
    # solve the riccati equation
    S0 = x0.var().item() # Oksendal Thm 6.2.8
    S = solve_ivp(func_riccati, t_span=(to_numpy(ts[0]), to_numpy(ts[-1])), y0=[S0,], t_eval=to_numpy(ts), args=(F, C, G, D)).y
    S = torch.from_numpy(S).float().to(x0.device)
    S = S.squeeze(0)
    
    #ts = to_numpy(ts)
    x_ce = torch.zeros(x0.shape[0], len(ts), 1, device=x0.device)
    x_ce[:,0,:] = x0.mean()
    for i, t in enumerate(ts[:-1]):
        h = ts[i+1] - ts[i]
        dY = obs[:,i+1,:] - obs[:,i,:]
        x_ce[:,i+1,:] = x_ce[:,i,:] + (F(t) - (G(t)**2*S[i])/D(t)**2 ) * x_ce[:,i,:] * h + (G(t)*S[i])/D(t)**2 * dY

    return x_ce

