import os
import pickle
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


class SDE(nn.Module):

    def __init__(self, rho, sigma):
        
        super().__init__()
        self.rho = rho
        self.sigma = sigma

    def b(self, x, y):
        """
        Drift of state process Xt
        """
        # TODO

    def B(self, x, y):
        """
        Drift of observation process Yt
        """
        # TODO

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
            z = torch.randn(batch_size, 2)
            dW = torch.sqrt(h) * z
            dW[:,0] = self.rho**2 * dW[:,0] + (1-self.rho**2) * dW[:,1]

            x = x + self.b(x,y)*h + self.sigma * dW[:,0]
            y = y + self.B(x,y)*h + dW[:,1]

            xy = torch.cat([xy, torch.stack((x,y),1).unsqueeze(1)],1)
        return xy






def rolling_window(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)




def get_data(data_type, p, q, **data_params):
    
    if data_type == 'BTCUSD':
        filename = data_params.get('filename')
        if filename == None:
            x_real = pd.read_csv('data/bitmex_BTCUSD_1m.csv', header=0)
        else:
            x_real = pd.read_csv(filename, header=0)

        #x_real['log_return'] = np.log(x_real['<CLOSE>'] / x_real['<OPEN>'])
        x_real['close_heikenashi'] = 0.25 * (x_real['<CLOSE>'] + x_real['<OPEN>'] + x_real['<HIGH>'] + x_real['<LOW>'])
        open_heikenashi = 0.5 * (x_real['<OPEN>'] + x_real['<CLOSE>'])
        x_real['open_heikenashi'] = np.zeros(x_real['close_heikenashi'].shape)
        x_real.loc[1:,'open_heikenashi'] = open_heikenashi[:-1].values
        x_real['log_return_HA'] = np.log(x_real['close_heikenashi'] / x_real['open_heikenashi'])
        
        
        columns = ['open_heikenashi','close_heikenashi','log_return_HA','<OPEN>','<CLOSE>']
        x_real = torch.tensor(x_real.loc[1:,columns].values).float()
        #x_open_HA = x_real['open_heikenashi'].values[1:]
        #x_open_HA = torch.tensor(x_open_HA[...,None]).float()
        #x_close_HA = x_real['close_heikenashi'].values[1:]
        #x_close_HA = torch.tensor(x_close_HA[...,None]).float()
        #x_open = torch.tensor(x_real['<OPEN>'].values[1:]).unsqueeze(1).float()
        #x_close = torch.tensor(x_real['<CLOSE>'].values[1:]).unsqueeze(1).float()

        #x_real_HA = x_real['log_return_HA'].values[1:]
        #x_real_HA = torch.tensor(x_real_HA[...,None]).float()
        

        
    else:
        raise NotImplementedError('Dataset %s not valid' % data_type)
    #assert x_real.shape[0] == 1
    x_real = rolling_window(x_real, p+q)
    #x_real_HA = rolling_window(x_real_HA, p + q)
    #x_open_HA = rolling_window(x_open_HA, p + q)
    #x_close_HA = rolling_window(x_close_HA, p + q)
    return x_real, columns #x_real_HA, x_open_HA, x_close_HA, x_open, x_close


def get_data_prediction(filename: str, p: int = 720):
    x_real = pd.read_csv(filename, header=0)
    assert x_real.shape[0] == 721, "need 721 data points to calculate 720 HA data points"

    x_real['close_heikenashi'] = 0.25 * (x_real['<CLOSE>'] + x_real['<OPEN>'] + x_real['<HIGH>'] + x_real['<LOW>'])
    open_heikenashi = 0.5 * (x_real['<OPEN>'] + x_real['<CLOSE>'])
    x_real['open_heikenashi'] = np.zeros(x_real['close_heikenashi'].shape)
    x_real.loc[1:,'open_heikenashi'] = open_heikenashi[:-1].tolist()
    x_real['log_return'] = np.log(x_real['close_heikenashi'] / x_real['open_heikenashi'])
    
    x_open = x_real['open_heikenashi'].values[1:]
    x_open = torch.tensor(x_open[...,None]).float()
    x_close = x_real['close_heikenashi'].values[1:]
    x_close = torch.tensor(x_close[...,None]).float()

    x_real = x_real['log_return'].values[1:]
    x_real = torch.tensor(x_real[...,None]).float()
    
    x_real = x_real.unsqueeze(0)#€rolling_window(x_real, p)
    x_open = x_open.unsqueeze(0)#rolling_window(x_open, p)
    x_close = x_close.unsqueeze(0)#rolling_window(x_close, p)
    return x_real, x_open, x_close





def binarize_it(v):
    output = (v>0).astype(int)
    return output


def test_heikenashi():

    x_real = pd.read_csv('data/bitmex_BTCUSD_1m.csv', header=0)
    #x_real['log_return'] = np.log(x_real['<CLOSE>'] / x_real['<OPEN>'])
    x_real['close_heikenashi'] = 0.25 * (x_real['<CLOSE>'] + x_real['<OPEN>'] + x_real['<HIGH>'] + x_real['<LOW>'])
    open_heikenashi = 0.5 * (x_real['<OPEN>'] + x_real['<CLOSE>'])
    x_real['open_heikenashi'] = np.zeros(x_real['close_heikenashi'].shape)
    x_real.loc[1:,'open_heikenashi'] = open_heikenashi[:-1].tolist()
    x_real['log_return_heikenashi'] = np.log(x_real['close_heikenashi'] / x_real['open_heikenashi'])
    x_real['log_return']  = np.log(x_real['<CLOSE>']/x_real['<OPEN>'])    
    x_real['bin_log_return_heikenashi'] = binarize_it(x_real['log_return_heikenashi'])
    x_real['bin_log_return'] = binarize_it(x_real['log_return'])
    heikenashi_wrong = np.logical_xor(x_real['bin_log_return_heikenashi'], x_real['bin_log_return'])
    x_real['heikenashi_correct'] = np.logical_not(heikenashi_wrong)
    x_real.to_csv('data/heikenashi.csv')
    confusion_matrix_ = confusion_matrix(x_real['bin_log_return_heikenashi'], x_real['bin_log_return'])
    print(confusion_matrix_)
    print(confusion_matrix_ / confusion_matrix_.sum())
    return 0


if __name__ == '__main__':
    test_heikenashi()

