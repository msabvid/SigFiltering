import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from lib.sigcwgan import SigCWGAN
from lib.data import linear_sdeint as sdeint
from lib.utils import to_numpy, set_seed

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def main(device: str,
         seed: int,
         num_epochs: int,
         depth: int,
         T: float,
         n_steps: int,
         window_length: int,
         results_path: str,
         **kwargs):

    
    # We generate the data
    print("Generating the data...")
    t = torch.linspace(0,T,n_steps+1).to(device)
    x0 = torch.randn(10000, device=device)
    y0 = torch.randn_like(x0)
    xy = sdeint(x0,y0,t)

    # SigCWGAN
    sigcwgan = SigCWGAN(depth=depth, 
                        x_real_obs=xy[...,1].unsqueeze(2), 
                        x_real_state=xy[...,0].unsqueeze(2),
                        t=t,
                        window_length=window_length)
    sigcwgan.to(device)
    t_future = t[n_steps//2:]#n_steps//2+20]
    sigcwgan.fit(num_epochs=num_epochs, t_future=t_future, mc_samples=1000, batch_size=200, filename = os.path.join(results_path, 'res.pth.tar'))
    sigcwgan.save(os.path.join(results_path, 'res.pth.tar'))




if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--T', default=1., type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--window_length', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    config = vars(args)
    config.pop('device')
    config['device'] = device

    results_path = os.path.join(args.base_dir, 'sigcwgan')#os.path.join(args.base_dir, "BS", args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    config['results_path'] = results_path

    set_seed(0) # for reproducibility
    
    main(**config)
