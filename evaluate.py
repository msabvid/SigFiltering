import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product

from lib.sigcwgan import SigCWGAN
from lib.conditional_expectation import ConditionalExpectation
from lib.data import linear_sdeint as sdeint
from lib.data import kalman_filter, F, C, G, D
from lib.utils import to_numpy, set_seed

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def main(device: str,
         seed: int,
         depth: int,
         T: float,
         n_steps: int,
         window_length: int,
         base_dir: str,
         **kwargs):
    
    set_seed(10)

    sigcwgan_dir = os.path.join(base_dir, 'sigcwgan')
    # We generate the data
    print("Generating the data...")
    t = torch.linspace(0,T,n_steps+1).to(device)
    x0 = torch.ones(10, device=device)
    y0 = torch.ones_like(x0)
    xy = sdeint(x0,y0,t)
    x_ce = kalman_filter(obs = xy[...,1].unsqueeze(2), x0=x0, ts=t, F=F, C=C, G=G, D=D)
    x_ce = to_numpy(x_ce)


    # SigCWGAN
    sigcwgan = SigCWGAN(depth=depth, 
                        x_real_obs=xy[...,1].unsqueeze(2), 
                        x_real_state=xy[...,0].unsqueeze(2),
                        t=t,
                        window_length=window_length)
    sigcwgan.to(device)
    sigcwgan.load(os.path.join(sigcwgan_dir, 'res.pth.tar'), device=device)
    t_future = t[n_steps//2:]#n_steps//2+20]
    sample = sigcwgan.sample(x_real_obs=xy[...,1].unsqueeze(2), 
                             t_future=t_future,
                             mc_samples=1000)

    
    field = []
    sample_G = sample.view(-1,len(t_future), 1)
    for i, t_ in enumerate(t_future):
        field.append(sigcwgan.node_gen.func(t_, sample_G[:,i,:]))
    field = torch.stack(field,1)
    sample = to_numpy(sample)
    
    
    # 1st, 2nd and 3rd order moments
    #ce_results = defaultdict(list)
    #for seed, m in product(range(1), [1,2,3]):
    #    ce_path = os.path.join(base_dir, 'ce', 'seed{}'.format(seed), 'm={}'.format(m))
    #    ce = ConditionalExpectation(depth=depth, 
    #                                x_real_obs=xy[...,1].unsqueeze(2), 
    #                                x_real_state=xy[...,0].unsqueeze(2),
    #                                t=t,
    #                                window_length=window_length,
    #                                m=m)
    #    ce.to(device)
    #    ce.load(os.path.join(ce_path, 'res.pth.tar'), device=device)
    #    with torch.no_grad():
    #        ce_results[m].append(ce(t_future=t_future, batch_x_real_obs=xy[...,1].unsqueeze(2)))
    #
    #ce_results2 = {}
    #for m in [1,2,3]:
    #    ce_results2[m] = to_numpy(torch.stack(ce_results[m], 0))
    xy = to_numpy(xy)
    t = to_numpy(t)
    t_future = to_numpy(t_future)
    ind_past = (t <= t_future[0])
    for i in range(10):
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(t[ind_past], xy[i,ind_past,0], label=r'$X_t(\omega)$')
        ax.plot(t[ind_past], xy[i,ind_past,1], label=r'$Y_t(\omega)$')
        ax.plot(t, x_ce[i,:,0], label=r'Kalman filter - $E[X_t | F_t^Y](\omega)$')
        #for j in range(ce_results2[1].shape[0]):
        #    ax.plot(t_future, ce_results2[m][j,i,:,0], color='silver', alpha=0.5)
        ax.plot(t_future, sample.mean(0)[i], linewidth=2, label='CSigWGAN')
        for j in range(min(90,sample.shape[0])):
            ax.plot(t_future, sample[j,i,:,0], color='red', alpha=0.2, linewidth=0.5)
        ax.set_xlabel('time')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, 'result_{}.pdf'.format(i)))
        plt.close()
        
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(sample[:,i,0], bins=50, density=True)
        fig.savefig(os.path.join(base_dir, 'hist_{}.pdf'.format(i)))
        plt.close()
    return 0





if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--depth', default=3, type=int)
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

    main(**config)
