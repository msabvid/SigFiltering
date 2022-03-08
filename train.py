import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from lib.sigcwgan import SigCWGAN
from lib.data import SDE_Linear as SDE
from lib.utils import to_numpy

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def main(device: str,
         seed: int,
         num_epochs: int,
         depth: int,
         T: float,
         n_steps: int,
         sigma: float,
         rho: float,
         window_length: int,
         base_dir: str,
         **kwargs):

    
    # We generate the data
    print("Generating the data...")
    sde = SDE(rho=rho, sigma=sigma)
    t = torch.linspace(0,T,n_steps+1).to(device)
    x0 = torch.ones(20000, device=device)
    y0 = torch.ones_like(x0)
    xy = sde.sdeint(x0,y0,t)

    # SigCWGAN
    sigcwgan = SigCWGAN(depth=depth, 
                        x_real_obs=xy[...,1].unsqueeze(2), 
                        x_real_state=xy[...,0].unsqueeze(2),
                        t=t,
                        window_length=window_length)
    sigcwgan.to(device)
    t_future = t[n_steps//2:]
    sigcwgan.train(num_epochs=num_epochs, t_future=t_future, mc_samples=50)

    # save weights
    res = {"rde_xy":sigcwgan.neural_rde_xy.state_dict(), "rde_gen":sigcwgan.neural_rde_gen.state_dict(),
            "loss_xy":sigcwgan.loss_xy, "loss_gen":sigcwgan.loss_gen}
    torch.save(res, os.path.join(base_dir, 'res.pth.tar'))

    plot(**locals())

def plot(device: str,
        seed: int,
        num_epochs: int,
        depth: int,
        T: float,
        n_steps: int,
        sigma: float,
        rho: float,
        window_length: int,
        base_dir: str,
        **kwargs):
    
    sde = SDE(rho=rho, sigma=sigma)
    t = torch.linspace(0,T,n_steps+1).to(device)
    x0 = torch.ones(10, device=device)
    y0 = torch.ones_like(x0)
    xy = sde.sdeint(x0,y0,t)

    res = torch.load(os.path.join(base_dir, 'res.pth.tar'), map_location=device)
    sigcwgan = SigCWGAN(depth=depth, 
                        x_real_obs=xy[...,1].unsqueeze(2), 
                        x_real_state=xy[...,0].unsqueeze(2),
                        t=t,
                        window_length=window_length)
    sigcwgan.to(device)
    sigcwgan.neural_rde_xy.load_state_dict(res['rde_xy'])
    sigcwgan.neural_rde_gen.load_state_dict(res['rde_gen'])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    ax[0].plot(res['loss_xy'])
    ax[0].set_title('Loss xy')
    ax[1].plot(res['loss_gen'])
    ax[1].set_title('loss gen')
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, 'loss.pdf'))
    
    t_future = t[n_steps//2:]
    t_past = t[t<=t_future[0]]
    mc_samples=50
    with torch.no_grad():
        pred = sigcwgan.sample(x_real_obs = xy[...,1].unsqueeze(2), t_future=t[n_steps//2:], mc_samples=mc_samples)
    
    for i in range(10):
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(to_numpy(t), to_numpy(xy[i,:,1]), label="obs")
        ax.plot(to_numpy(t_past), to_numpy(xy[i,:len(t_past),0]), label="state")
        for j in range(mc_samples):
            ax.plot(to_numpy(t_future[::window_length]), to_numpy(pred[i*mc_samples + j,:,0]), color="green", alpha=0.2)
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, "sample_{}.pdf".format(i)))
        plt.close()

    pred = sigcwgan.predict(xy[...,1].unsqueeze(2))
    for i in range(10):
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(to_numpy(t), to_numpy(xy[i,:,1]), label="obs") 
        ax.plot(to_numpy(t), to_numpy(xy[i,:,0]), label="state") 
        ax.plot(to_numpy(t[::window_length]), to_numpy(pred[i,:,0]), label="pred")
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, "pred_{}.pdf".format(i)))
        plt.close()


        







if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)


    parser.add_argument('--sigma', default=0.3, type=float, help='diffusion X process')
    parser.add_argument('--rho', default=0., type=float)

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=1., type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--window_length', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--plot', action='store_true', default=False)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    config = vars(args)
    config.pop('device')
    config['device'] = device

    results_path = args.base_dir#os.path.join(args.base_dir, "BS", args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if args.plot:
        plot(**config)
    else:
        main(**config)
