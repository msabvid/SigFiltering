import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from lib.sigcwgan import SigCWGAN
from lib.data import sdeint
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
         rho: float,
         window_length: int,
         base_dir: str,
         **kwargs):

    
    # We generate the data
    print("Generating the data...")
    t = torch.linspace(0,T,n_steps+1).to(device)
    x0 = torch.ones(10000, device=device)
    y0 = torch.ones_like(x0)
    xy = linear_sdeint(x0,y0,t)

    # SigCWGAN
    sigcwgan = SigCWGAN(depth=depth, 
                        x_real_obs=xy[...,1].unsqueeze(2), 
                        x_real_state=xy[...,0].unsqueeze(2),
                        t=t,
                        window_length=window_length)
    sigcwgan.to(device)
    t_future = t[n_steps//2:n_steps//2+20]
    sigcwgan.fit(num_epochs=num_epochs, t_future=t_future, mc_samples=50, batch_size=200)

    # save weights
    res = {"nrde_filtration":sigcwgan.nrde_filtration.state_dict(), "node_gen":sigcwgan.node_gen.state_dict(), "loss_gen":sigcwgan.loss_gen}
    torch.save(res, os.path.join(base_dir, 'res.pth.tar'))

    plot(**locals())

def plot(device: str,
        seed: int,
        num_epochs: int,
        depth: int,
        T: float,
        n_steps: int,
        rho: float,
        window_length: int,
        base_dir: str,
        **kwargs):
    
    t = torch.linspace(0,T,n_steps+1).to(device)
    x0 = torch.ones(10, device=device)
    y0 = torch.ones_like(x0)
    #xy = sde.sdeint(x0,y0,t)
    xy = linear_sdeint(x0,y0,t,)

    res = torch.load(os.path.join(base_dir, 'res.pth.tar'), map_location=device)
    sigcwgan = SigCWGAN(depth=depth, 
                        x_real_obs=xy[...,1].unsqueeze(2), 
                        x_real_state=xy[...,0].unsqueeze(2),
                        t=t,
                        window_length=window_length)
    sigcwgan.to(device)
    sigcwgan.nrde_filtration.load_state_dict(res['nrde_filtration'])
    sigcwgan.node_gen.load_state_dict(res['node_gen'])

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
        pred_ce = sigcwgan._filtering(xy[...,1].unsqueeze(2))
    textstr = '\n'.join((
        r'$dX_t = F(t)X_tdt + C(t)dW_t^X$',
        r'$dY_t = G(t)X_tdt + dW_t^Y$' ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(10):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
        ax[0].plot(to_numpy(t), to_numpy(xy[i,:,1]), label=r"$Y_t(\omega)$") 
        ax[0].plot(to_numpy(t), to_numpy(xy[i,:,0]), label=r"$X_t(\omega)$") 
        ax[0].plot(to_numpy(t[::window_length]), to_numpy(pred_ce[i,:,0]), label="pred nrde")
        ax[0].plot(to_numpy(t), to_numpy(x_ce[i,:,0]), label=r"kalman filter - $E(X_t | F_t^Y)(\omega)$") 
        ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[0].set_title("Filtering")
        ax[0].legend(loc='lower left')
        
        ax[1].plot(to_numpy(t), to_numpy(xy[i,:,1]), label=r"$Y_t(\omega)$")
        ax[1].plot(to_numpy(t_past), to_numpy(xy[i,:len(t_past),0]), label=r"$X_t(\omega)$")
        for j in range(mc_samples):
            ax[1].plot(to_numpy(t_future[::window_length]), to_numpy(pred[i*mc_samples + j,:,0]), color="green", alpha=0.2)
        ax[1].legend()
        ax[1].set_title("Prediction")
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, "filtering_{}.pdf".format(i)))
        plt.close()






if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)

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
