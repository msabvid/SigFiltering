import torch
import torch.nn as nn
import signatory
import torchcde
#from sklearn.linear_model import LinearRegression
from torch import optim
from tqdm import tqdm

from lib.nrde import NeuralCDE
from lib.networks import LinearRegression 
from lib.utils import to_numpy


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


def compute_sig(depth, x, stream: bool = False):
    """
    Parameters
    ----------
    depth: int
        depth of signature
    x: torch.Tensor
        path. Tensor of shape [batch_size, L, dim]
    stream: bool
        Bool indicating whether we calculate the logsig of all paths (x)_{1..j} for j=2..L
    
    Returns
    -------
    logsig: torch.Tensor
        if stream=True, tensor of shape [batch_size, L, logsig_channels]
        if stream=False, tensor of shape [batch_size, logsig_channels]
    """
    return signatory.signature(x, depth, stream=stream, basepoint=True)


def calibrate_sigw1_metric(depth, x_past_obs, x_future_state):
    """
    Calibrates the sigw1_metric to calculate conditional expectations from the real measure
    """

    sig_past_obs = compute_sig(depth=depth, x=x_past_obs, stream=False)
    sig_future_state = compute_sig(depth=depth, x=x_future_state, stream=False)
    
    #X, Y = to_numpy(sig_past_obs), to_numpy(sig_future_state)
    #lm = LinearRegression()
    #lm.fit(X,Y)
    #sig_pred = torch.from_numpy(lm.predict(X)).float().to(x_future_state.device)
    lm = LinearRegression(n_in = sig_past_obs.shape[-1], n_out=sig_future_state.shape[-1])
    lm.to(sig_past_obs.device)
    lm.fit(X=sig_past_obs, Y=sig_future_state, n_epochs=10000)
    sig_pred = lm(sig_past_obs)

    return sig_pred



def augment_with_time(t, *args):
    """
    Augment all the paths in args with time
    """
    for x in args:
        ts = t.reshape(1,-1,1).repeat(x.shape[0],1,1)
        yield torch.cat([ts, x],2)



class SigCWGAN(nn.Module):
    
    def __init__(self, depth: int, x_real_obs: torch.Tensor, x_real_state: torch.Tensor, t: torch.Tensor, window_length: int):
        super().__init__()

        self.depth = depth
        self.x_real_obs = x_real_obs
        self.x_real_state = x_real_state
        self.t = t
        self.window_length = window_length
        
        # 1. Neural RDE to model (X,Y)
        logsig_channels = signatory.logsignature_channels(in_channels=x_real_obs.shape[-1]+1, depth=depth) # +1 because of time
        self.neural_rde_xy = NeuralCDE(input_channels=logsig_channels, hidden_channels=8, output_channels=1, interpolation="linear")
        # 2. Neural RDE to generate new future data
        self.neural_rde_gen = NeuralCDE(input_channels = logsig_channels, hidden_channels=8, output_channels=1, interpolation="linear", gen=True)

        
    def train(self, num_epochs, t_future: torch.Tensor, mc_samples: int):
        
        # 1. Training of first Neural RDE
        print("Training Neural RDE to model (X,Y)")
        x_real_obs_t, = augment_with_time(self.t, self.x_real_obs)
        obs_logsig = torchcde.logsig_windows(x_real_obs_t, self.depth, window_length=self.window_length)
        obs_coeffs = torchcde.linear_interpolation_coeffs(obs_logsig)

        train_dataset = torch.utils.data.TensorDataset(obs_coeffs, self.x_real_state)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 200)

        optimizer = torch.optim.Adam(self.neural_rde_xy.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            for i, (batch_coeffs, batch_y) in enumerate(train_dataloader):
                pbar.write("batch {} of {}".format(i, len(train_dataloader)))
                optimizer.zero_grad()
                _, pred = self.neural_rde_xy(batch_coeffs)
                loss = loss_fn(pred, batch_y[:,::self.window_length])
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.write("loss:{:.4f}".format(loss.item()))

        # 2. Training of second Neural RDE
        print("Training Neural RDE that generates future data")
        x_real_future = self.x_real_obs[:,self.t>=t_future[0],:]
        sig_x_real_future_ce = calibrate_sigw1_metric(depth=self.depth, 
                                                      x_past_obs=self.x_real_obs[:,self.t<=t_future[0]],
                                                      x_future_state=x_real_future[:,::self.window_length]
                                                     ).detach()
        h = t_future[1:] - t_future[:-1]
        brownian = torch.zeros(self.x_real_obs.shape[0], len(t_future), 1, device=self.x_real_obs.device)
        brownian[:,1:,:] = torch.sqrt(h.reshape(1,-1,1)) * torch.randn_like(brownian[:,1:,:])
        brownian_t, = augment_with_time(t_future, brownian)
        brownian_logsig = torchcde.logsig_windows(brownian_t, self.depth, window_length=self.window_length)
        
        brownian_coeffs = torchcde.linear_interpolation_coeffs(brownian_logsig)

        obs_logsig = torchcde.logsig_windows(x_real_obs_t[:,self.t<=t_future[0]], self.depth, window_length=self.window_length)
        obs_coeffs = torchcde.linear_interpolation_coeffs(obs_logsig)
        train_dataset = torch.utils.data.TensorDataset(brownian_coeffs, obs_coeffs, sig_x_real_future_ce)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 200)

        optimizer = torch.optim.Adam(self.neural_rde_gen.parameters(), lr=0.001)
        
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            for i, (batch_br_coeffs, batch_obs_coeffs, batch_sig_x_real) in enumerate(train_dataloader):
                pbar.write("batch {} of {}".format(i, len(train_dataloader)))
                optimizer.zero_grad()
                with torch.no_grad():
                    z, _ = self.neural_rde_xy(batch_obs_coeffs)
            
                # Monte Carlo!!!!
                batch_br_coeffs_mc = batch_br_coeffs.repeat(mc_samples, *[1]*(batch_br_coeffs.dim()-1))
                z_mc = z[:,-1,:].repeat(mc_samples, 1)
                _, pred = self.neural_rde_gen(batch_br_coeffs_mc, z=z_mc)
                sig_pred_ce = compute_sig(depth=self.depth, x=pred).reshape(mc_samples, *batch_sig_x_real.shape).mean(0)
                loss = sigcwgan_loss(batch_sig_x_real, sig_pred_ce)
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.write("loss:{:.4f}".format(loss.item()))
        
        return 0
                

