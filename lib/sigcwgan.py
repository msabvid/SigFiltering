import torch
import torch.nn as nn
import signatory
import torchcde
from torch import optim
from torchdiffeq import odeint

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from lib.nde import NeuralCDE, NeuralODE
from lib.networks import FFN
from lib.utils import to_numpy, toggle
from lib.augmentations import VisiTrans, Cumsum, AddTime, Scale, Basepoint, apply_augmentations


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


def compute_sig(depth, x, stream: bool = False, basepoint=False):
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
    #augmentations = (Basepoint(), Cumsum(), AddTime(), Scale(scale=0.05))
    augmentations = (Basepoint(), AddTime(), Scale(scale=0.5))
    y = apply_augmentations(x=x, augmentations=augmentations)
    return signatory.signature(y, depth, stream=stream, basepoint=basepoint)


def calibrate_sigw1_metric(depth, x_past_obs, x_future_state):
    """
    Calibrates the sigw1_metric to calculate conditional expectations from the real measure
    """

    sig_past_obs = compute_sig(depth=depth+1, x=x_past_obs, stream=False)
    sig_future_state = compute_sig(depth=depth, x=x_future_state, stream=False)
    
    #X, Y = to_numpy(sig_past_obs), to_numpy(sig_future_state)
    #lm = LinearRegression()
    #lm.fit(X,Y)
    #sig_pred = torch.from_numpy(lm.predict(X)).float().to(x_future_state.device)
    #lm = LinearRegression(n_in = sig_past_obs.shape[-1], n_out=sig_future_state.shape[-1])
    #lm.to(sig_past_obs.device)
    #lm.fit(X=sig_past_obs, Y=sig_future_state, n_epochs=10000)
    #sig_pred = lm(sig_past_obs)
    #sig_pred = torch.from_numpy(sig_pred).float().to(x_past_obs.device)
    
    m = FFN(sizes=[sig_past_obs.shape[-1], 128, sig_future_state.shape[-1]])
    m.to(sig_past_obs.device)
    scaler = StandardScaler()
    sig_past_obs = scaler.fit_transform(to_numpy(sig_past_obs))
    sig_past_obs = torch.from_numpy(sig_past_obs).float().to(x_past_obs.device)
    m.fit(X=sig_past_obs, Y=sig_future_state, n_epochs=30000)

    with torch.no_grad():
        sig_pred = m(sig_past_obs)
    
    return sig_pred.detach()



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
        
        # 1. Neural RDE to model the information carried by the filtration F_t^Y
        logsig_channels = signatory.logsignature_channels(in_channels=x_real_obs.shape[-1]+1, depth=depth) # +1 because of time
        self.nrde_filtration = NeuralCDE(input_channels=logsig_channels, hidden_channels=10, interpolation="linear")
        # 2. Neural RDE to generate new future data
        self.node_gen = NeuralODE(hidden_channels=10, output_channels=1 )
        
        self.loss_gen = []
       
    def sample(self, x_real_obs: torch.Tensor, t_future: torch.Tensor, mc_samples: int):
        
        ind_past = self.t <= t_future[0]
        batch_size = x_real_obs.shape[0]

        x_real_obs_t, = augment_with_time(self.t[ind_past], x_real_obs[:, ind_past,:])
        obs_logsig = torchcde.logsig_windows(x_real_obs_t, self.depth, window_length=self.window_length)
        obs_coeffs = torchcde.linear_interpolation_coeffs(obs_logsig)
        
        with torch.no_grad():
            z = self.nrde_filtration(obs_coeffs, t='interval')
            z_mc = z[:,-1,:].repeat(mc_samples, 1)
            pred = self.node_gen(z0=z_mc, t=t_future)
        return pred.reshape(mc_samples, batch_size, *pred.shape[1:]) # (mc_samples, batch_size, len(t_future), dim)

    def fit(self, num_epochs: int, batch_size: int, t_future: torch.Tensor, mc_samples: int, **kwargs):

        ind_past = self.t <= t_future[0]
        ind_future = torch.logical_and(self.t >= t_future[0], self.t <= t_future[-1])

        x_real_obs_t, = augment_with_time(self.t[ind_past], self.x_real_obs[:, ind_past,:])
        obs_logsig = torchcde.logsig_windows(x_real_obs_t, self.depth, window_length=self.window_length)
        obs_coeffs = torchcde.linear_interpolation_coeffs(obs_logsig)
        
        x_real_future = self.x_real_state[:, ind_future, :]
        sig_x_real_future_ce = calibrate_sigw1_metric(depth=self.depth, 
                                                      x_past_obs=self.x_real_obs[:,ind_past],
                                                      x_future_state=x_real_future
                                                      )
        train_dataset = torch.utils.data.TensorDataset(obs_coeffs, sig_x_real_future_ce)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        optimizer = optim.RMSprop(list(self.nrde_filtration.parameters())+list(self.node_gen.parameters()), lr=0.004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.98)
    
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            for i, (batch_coeffs, batch_sig_x_real) in enumerate(train_dataloader):
                pbar.write("batch {} of {}".format(i, len(train_dataloader)))
                optimizer.zero_grad()
                z = self.nrde_filtration(batch_coeffs, t='interval')
                # Monte Carlo
                z_mc = z[:,-1,:].repeat(mc_samples, 1)
                pred = self.node_gen(z0=z_mc, t=t_future)
                sig_pred_ce = compute_sig(depth=self.depth, x=pred).reshape(mc_samples, *batch_sig_x_real.shape).mean(0)
                loss = sigcwgan_loss(batch_sig_x_real, sig_pred_ce)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.loss_gen.append(loss.item())
            pbar.update(1)
            pbar.write("loss: {:.4f}".format(loss.item()))
            try:
                self.save(kwargs['filename'])
            except:
                pass

    def save(self, filename):
        state = {'nrde_filtration':self.nrde_filtration.state_dict(), 'node_gen':self.node_gen.state_dict(), 'loss_gen':self.loss_gen}
        torch.save(state, filename)

    def load(self, filename, device: str):
        state = torch.load(filename, map_location = device)
        self.nrde_filtration.load_state_dict(state['nrde_filtration'])
        self.node_gen.load_state_dict(state['node_gen'])



                

