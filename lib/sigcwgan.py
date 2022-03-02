from dataclasses import dataclass

import torch
from sklearn.linear_model import LinearRegression
from torch import optim
from tqdm import tqdm

from lib.nrde import NeuralCDE

from lib.algos.base import BaseAlgo, BaseConfig
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures
from lib.utils import sample_indices, to_numpy


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


def compute_logsig(depth, x, stream: bool = False):
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
    return signatory.logsignature(x, depth, stream=stream, basepoint=True)


def calibrate_sigw1_metric(depth, x_past_obs, x_future_state):
    """
    Calibrates the sigw1_metric to calculate conditional expectations from the real measure
    """

    sig_past_obs = compute_logsig(depth=depth, x=x_past_obs, stream=False)
    sig_future_state = compute_logsig(depth=depth, x=x_future_state, stream=False)
    
    X, Y = to_numpy(sig_past_obs), to_numpy(sig_future_state)
    lm = LinearRegression()
    lm.fit(X,Y)
    sig_pred = torch.from_numpy(lm.predict(X)).float().to(x_future_state.device)

    return sig_pred


def sample_sig_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake


def augment_with_time(t, *args):
    """
    Augment all the paths in args with time
    """
    for x in args:
        ts = t.reshape(1,-1,1).repeat(x.shape[0],1,1)
        yield torch.cat([ts, x],2)


        



class SigCWGAN():
    
    
    def __init__(self, depth: int, x_real_obs: torch.Tensor, x_real_state: torch.Tensor, window_size: int):
        
        self.depth = depth
        self.x_real_obs = x_real_obs
        self.x_real_state = x_real_state
        self.window_size = window_size
        
        # 1. Neural RDE to model (X,Y)
        self._x_real_obs_t, = augment_with_time(t, x_real_obs)
        logsig_channels = signatory.logsignature_channels(in_channels=self._x_real_obs_t.shape[-1], depth=depth)
        self.neural_rde_xy = NeuralCDE(input_channels=logsig_channels, hidden_channels=8, output_channels=1, interpolation="linear")

        # 2. Neural RDE to generate new future data
        self.neural_rde_gen = NeuralCDE(input_channels = logsig_channels, hidden_channels=8, output_channels=1, interpolation="linear", gen=True)

        
        
        self.sig_config = config
        self.mc_size = config.mc_size

        self.x_past = x_real[:, :self.p]


    
    def _train_neural_rde_xy(self, num_epochs):

        train_logsig = torchcde.logsig_windows(self.x_real_obs, self.depth, window_length=self.window_length)
        train_coeffs = torchcde.linear_interpolation_coeffs(train_logsig)

        train_dataset = torch.utils.data.TensorDataset(train_coeffs, self.x_real_state)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32)

        optimizer = torch.optim.Adam(self.neural_rde_xy.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            for batch_coeffs, batch_y in train_dataloader:
                optimizer.zero_grad()
                pred = self.neural_rde_xy(batch_coeffs)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.write("loss:{:.4f}".format(loss.item()))
    
    def _train_neural_rde_gen(self, num_epochs))


    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        self.evaluate(x_fake)
