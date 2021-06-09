import os
from shutil import copyfile

from haste_pytorch import LayerNormLSTM
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.functional import softplus

from traces import Trace
from typing import Any

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader

from block_rnn import block_rnn

from matplotlib.pyplot import *

class ToByteTensor(transforms.ToTensor):
    def __call__(self, img):
        x = super().__call__(img)
        return (x * 255).long()


class Input(nn.Module):

    def __init__(self, n_in, input_shape=(3,32,32)):
        super(Input, self).__init__()
        self.C, self.H, self.W = input_shape
        self.emb_y = nn.Embedding(256, n_in)
        self.emb_c = nn.Embedding(self.C, n_in)
        self.emb_h = nn.Embedding(self.H, n_in)
        self.emb_w = nn.Embedding(self.W, n_in)
        self.emb_y.weight.data.normal_(0.0, 0.01)
        self.emb_c.weight.data.normal_(0.0, 0.01)
        self.emb_h.weight.data.normal_(0.0, 0.01)
        self.emb_w.weight.data.normal_(0.0, 0.01)


    def coord_seq(self, device, mode='HWC'):
        if mode == 'HWC':
            H = torch.arange(self.H, device=device).view(-1, 1, 1).expand(self.H, self.W, self.C)
            W = torch.arange(self.W, device=device).view(1, -1, 1).expand(self.H, self.W, self.C)
            C = torch.arange(self.C, device=device).view(1, 1, -1).expand(self.H, self.W, self.C)
            cs = torch.stack([C, H, W], 3)  # always to be stacked in order CHW !
            cs = cs.reshape(1, -1, 3)
            return cs

        elif mode == 'CHW':
            raise NotImplementedError("mode CHW is not yet implemented")
        else:
            raise ValueError("unknown mode '{}'".format(mode))

    def forward(self, y):
        return self.emb_y(y[..., 0]) + \
               self.emb_c(y[..., 1]) + \
               self.emb_h(y[..., 2]) + \
               self.emb_w(y[..., 3])


class OutputModule(nn.Module):

    def __init__(self, n_hidden, p0=0.0, loc_scale=False):
        super().__init__()
        self.p0 = p0
        self.loc_scale = loc_scale
        if self.loc_scale:
            self.ffwd = nn.Sequential(nn.Linear(n_hidden, n_hidden//2),
                                      nn.ReLU(),
                                      nn.Linear(n_hidden//2, 2))
        else:
            self.ffwd = nn.Sequential(nn.Linear(n_hidden, (n_hidden+256)//2),
                                      nn.ReLU(),
                                      nn.Linear((n_hidden+256)//2, 256))

        self.loss = nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def logits_from_loc_scale(z):
        mu = torch.tanh(z[..., 0:1])
        rho = softplus(z[..., 1:2])

        logits = torch.linspace(-1, 1, 256, device=z.device, dtype=z.dtype).repeat(mu.shape)
        logits = -(logits-mu)**2 / rho
        return logits

    def forward(self, x, y=None):
        p0 = self.p0
        if y is None:
            logits = self.ffwd(x)
            if self.loc_scale:
                logits = self.logits_from_loc_scale(logits)
            logits = torch.log((1.0-p0)*logits.softmax(dim=2) + p0/256.0)
            dist = torch.distributions.Categorical(logits=logits)
            y = dist.sample()
            return -dist.log_prob(y), y
        else:
            logits = self.ffwd(x)
            if self.loc_scale:
                logits = self.logits_from_loc_scale(logits)
            logits = logits.transpose(1, 2)
            logits = torch.log((1.0-p0)*logits.softmax(dim=1) + p0/256.0)
            return self.loss(logits, y)

class ResLSTM(nn.Module):
    def __init__(self, n_units, n_layers, batch_first):
        super(ResLSTM, self).__init__()
        assert len(n_units) == n_layers+1
        self.n_units = n_units
        self.n_layers = n_layers
        self.lstm = nn.ModuleList([LayerNormLSTM(n_units[i], n_units[i+1],
                                                 batch_first=batch_first,
                                                 forget_bias=1.0,
                                                 dropout=0.0,
                                                 zoneout=0.0)
                                   for i in range(self.n_layers)])

    def forward(self, x, hc=None):

        hc_out = [None]*self.n_layers
        z = x
        for i in range(self.n_layers):
            z, hc_out[i] = self.lstm[i](x, None if hc is None else hc[i])
            # if i > 0 or self.n_in == self.n_hidden:
            #     z = z + x
            x = z
        return z, hc_out


class GenCifar(pl.LightningModule):
    batch_size: int = 32
    sample_negatives = 0
    n_units: [int] = [16, 32, 64, 128, 256]
    n_hidden: int = 256
    n_layers: int = 4
    n_chunks: int = 5
    p0: float = 5.0/3072
    lr: float = 1e-3

    transform = ToByteTensor()

    def __init__(self):
        super().__init__()

        # self.inp = nn.Sequential(nn.Embedding(256, self.n_hidden),
        #                          nn.Linear(self.n_hidden, self.n_hidden),
        #                          nn.ReLU())
        self.inp = Input(self.n_units[0])
        # self.rnn = nn.LSTM(self.n_hidden, self.n_hidden, num_layers=self.n_layers, batch_first=True)
        self.rnn = ResLSTM(self.n_units, n_layers=self.n_layers, batch_first=True)

        self.out = OutputModule(self.n_hidden, self.p0)

    def forward(self, batch, sample_negatives=0):
        y, _ = batch
        N, C, H, W = y.shape

        if sample_negatives:
            y_samp = self.sample(sample_negatives)
            loss_weights = torch.ones(N+sample_negatives, device=y.device, dtype=torch.float)
            loss_weights[N:] = -1.0
            loss_weights = loss_weights/N
            N = N+sample_negatives
            y = torch.cat([y, y_samp], 0)
        else:
            loss_weights = None

        y = y.permute(0, 2, 3, 1).reshape(N, H * W * C)
        x = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)
        cs = self.inp.coord_seq(device=x.device).expand(N, H*W*C, 3)
        x = torch.cat([x.unsqueeze(2), cs], 2)

        loss = chunked_rnn(self.inp, self.rnn, self.out, x, y, None, self.n_chunks,
                           loss_weights=loss_weights)
        return loss

    def sample(self, N, sample_shape=(3, 32, 32)):
        C, H, W = sample_shape
        x = torch.zeros(N, C * H * W, 1, dtype=torch.long, device=self.device)
        cs = self.inp.coord_seq(device=self.device).expand(N, H * W * C, 3)
        hc = None
        with torch.no_grad():
            for t in range(C * H * W):
                tmp = self.inp(torch.cat([x[:, t - 1:t] if t > 0 else x[:, :1], cs[:, t:t + 1]], dim=2))
                tmp, hc = self.rnn(tmp, hc)
                _, x[:, t:t + 1, 0] = self.out(tmp)

        x = x[..., 0].reshape(N, H, W, C).permute(0, 3, 1, 2)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, self.sample_negatives)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('val_loss', loss)
        return loss

    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int, *args, ** kwargs) -> None:
        pass

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, optimizer_idx: int):
        pass

    def train_dataloader(self) -> DataLoader:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=8)
        return trainloader

    def val_dataloader(self) -> DataLoader:
        valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=self.transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=8)
        return valloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        #return [optimizer],[scheduler]

    def log_samples(self):
        with torch.no_grad():
            x = self.sample(20).cpu().numpy()
        fig = figure(figsize=(6, 5))
        imshow(x.transpose(0, 2, 3, 1)
               .reshape(4, 5, 32, 32, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(4*32, 5*32, 3))
        axis('off')
        return fig

    def log_coord_sequences(self):
        shapes = {256: (16, 16),
                  16: (4, 4)}

        with torch.no_grad():
            cs = self.inp.coord_seq(device=self.device)
            y = self.inp.emb_c(cs[..., 0]) + self.inp.emb_h(cs[..., 1]) + self.inp.emb_w(cs[..., 2])
            n1,n2 = shapes[y.shape[-1]]
            y = y.reshape(32, 32, 3, n1, n2).permute(3, 0, 4, 1, 2).reshape(n1 * 32, n2 * 32, 3)
            y_range = torch.quantile(y.view(-1), torch.tensor([0.05, 0.95], device=self.device), 0)
            y = (y - y_range[0]) / (y_range[1] - y_range[0])
            y = y.clip(0.0, 1.0)
        fig = figure(figsize=(10, 10))
        imshow(y.cpu().numpy())
        axis('off')
        return fig


def tracked_gradient_global_only():
    def remove_per_weight_norms(func):
        def f(*args):
            norms = func(*args)
            norms = dict(filter(lambda elem: '_total' in elem[0], norms.items()))
            return norms
        return f
    pl.core.grads.GradInformation.grad_norm = remove_per_weight_norms(pl.core.grads.GradInformation.grad_norm)


class LoggingCallback(pl.Callback):

    def on_epoch_start(self, trainer, model):
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            fig = model.log_samples()
            logger.experiment.log({'samples': wandb.Image(fig)}, commit=False)
            del fig
            fig = model.log_coord_sequences()
            logger.experiment.log({'coords': wandb.Image(fig)}, commit=False)
            del fig


if __name__ == '__main__':

    tracked_gradient_global_only()
    model = GenCifar()
    checkpoint = "wandb/run-20210506_182336-dz7x1ica/files/cifar-gen/dz7x1ica/checkpoints/epoch=112-step=176618.ckpt"
    model.load_from_checkpoint(checkpoint)
    logger = WandbLogger(project='cifar-gen')
    log_dir = logger.experiment.dir

    if rank_zero_only.rank == 0:
        for fn in ['gen_cifar.py', 'traces.py', '../chunked_rnn.py', '../utilities.py']:
            filename = os.path.basename(fn)
            copyfile(fn, os.path.join(log_dir, filename))

    logging_cb = LoggingCallback()

    traces = Trace(13, None, columns=8,
                   log_buffers=False  # to not log all the batch norm running means and variances
                   )

    trainer = pl.Trainer(max_epochs=1000,
                         gpus=1,
                         # accelerator="ddp",
                         # automatic_optimization=False,
                         logger=logger,
                         callbacks=[traces, logging_cb],
                         gradient_clip_val=8,
                         track_grad_norm=2,
                         resume_from_checkpoint=checkpoint
                         )

    trainer.fit(model)
