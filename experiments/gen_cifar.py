from typing import Any

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from chunked_rnn import chunked_rnn


class ToByteTensor(transforms.ToTensor):
    def __call__(self, img):
        x = super().__call__(img)
        return (x * 255).long()


class OutputModule(nn.Module):

    def __init__(self, n_hidden):
        super().__init__()
        self.ffwd = nn.Sequential(nn.Linear(n_hidden, 2 * n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(2 * n_hidden, 256))
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        logits = self.ffwd(x).transpose(1, 2)
        return self.loss(logits, y)


class GenCifar(pl.LightningModule):
    batch_size: int = 16
    n_hidden: int = 128
    n_layers: int = 4
    n_chunks: int = 100

    transform = ToByteTensor()

    def __init__(self):
        super().__init__()

        self.inp = nn.Sequential(nn.Embedding(256, self.n_hidden),
                                 nn.Linear(self.n_hidden, self.n_hidden),
                                 nn.ReLU())
        self.rnn = nn.LSTM(self.n_hidden, self.n_hidden, num_layers=self.n_layers, batch_first=True)
        self.out = OutputModule(self.n_hidden)

    def forward(self, batch):
        x, y = batch
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, C * H * W)
        y = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)

        loss = chunked_rnn(self.inp, self.rnn, self.out, x, y, None, self.n_chunks)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('loss', loss)
        return loss

    def train_dataloader(self) -> DataLoader:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)
        return trainloader

    def val_dataloader(self) -> DataLoader:
        valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=self.transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)
        return valloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer],[scheduler]



if __name__ == "__main__":
    model = GenCifar()

    trainer = pl.Trainer(max_epochs=50,
                         gpus=0,
                         automatic_optimization=False,
                         # logger=tb_logger,
                         # callbacks=[traces],
                         # gradient_clip_val=10000,
                         track_grad_norm=2)

    trainer.fit(model)