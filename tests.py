from typing import Any

import torch
import torch.nn as nn

from chunked_rnn import chunked_rnn

n_batch = 8
n_x = 2
n_in = 18
n_hidden = 32
n_out = 5

n_seq = 100

# input model
inp = nn.Sequential( nn.Linear(n_in ,10), nn.ELU(), nn.Linear(10 ,n_in), nn.ELU()).to(torch.double)


# output model
class Outp(nn.Sequential):

    def __init__(self):
        super().__init__( nn.Linear(n_hidden ,10) ,nn.ELU() ,nn.Linear(10 ,n_out))

    def forward(self, z, y):
        logits = super().forward(z)
        return -torch.distributions.Categorical(logits=logits).log_prob(y)


outp = Outp().to(torch.double)


# rnn models
class FancyRNN(nn.Module):

    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.gru = nn.GRU(n_in, 2* n_hidden, batch_first=True)
        self.lstm = nn.LSTM(2 * n_hidden, n_hidden, batch_first=True, num_layers=3)

    def forward(self, x, h):
        tmp, h_gru = self.gru(x, None if h is None else h[0])
        z, h_lstm = self.lstm(tmp, None if h is None else h[1])

        return z, (h_gru, h_lstm)


rnn1 = nn.GRU(n_in, n_hidden, batch_first=True).to(torch.double)
rnn2 = nn.LSTM(n_in, n_hidden, batch_first=True, num_layers=3).to(torch.double)
rnn3 = FancyRNN(n_in, n_hidden).to(torch.double)

all_good = True

for rnn in [rnn1, rnn2, rnn3]:
    for n in [1,2,7,10]:
        print("{} chunks".format(n))
        mods = nn.ModuleList([inp, outp, rnn])

        x = torch.randn(n_batch, n_seq, n_in, dtype=torch.double)
        y = torch.randint(0, n_out, (n_batch, n_seq))

        mods.zero_grad()
        z, h = rnn(inp(x), None)
        l_std = outp(z, y).sum(1).mean()
        l_std.backward()
        l_std = l_std.item()
        g_std = [p.grad.clone() for p in mods.parameters()]
        print("loss (standard computation) {:.6f}".format(l_std))

        mods.zero_grad()
        l_chunk = chunked_rnn(inp, rnn, outp, x, y, None, n)
        g_chunk = [p.grad.clone() for p in mods.parameters()]
        print("loss (chunked computation) {:.6f}".format(l_chunk))

        good = all([torch.allclose(g1, g2) for g1, g2 in zip(g_std, g_chunk)])
        print("losses and gradients equal:", good)
        print()

        all_good = all_good and good

print("All tests passed." if good else "SOME TESTS FAILED!")