from collections import namedtuple
import sys
from typing import NamedTuple
sys.path.append(".")


import chunked_rnn

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence


import pytest

def valid_packed_sequence(x: PackedSequence) -> bool :

    assert isinstance(x, PackedSequence)

    # that whole thing is not empty
    assert x.data.shape[0] > 0

    # all batch sizes must sum up to the length of all sequences
    assert x.data.shape[0] == x.batch_sizes.sum()

    # batch sizes must be in descending order
    assert all([x.batch_sizes[i]>=x.batch_sizes[i+1] for i in range(len(x.batch_sizes)-1)])

    # both index lists have length N
    N = x.batch_sizes.max()
    assert N>0
    rangeN = torch.arange(N)
    assert torch.equal(x.sorted_indices.sort()[0], rangeN)
    assert torch.equal(x.unsorted_indices.sort()[0], rangeN)

    # both index lists are inverse to each other
    assert torch.equal(x.sorted_indices[x.unsorted_indices], rangeN)
    assert torch.equal(x.unsorted_indices[x.sorted_indices], rangeN)


def equal_packed_sequences(x, y):

    assert torch.equal(x.data, y.data)
    assert torch.equal(x.batch_sizes, y.batch_sizes)
    assert torch.equal(x.sorted_indices, y.sorted_indices)
    assert torch.equal(x.unsorted_indices, y.unsorted_indices)


@pytest.mark.parametrize("num_channels",[(), (3,), (4,5)])
@pytest.mark.parametrize("num_chunks",[1,5,10,100])
def test_chunk_packed_sequence(num_channels, num_chunks):
    num_batch = 10
    max_seq_len = 100
    min_seq_len = 1
    #num_channels = (3,),
    #num_chunks = 8 ):

    lens = torch.randint(min_seq_len, max_seq_len+1, (num_batch,)) 
    x = pack_sequence([torch.randn(l, *num_channels) for l in lens], enforce_sorted=False)

    x_chunk = chunked_rnn.chunk_packed_sequence(x, num_chunks)

    for x_ in x_chunk:
        valid_packed_sequence(x_)

    x_restore = chunked_rnn.combine_packed_sequence(x_chunk, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    equal_packed_sequences(x, x_restore)







# testing chunked_rnn with non-packed sequences

Params = namedtuple("Params", "n_batch n_x n_in n_hidden n_out n_seq dtype")

@pytest.fixture
def params():
    return Params(
        n_batch = 8,
        n_x = 2,
        n_in = 18,
        n_hidden = 32,
        n_out = 5,
        n_seq = 100,
        dtype = torch.double
    )

# input model
@pytest.fixture
def inp(params):
    return nn.Sequential( nn.Linear(params.n_in ,10), nn.ELU(), nn.Linear(10 ,params.n_in), nn.ELU()).to(params.dtype)


# output model
class Outp(nn.Sequential):

    def __init__(sel, n_hidden, n_out):
        super().__init__( nn.Linear(n_hidden ,10) ,nn.ELU() ,nn.Linear(10 ,n_out))

    def forward(self, z, y):
        logits = super().forward(z)
        return -torch.distributions.Categorical(logits=logits).log_prob(y)

@pytest.fixture
def outp(params):
    return Outp(params.n_hidden, params.n_out).to(params.dtype)


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


@pytest.fixture
def rnn(params):
    return nn.GRU(params.n_in, params.n_hidden, batch_first=True).to(torch.double)
@pytest.fixture
def rnn2(params):
    return nn.LSTM(params.n_in, params.n_hidden, batch_first=True, num_layers=3).to(torch.double)
@pytest.fixture
def rnn3(params):
    return FancyRNN(params.n_in, params.n_hidden).to(torch.double)


#@pytest.mark.parametrize("rnn",[rnn1,rnn2,rnn3])
@pytest.mark.parametrize("loss_scale",["mean","sum"])
@pytest.mark.parametrize("n",[1,2,7,10])
def test_chunked_rnn(params, inp, outp, rnn, n, loss_scale):
        #print("{} chunks".format(n))
        mods = nn.ModuleList([inp, outp, rnn])

        x = torch.randn(params.n_batch, params.n_seq, params.n_in, dtype=torch.double)
        y = torch.randint(0, params.n_out, (params.n_batch, params.n_seq))

        mods.zero_grad()
        z, h = rnn(inp(x), None)
        if loss_scale == "mean":
            l_std = outp(z, y).mean(1).mean()
        else:
            l_std = outp(z, y).sum(1).mean()

        l_std.backward()
        l_std = l_std.item()
        g_std = [p.grad.clone() for p in mods.parameters()]
        print("loss (standard computation) {:.6f}".format(l_std))

        mods.zero_grad()
        l_chunk = chunked_rnn.chunked_rnn(inp, rnn, outp, x, y, None, n, loss_scale=loss_scale)
        g_chunk = [p.grad.clone() for p in mods.parameters()]
        print("loss (chunked computation) {:.6f}".format(l_chunk))

        assert all([torch.allclose(g1, g2) for g1, g2 in zip(g_std, g_chunk)])
        #print("losses and gradients equal:", good)
        #print()


