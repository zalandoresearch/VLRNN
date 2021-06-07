from collections import namedtuple
import sys
from typing import NamedTuple
sys.path.append(".")


import chunked_rnn

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence


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


def test_lens_of_packed_sequence():
    num_batch = 10
    max_seq_len = 100
    min_seq_len = 1
    num_channels = ()
    lens = torch.randint(min_seq_len, max_seq_len+1, (num_batch,)) 
    x = pack_sequence([torch.randn(l,) for l in lens], enforce_sorted=False)

    assert torch.equal(chunked_rnn.lens_of_packed_sequence(x), lens)



# testing chunked_rnn with non-packed sequences
Globals = namedtuple("globals", "n_batch n_x n_in n_hidden n_out n_seq dtype var")

@pytest.fixture(scope="module", params=[True, False], ids=["PACKED", "FIXED"])
def globals(request):
    return Globals(
        n_batch = 8,
        n_x = 2,
        n_in = 18,
        n_hidden = 32,
        n_out = 5,
        n_seq = 100,
        dtype = torch.double,
        var = request.param
    )

# input model
class Inp(nn.Sequential):
    def __init__(self, n_in):
        super().__init__(nn.Linear(n_in ,10), nn.ELU(), nn.Linear(10 ,n_in), nn.ELU())

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super().forward(x)
        else:
            return PackedSequence(data=super().forward(x.data), batch_sizes=x.batch_sizes, 
                                  sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)

@pytest.fixture(scope="module")
def inp(globals):
    return Inp(globals.n_in).to(globals.dtype)


# output model
class Outp(nn.Sequential):

    def __init__(sel, n_hidden, n_out):
        super().__init__( nn.Linear(n_hidden ,10) ,nn.ELU() ,nn.Linear(10 ,n_out))

    def _forward(self, z, y):
        logits = super().forward(z)
        return -torch.distributions.Categorical(logits=logits).log_prob(y)

    def forward(self, z, y):
        if isinstance(z, torch.Tensor):
            return self._forward(z,y)
        else:
            l = PackedSequence(data=self._forward(z.data, y.data), batch_sizes=z.batch_sizes, 
                               sorted_indices=z.sorted_indices, unsorted_indices=z.unsorted_indices)    
            return l

@pytest.fixture(scope="module")
def outp(globals):
    return Outp(globals.n_hidden, globals.n_out).to(globals.dtype)


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



@pytest.fixture(scope="module", params=['GRU','LSTM','FANCY'])
def rnn(request,globals):
    if request.param=='GRU':
        return nn.GRU(globals.n_in, globals.n_hidden, batch_first=True).to(globals.dtype)
    if request.param=='LSTM':
        return nn.LSTM(globals.n_in, globals.n_hidden, batch_first=True, num_layers=3).to(globals.dtype)
    if request.param=='FANCY':
        return FancyRNN(globals.n_in, globals.n_hidden).to(globals.dtype)




def create_sequences(globals):

    if globals.var:
        lens = torch.randint(globals.n_seq//5, globals.n_seq+1, (globals.n_batch,)) 
        x = pack_sequence([torch.randn(l, globals.n_in, dtype=globals.dtype) for l in lens], enforce_sorted=False)
        y = pack_sequence([torch.randint(0, globals.n_out, (l,)) for l in lens], enforce_sorted=False)
       
    else:
        x = torch.randn(globals.n_batch, globals.n_seq, globals.n_in, dtype=globals.dtype)
        y = torch.randint(0, globals.n_out, (globals.n_batch, globals.n_seq))

    return x,y    




@pytest.mark.parametrize("loss_scale",["sum","mean"])
@pytest.mark.parametrize("n",[1,2,7,10])
def test_chunked_rnn( globals, inp, outp, rnn, n, loss_scale):
        #print("{} chunks".format(n))
        mods = nn.ModuleList([inp, outp, rnn])

        x,y = create_sequences(globals)

        mods.zero_grad()
        x_ = inp(x)
        z, h = rnn(x_, None)

        l_std = outp(z, y)
        if globals.var:
            l_std, lens = pad_packed_sequence(l_std, batch_first=True) 
            l_std = l_std.sum(1)
            if loss_scale == "mean":
                l_std /= lens
        else:
            l_std = l_std.mean(1) if loss_scale == "mean" else l_std.sum(1)
        l_std = l_std.mean()

        l_std.backward()
        l_std = l_std.item()
        g_std = [p.grad.clone() for p in mods.parameters()]
        print("loss (standard computation) {:.6f}".format(l_std))

        mods.zero_grad()
        l_chunk = chunked_rnn.chunked_rnn(inp, rnn, outp, x, y, None, n, loss_scale=loss_scale)
        g_chunk = [p.grad.clone() for p in mods.parameters()]
        print("loss (chunked computation) {:.6f}".format(l_chunk))

        assert all([torch.allclose(g1, g2) for g1, g2 in zip(g_chunk, g_std )])
        #print("losses and gradients equal:", good)
        #print()


