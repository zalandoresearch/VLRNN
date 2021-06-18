from collections import namedtuple
from typing import NamedTuple

import sys
sys.path.append(".")
from vlrnn import BlockRNN, breakup_packed_sequence, combine_packed_sequence
from tests.test_utilities import equal_packed_sequences


import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence


import pytest


@pytest.fixture(scope='module', params=[
    torch.device('cpu'), 
    pytest.param(torch.device('cuda'), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))], 
    ids=['CPU','CUDA'])
def device(request):
    return request.param


# testing chunked_rnn with non-packed sequences
Globals = namedtuple("globals", "n_batch n_x n_in n_hidden n_out n_seq dtype device var")

@pytest.fixture(scope="module", params=[True, False], ids=["PACKED", "FIXED"])
def globals(request, device):
    return Globals(
        n_batch = 8,
        n_x = 2,
        n_in = 18,
        n_hidden = 32,
        n_out = 5,
        n_seq = 100,
        dtype = torch.double,
        device = device,
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
    return Inp(globals.n_in).to(globals.device, dtype=globals.dtype)


# output model
class Outp(nn.Sequential):

    def __init__(sel, n_hidden, n_out):
        super().__init__( nn.Linear(n_hidden ,10) ,nn.ELU() ,nn.Linear(10 ,n_out))

    def _forward(self, z, y, return_y):
        logits = super().forward(z)
        loss =  -torch.distributions.Categorical(logits=logits).log_prob(y)
        if return_y:
            return loss, y
        else:
            return loss

    def forward(self, z, y, return_y=False):

        
        if isinstance(z, torch.Tensor):
            return self._forward(z, y, return_y)
        else:
            res = self._forward(z.data, y.data, return_y)
            if return_y:
                res = ( PackedSequence(data=res[0], batch_sizes=z.batch_sizes, 
                               sorted_indices=z.sorted_indices, unsorted_indices=z.unsorted_indices),
                        PackedSequence(data=res[1], batch_sizes=z.batch_sizes, 
                               sorted_indices=z.sorted_indices, unsorted_indices=z.unsorted_indices) )
            else:
                res = PackedSequence(data=res, batch_sizes=z.batch_sizes, 
                               sorted_indices=z.sorted_indices, unsorted_indices=z.unsorted_indices)
            return res

@pytest.fixture(scope="module")
def outp(globals):
    return Outp(globals.n_hidden, globals.n_out).to(globals.device, dtype=globals.dtype)


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
        return nn.GRU(globals.n_in, globals.n_hidden, batch_first=True).to(globals.device, dtype=globals.dtype)
    if request.param=='LSTM':
        return nn.LSTM(globals.n_in, globals.n_hidden, batch_first=True, num_layers=3).to(globals.device, dtype=globals.dtype)
    if request.param=='FANCY':
        return FancyRNN(globals.n_in, globals.n_hidden).to(globals.device, dtype=globals.dtype)




def create_sequences(globals):

    if globals.var:
        lengths = torch.randint(globals.n_seq//5, globals.n_seq+1, (globals.n_batch,)) 
        x = pack_sequence([torch.randn(l, globals.n_in, dtype=globals.dtype, device=globals.device) for l in lengths], enforce_sorted=False)
        y = pack_sequence([torch.randint(0, globals.n_out, (l,), device=globals.device) for l in lengths], enforce_sorted=False)
       
    else:
        x = torch.randn(globals.n_batch, globals.n_seq, globals.n_in, dtype=globals.dtype, device=globals.device)
        y = torch.randint(0, globals.n_out, (globals.n_batch, globals.n_seq), device=globals.device)

    return x,y    

def equal_sequence(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return torch.equal(x,y)
    elif isinstance(x, PackedSequence) and isinstance(y, PackedSequence):
        return equal_packed_sequences(x, y)
    else:
        return False

@pytest.mark.parametrize("loss_scale",["SUM","MEAN"])
@pytest.mark.parametrize("N",[1,2,7,10])
def test_BlockRNN( globals, outp, rnn, N, loss_scale):

        mods = nn.ModuleList([outp, rnn])
        x,y = create_sequences(globals)
 
        #################################################
        # running as plain RNN
        #################################################
        
        mods.zero_grad()
        z, h = rnn(x, None)
        l_std = outp(z, y)
        if globals.var: #'PACKED'
            l_std, lens = pad_packed_sequence(l_std, batch_first=True) 
            l_std = l_std.sum(1)
            if loss_scale == "MEAN":
                l_std /= lens.to(globals.device) 
        l_std = l_std.mean() if loss_scale == "MEAN" else l_std.sum()
        print(l_std)
    
        l_std.backward()
        l_std = l_std.item()
        g_std = [p.grad.clone() for p in mods.parameters()]
        print("loss (standard computation) {:.6f}".format(l_std))


        #################################################
        # running as BlockRNN
        #################################################

        mods.zero_grad()
        vlrnn = BlockRNN(rnn, outp, loss_scale)
        l_chunk = vlrnn(x, None, y,  N)
        g_chunk = [p.grad.clone() for p in mods.parameters()]
        print("loss (chunked computation) {:.6f}".format(l_chunk))
        l_chunk = l_chunk.item()

        assert all([torch.allclose(g1, g2) for g1, g2 in zip(g_chunk, g_std )])
        #print("losses and gradients equal:", good)
        #print()

        l_chunk, y_ = vlrnn(x, None, y,  N, return_y=True)
        equal_sequence(y, y_)

        l_chunk, y_, h = vlrnn(x, None, y,  N, return_y=True, return_h='last')
        l_chunk, h = vlrnn(x, None, y,  N, return_h='last')
        