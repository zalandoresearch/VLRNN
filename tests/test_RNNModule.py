from collections import namedtuple
from vlrnn.block_rnn import RNNModule
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

import sys
sys.path.append(".")
from vlrnn import OutputModule



import pytest



# testing chunked_rnn with non-packed sequences
Globals = namedtuple("globals", "n_batch n_seq")

@pytest.fixture
def globals():
    return Globals(
        n_batch = 8,
        n_seq = 100,
    )

@pytest.fixture
def sequences(globals):

    lengths = torch.randint(globals.n_seq//5, globals.n_seq+1, (globals.n_batch,)) 
    xp1 = pack_sequence([torch.randn(l, 10) for l in lengths], enforce_sorted=False)
    xp2 = pack_sequence([torch.randn(l,) for l in lengths], enforce_sorted=False)
    xp3 = pack_sequence([torch.randint(0, 15, (l,)) for l in lengths], enforce_sorted=False)
       
    x1 = torch.randint(0, 15, (globals.n_batch, globals.n_seq) )
    x2 = torch.randint(0, 15, (globals.n_batch, globals.n_seq) )
    x3 = torch.randn(globals.n_batch, globals.n_seq, 10)

    x_short = torch.randn(globals.n_batch, globals.n_seq-2, 10)

    h1 = torch.randn(globals.n_batch)
    h2 = torch.randint(0,5,(globals.n_batch, 1))
    h_bad = torch.randn(globals.n_batch+1, globals.n_seq)
    return  xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad



def test_forward0(sequences):
    xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad = sequences

    class Rnn(RNNModule):
        def forward(self, x1, *args, h=None):
            return x1, h1
            
    with pytest.raises(TypeError):
        Rnn()( x1) # forward() cannot be declared with *args


def test_forward1(sequences):
    xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad = sequences

    class Rnn(RNNModule):
        def forward(self, x1, x2=None, h=None):
            return x1, h1
            
    Rnn()( x1) 
    Rnn()( x1, x2)
    Rnn()( x1=x2, x2=x1, h=h1)

    Rnn()( xp1) 
    Rnn()( xp1, xp2)
    Rnn()( x1=xp2, x2=xp1, h=h1)


    with pytest.raises(TypeError):
        Rnn()(x1=None, x2=x2) # first arg mut not be None
        
    with pytest.raises(TypeError):
        Rnn()(x1, xp1) # mixed packed and non-packed sequences

    with pytest.raises(ValueError):
        Rnn()(xp1, h=h_bad) # ValueError: hidden state 0 must be batch-first Tensor with batch_size 8, but was shape torch.Size([9, 100])


def test_forward2(sequences):
    xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad = sequences

    class Rnn(RNNModule):
        def forward(self, x1, x2=None, h=None):
            return (x1, None), (h1 if h is None else h, h2)
            
    Rnn()( x1) 
    Rnn()( x1, x2)
    Rnn()( x1=x2, x2=x1, h=h1)

    Rnn()( xp1) 
    Rnn()( xp1, xp2)
    Rnn()( x1=xp2, x2=xp1, h=h1)


def test_forward3(sequences):
    xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad = sequences

    class Rnn(RNNModule):
        def forward(self, x1, x2=None, h=None):
            return (x1, None), h_bad
        
    with pytest.raises(ValueError):
        Rnn()( x1) # hidden state output h[0] must be batch-first Tensor with batch_size 8, but was shape torch.Size([9, 100])

    with pytest.raises(ValueError):
        Rnn()( xp1) # hidden state output h[0] must be batch-first Tensor with batch_size 8, but was shape torch.Size([9, 100])


def test_forward4(sequences):
    xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad = sequences

    class Rnn(RNNModule):
        def forward(self, x1, x2=None, h=None):
            return (xp2, None), h # bad can't return PackedSequence when x was Tensor
            
    with pytest.raises(TypeError):
        Rnn()( x1) # output z[0] of forward() must be a sequence Tensor

    Rnn()( xp1) # good


def test_forward5(sequences):
    xp1, xp2, xp3, x1, x2, x3, x_short, h1, h2, h_bad = sequences

    class Rnn(RNNModule):
        def forward(self, x1, x2=None, h=None):
            x1_, l = pad_packed_sequence(x1, batch_first=True)
            x1 = pack_padded_sequence(x1_.roll(1, dims=0),l.roll(1, dims=0),batch_first=True, enforce_sorted=False) 
            return (xp2, x1), h # bad PackedSequence with wrong lengths
    
    with pytest.raises(ValueError):
        Rnn()( xp1) # output z[1] of forward() must be PackedSequence with sequence lengths ..., but has lengths ...




