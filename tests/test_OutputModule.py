from collections import namedtuple
import torch
from torch.nn.utils.rnn import pack_sequence

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
    zp1 = pack_sequence([torch.randn(l, 10) for l in lengths], enforce_sorted=False)
    zp2 = pack_sequence([torch.randn(l,) for l in lengths], enforce_sorted=False)
    zp3 = pack_sequence([torch.randint(0, 15, (l,)) for l in lengths], enforce_sorted=False)
       
    z1 = torch.randint(0, 15, (globals.n_batch, globals.n_seq) )
    z2 = torch.randint(0, 15, (globals.n_batch, globals.n_seq) )
    z3 = torch.randn(globals.n_batch, globals.n_seq, 10)

    z_short = torch.randn(globals.n_batch, globals.n_seq-2, 10)
    
    return  z1, z2, z3, zp1, z_short






def test_forward0(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, *args, **kwargs):
            return torch.randn(*z1.shape[:2])

    with pytest.raises(TypeError):
        Out()( z1) # forward() must be called with two arguments (and optional kwargs), but had 1 fixed arguments
    with pytest.raises(TypeError):
        Out()( z1, z3) # forward() must be called with two arguments (and optional kwargs), but had 1 fixed arguments


def test_forward1(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, y, z3=None):
            return torch.randn(*z1.shape[:2])

    with pytest.raises(TypeError):
        Out()( z1) # forward() must be called with argument z2, which is missing


def test_forward2(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z, y=None, z3=None):
            return torch.randn(*z.shape[:2])

    Out()( z1, z2, z3=z3) # ok, z3 is extra arg
    Out()( z1, z3=z3) # ok, y is default None

    with pytest.raises(TypeError):
        Out() (zp1) # no packed sequences in OutputModule


def test_forward3(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z, y, z3=None):
            return torch.randn(*z.shape[:2])

    Out()( z1, None, z3) # ok, y can be None
    with pytest.raises(TypeError):
        Out()( None, z1, z3) # forward() must be called with argument z: (<class 'torch.Tensor'>, <class 'torch.nn.utils.rnn.PackedSequence'>, <class 'tuple'>), was called with None


def test_forward4(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])

    Out()( z1=z1, z2=z3) # good


def test_forward5(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])
    
    Out()( z1, z2, z_short) # ok, z_short is extra arg


def test_forward6(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z, y, z3=None):
            return torch.randn(*z[0].shape[:2])
    
    with pytest.raises(ValueError):
        Out()( (z1, z2), z_short) # argument z3 of forward() must be (8,100,...) sequence Tensor, but had shape torch.Size([8, 98, 10])

    Out()( (z1, z2), z3) # ok


def test_forward7(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None, mode='Train'): # 
            return torch.randn(*z1.shape[:2])
    
    Out()( z1, z2, z3, 'eval') # z3 and 'eval' are interpreted as extra paremeters
    Out()( z1, z2, z3=z3, mode='eval') # z3 and 'eval' are interpreted as extra paremeters


def test_forward8(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None, **kwargs): 
            mode = kwargs['mode']
            return torch.randn(*z1.shape[:2])
    
    Out()( z1, z2, z3, mode='eval') # good
    

def test_forward9(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None): 
            return torch.randn(*z1.shape[:2]), z3 # returning an extra sequence is fine
    
    Out()( z1, z2, z3) # good
    Out()( z1, z2) # also good, second returned argument is None
    
    
def test_forward10(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None): 
            return None, torch.randn(*z1.shape[:2]) # bad, must at least return loss sequence
    
    with pytest.raises(TypeError):
        Out()( z1, z2) # TypeError: forward() must return loss, which is missing
    
    
def test_forward10(sequences):
    z1, z2, z3, zp1, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None): 
            return z_short # bad, returned loss sequence has wrong shape
    
    with pytest.raises(ValueError):
        Out()( z1, z2) # TypeError: forward() must return loss, which is missing
    
    