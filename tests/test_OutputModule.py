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
    
    return  z1, z2, z3, z_short






def test_forward0(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, *args, **kwargs):
            return torch.randn(*z1.shape[:2])

    with pytest.raises(TypeError):
        Out()( z1) # forward() cannot be declared with *args


def test_forward1(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])

    with pytest.raises(TypeError):
        Out()( z1) # forward() must be called with argument z2, which is missing


def test_forward2(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])

    Out()( z1, z2, z3) # good


def test_forward3(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])

    with pytest.raises(TypeError):
        Out()( z1, None, z3) # forward() must be called with argument z1, which is missing (arg is None but not optional)


def test_forward4(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])

    Out()( z1=z1, z2=z3) # good


def test_forward5(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])
    
    with pytest.raises(ValueError):
        Out()( z1, z2, z_short) # argument z3 of forward() must be (8,100,...) sequence Tensor, but had shape torch.Size([8, 98, 10])


def test_forward6(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None):
            return torch.randn(*z1.shape[:2])
    
    with pytest.raises(ValueError):
        Out()( z1, z2, z_short) # argument z3 of forward() must be (8,100,...) sequence Tensor, but had shape torch.Size([8, 98, 10])


def test_forward7(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None, mode='Train'): # bad, mode is interpreted as sequence
            return torch.randn(*z1.shape[:2])
    
    with pytest.raises(TypeError):
        Out()( z1, z2, z3, 'eval') # forward() must be called with argument mode: (<class 'torch.Tensor'>, <class 'NoneType'>), was called with eval

def test_forward8(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None, **kwargs): 
            mode = kwargs['mode']
            return torch.randn(*z1.shape[:2])
    
    Out()( z1, z2, z3, mode='eval') # good
    

def test_forward9(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None): 
            return torch.randn(*z1.shape[:2]), z3 # returning two sequences is fine
    
    Out()( z1, z2, z3) # good
    Out()( z1, z2) # also good, second returned argument is None
    
    
def test_forward10(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None): 
            return None, torch.randn(*z1.shape[:2]) # bad, must at least return loss sequence
    
    with pytest.raises(TypeError):
        Out()( z1, z2) # TypeError: forward() must return loss, which is missing
    
    
def test_forward10(sequences):
    z1, z2, z3, z_short = sequences

    class Out(OutputModule):
        def forward(self, z1, z2, z3=None): 
            return z_short # bad, returned loss sequence has wrong shape
    
    with pytest.raises(ValueError):
        Out()( z1, z2) # TypeError: forward() must return loss, which is missing
    
    