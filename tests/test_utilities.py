from collections import namedtuple
from typing import NamedTuple

import sys
sys.path.append(".")
from utilities import BreakUp, Combine, breakup_packed_sequence, combine_packed_sequence, struct_equal


import block_rnn

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
def test_breakup_packed_sequence(num_channels, num_chunks):
    num_batch = 10
    max_seq_len = 100
    min_seq_len = 1
    #num_channels = (3,),
    #num_chunks = 8 ):

    lens = torch.randint(min_seq_len, max_seq_len+1, (num_batch,)) 
    x = pack_sequence([torch.randn(l, *num_channels) for l in lens], enforce_sorted=False)

    x_chunk = breakup_packed_sequence(x, num_chunks)

    for x_ in x_chunk:
        valid_packed_sequence(x_)

    x_restore = combine_packed_sequence(x_chunk, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    equal_packed_sequences(x, x_restore)


def test_lens_of_packed_sequence():
    num_batch = 10
    max_seq_len = 100
    min_seq_len = 1
    num_channels = ()
    lens = torch.randint(min_seq_len, max_seq_len+1, (num_batch,)) 
    x = pack_sequence([torch.randn(l,) for l in lens], enforce_sorted=False)

    assert torch.equal(block_rnn.lengths_of_packed_sequence(x), lens)




x = torch.arange(7*10).view(7,10)
y = torch.arange(10*5).view(10,5)
z = torch.randn(1,10)

xp = pack_sequence([torch.arange(10), torch.arange(10), torch.arange(10)])
yp = pack_sequence([torch.arange(8), torch.arange(10), torch.arange(5)], enforce_sorted=False)
zp = pack_sequence([torch.randn(10), torch.randn(8), torch.randn(5)])
    

@pytest.mark.parametrize("a, b, f",[
    (x,                         x,                          True),
    (x,                         y,                          False),
    ([x,(y,z)],                 [x,(y,z)],                  True),
    ([x,(y,z)],                 (x,[y,z]),                  False),
    ([x,{'foo':y, 'bar':z}],    [x,{'foo':y, 'bar':z}],     True),
    ([x,{'foo':y, 'bar':z}],    [x,{'foo':y, 'baz':z}],     False),
    (xp,                         xp,                        True),
    (xp,                         yp,                        False),
    ([xp,(yp,zp)],               [xp,(yp,zp)],              True),
    ([xp,(yp,zp)],               (xp,[yp,zp]),              False),
    ([xp,{'foo':yp, 'bar':zp}],  [xp,{'foo':yp, 'bar':zp}], True),
    ([xp,{'foo':yp, 'bar':zp}],  [xp,{'foo':yp, 'baz':zp}], False),
])
def test_struct_equal(a, b, f):
    assert struct_equal(a, b) == f
    assert struct_equal(b, a) == f




x = torch.arange(7*10).view(7,10)
y = torch.arange(10*10).view(10,10)
z = torch.randn(1,10)
xp = pack_sequence([torch.arange(8), torch.arange(10), torch.arange(5)], enforce_sorted=False)
yp = pack_sequence([torch.arange(8), torch.arange(10), torch.arange(5)], enforce_sorted=False)
zp = pack_sequence([torch.randn(8), torch.randn(10), torch.randn(5)], enforce_sorted=False)


@pytest.mark.parametrize("N",[1,3,10,20])
@pytest.mark.parametrize("a, kwargs",[
    (x,                         {}),    
    ([x,(y,z)],                 {}),    
    ([x,{'foo':y, 'bar':z}],    {}),    
    (xp,                        {'sorted_indices':xp.sorted_indices, 'unsorted_indices':xp.unsorted_indices}),    
    ([xp,(yp,zp)],              {'sorted_indices':xp.sorted_indices, 'unsorted_indices':xp.unsorted_indices}),    
    ([xp,{'foo':yp, 'bar':zp}], {'sorted_indices':xp.sorted_indices, 'unsorted_indices':xp.unsorted_indices}),    
])
def test_breakup_combine(a,N, kwargs):
    b = Combine(**kwargs)(BreakUp(N)(a))
    assert struct_equal(a,b)
