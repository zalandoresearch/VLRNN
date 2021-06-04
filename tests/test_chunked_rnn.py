import sys
sys.path.append(".")


import chunked_rnn

import torch
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


@pytest.mark.parametrize("num_channels,num_chunks",[
    ((),    1),
    ((),    5),
    ((),    10),
    ((),    100),
    ((3,),    1),
    ((3,),    5),
    ((3,),    10),
    ((3,),    100),
    ((4,5),    1),
    ((4,5),    5),
    ((4,5),    10),
    ((4,5),    10)
])
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


