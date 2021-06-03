import sys
sys.path.append(".")


import chunked_rnn

import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence


import pytest

def valid_packed_sequence(x: PackedSequence) -> bool :

    assert isinstance(x, PackedSequence)

    # all batch sizes must sum up to the length of all sequences
    assert x.data.shape[0] == x.batch_sizes.sum()

    # batch sizes must be in descending order
    assert all([x.batch_sizes[i]>=x.batch_sizes[i+1] for i in range(len(x.batch_sizes)-1)])


def _test_chunk_packed_sequence( num_batch = 10,
                                max_seq_len = 100,
                                min_seq_len = 1,
                                num_channels = (3,),
                                num_chunks = 8 ):

    lens = torch.randint(min_seq_len, max_seq_len+1, (num_batch,)) 
    x = pack_sequence([torch.randn(l, *num_channels) for l in lens], enforce_sorted=False)
    y = pack_sequence([torch.randn(l, *num_channels) for l in lens], enforce_sorted=False)



    x_chunk, y_chunk = chunked_rnn.chunk_packed_sequence(x, y, num_chunks)

    for x_ in x_chunk + y_chunk:
        valid_packed_sequence(x_)



def test_chunk_packed_sequence():
    for num_channels in [(), (3,), (4,5)]:
        for num_chunks in [1, 5, 10, 100]:
            _test_chunk_packed_sequence( num_channels=num_channels, num_chunks=num_chunks)

