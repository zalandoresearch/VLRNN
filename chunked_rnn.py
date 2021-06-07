from typing import List
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from utilities import struct_flatten, requires_grad, grad_of


def chunk_packed_sequence( x: PackedSequence, N) -> List[PackedSequence]:
    """chunk a packed sequence into approximately N axpprimately equally long packed sequences

    Args:
        x (PackedSequence): input sequence as returned from torch.nn.utils.rnn.pack_sequence
        N (int): number of output sequences

    Returns:
        list of packed sequence chunks, each being a valid packed sequence
    """

    M = x.batch_sizes.sum()

    x_out = []
    def append(last_i, i, last_j, j):
            #print(last_i,i, last_j, j)
            idxs = torch.arange(x.batch_sizes[last_i])
            x_out.append( PackedSequence(
                data = x.data[last_j:j],
                batch_sizes = x.batch_sizes[last_i:i],
                sorted_indices = idxs,
                unsorted_indices = idxs
            ))

    last_i = 0 # index into batch_sizes
    last_j = 0 # index into data
    j = 0
    for i in range(x.batch_sizes.shape[0]):
        if M<=0:
            append(last_i, i, last_j, j)
            M = M + x.batch_sizes.sum()
            last_i = i
            last_j = j
        else:
            M = M - N*x.batch_sizes[i]
        j = j + x.batch_sizes[i]
    append(last_i, x.batch_sizes.shape[0], last_j, j)

    return x_out


def combine_packed_sequence(x_chunk, sorted_indices=None, unsorted_indices=None)-> PackedSequence:
    """combine chunked packed sequences 

    Args:
        x_chunk (List[PackedSequence]): list of chunked packed sequences as returned from chunk_packed_sequence
        sorted_indices (torch.Tensor, optional): optional premutation ordder for the resulting packed sequence. Defaults to None.
        unsorted_indices (torch.Tensor, optional): optional inversepremutation ordder for the resulting packed sequence. Defaults to None.

    Returns:
        PackedSequence: resulting packed sequence
    """
         
    N = x_chunk[0].batch_sizes[0]
    x_data = torch.cat([x_.data for x_ in x_chunk], dim=0)
    x_batch_sizes = torch.cat([x_.batch_sizes for x_ in x_chunk], dim=0) 

    if sorted_indices is None:
        assert unsorted_indices is None
        x_sorted_indices = torch.arange(N)
        x_unsorted_indices = torch.arange(N)
    else:
        assert unsorted_indices is not None
        x_sorted_indices = sorted_indices
        x_unsorted_indices = unsorted_indices

    return PackedSequence( data = x_data,
                            batch_sizes=x_batch_sizes,
                            sorted_indices=x_sorted_indices,
                            unsorted_indices=x_unsorted_indices)
    



def lens_of_packed_sequence(x: PackedSequence) -> torch.Tensor:
    n_batch = x.batch_sizes[0]
    lens = (x.batch_sizes.unsqueeze(0)>torch.arange(n_batch).unsqueeze(1)).sum(1)
    return lens[x.unsorted_indices]



def chunk_lengths(chunks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    cum_chunks = chunks.cumsum(0)[:-1]
    lengths = lengths.view(-1, 1)
    lengths = torch.cat([lengths, lengths - cum_chunks.view(1, -1)], 1)
    lengths = torch.min(torch.max(lengths, torch.zeros_like(lengths)), chunks.view(1, -1))
    return lengths


def chunked_rnn(inp, rnn, outp, x, y, h0, N, lengths=None, loss_scale="mean"):
    # channels last:
    # x is (n_batch, n_seq, n_channels)

    assert loss_scale in ["mean","sum"]

    inp.zero_grad()
    rnn.zero_grad()
    outp.zero_grad()

    packed_seq = (isinstance(x, PackedSequence) and isinstance(y, PackedSequence))

    if packed_seq:
        N_total = len(x.batch_sizes)
        n_batch = x.batch_sizes[0]
        x_chunks = chunk_packed_sequence(x, N)
        y_chunks = chunk_packed_sequence(y, N)
        if loss_scale=="mean":
            lens = lens_of_packed_sequence(x)
            lens_chunks = chunk_packed_sequence(pack_sequence([torch.ones(l)*l for l in lens], enforce_sorted=False), N)

        assert len(x_chunks) == len(y_chunks)
    else:
        n_batch, N_total = x.shape[:2]
        x_chunks = x.chunk(N, dim=1)
        y_chunks = y.chunk(N, dim=1)

    N = len(x_chunks) # may be different that requested for packed sequences

    h_chunks = [h0] + [None] * N
    with torch.no_grad():
        for n in range(N):
            x_n = x_chunks[n]
            h_n = h_chunks[n]

            z, h = rnn(inp(x_n), h_n)
            h_chunks[n + 1] = h

    #$# loss_chunks = [None] * N

    loss = 0
    delta_h_n_plus_1 = None
    for n in reversed(range(N)):

        x_n = x_chunks[n]
        y_n = y_chunks[n]
        h_n = h_chunks[n]

        if torch.is_grad_enabled():
            if h_n is not None:
                # h_n.requires_grad = True
                requires_grad(h_n, True)

        z_n, h_n_plus_1 = rnn(inp(x_n), h_n)
        loss_n = outp(z_n, y_n)

        if packed_seq:
            if loss_scale == "mean":
                loss_n = (loss_n.data/lens_chunks[n].data).sum(0)/n_batch
            else:
                loss_n = loss_n.data.sum(0)/n_batch # packed sequences are (total) sequences first dim
        else:
            #$# loss_chunks[n] = loss_n.detach()
            loss_n = loss_n.sum(1).mean(0)
            if loss_scale == "mean":
                loss_n = loss_n / N_total


        # for a, b in zip(struct_flatten(h_n_plus_1), struct_flatten(h_chunks[n + 1])):
        #     assert torch.allclose(a, b)

        if torch.is_grad_enabled():
            if n < N - 1:
                loss_n.backward(retain_graph=True)
                h_n_plus_1_ = list(struct_flatten(h_n_plus_1))
                delta_h_n_plus_1_ = list(struct_flatten(delta_h_n_plus_1))

                torch.autograd.backward(h_n_plus_1_, grad_tensors=delta_h_n_plus_1_)
            else:
                loss_n.backward()

            if h_n is not None:
                delta_h_n_plus_1 = grad_of(h_n)
            else:
                delta_h_n_plus_1 = None

        loss = loss + loss_n.detach() #.item()

    return loss #$#, torch.cat(loss_chunks, 1)