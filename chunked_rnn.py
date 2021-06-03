import torch
from torch.nn.utils.rnn import PackedSequence

from utilities import struct_flatten, requires_grad, grad_of


def chunk_packed_sequence( x: PackedSequence, y: PackedSequence, N) -> PackedSequence:

    assert torch.equal(x.batch_sizes, y.batch_sizes)
    
    M = x.batch_sizes.sum()

    x_out = []
    y_out = []

    def append(last_i, i, last_j, j):
            print(last_i,i, last_j, j)
            x_out.append( PackedSequence(
                data = x.data[last_j:j],
                batch_sizes = x.batch_sizes[last_i:i]
            ))
            y_out.append( PackedSequence(
                data = y.data[last_j:j],
                batch_sizes = y.batch_sizes[last_i:i]
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

    return x_out, y_out



def chunk_lengths(chunks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    cum_chunks = chunks.cumsum(0)[:-1]
    lengths = lengths.view(-1, 1)
    lengths = torch.cat([lengths, lengths - cum_chunks.view(1, -1)], 1)
    lengths = torch.min(torch.max(lengths, torch.zeros_like(lengths)), chunks.view(1, -1))
    return lengths


def chunked_rnn(inp, rnn, outp, x, y, h0, N, lengths=None, loss_weights=None):
    # channels last:
    # x is (n_batch, n_seq, n_channels)

    N_total = x.shape[1]

    inp.zero_grad()
    rnn.zero_grad()
    outp.zero_grad()

    x_chunks = x.chunk(N, dim=1)
    y_chunks = y.chunk(N, dim=1)

    h_chunks = [h0] + [None] * N
    with torch.no_grad():
        for n in range(N):
            x_n = x_chunks[n]
            h_n = h_chunks[n]

            z, h = rnn(inp(x_n), h_n)
            h_chunks[n + 1] = h

    loss_chunks = [None] * N

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
        loss_n = outp(z_n, y_n)/N_total

        loss_chunks[n] = loss_n.detach()
        if loss_weights is None:
            loss_n = loss_n.sum(1).mean(0)
        else:
            loss_n = (loss_n.sum(1)*loss_weights).sum(0)

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

    return loss #, torch.cat(loss_chunks, 1)