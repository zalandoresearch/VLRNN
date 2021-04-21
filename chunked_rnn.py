import torch

from utilities import struct_flatten, requires_grad, grad_of


def chunked_rnn(inp, rnn, outp, x, y, h0, N):
    # channels last:
    # x is (n_batch, n_seq, n_channels)

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

        if h_n is not None:
            # h_n.requires_grad = True
            requires_grad(h_n, True)

        z_n, h_n_plus_1 = rnn(inp(x_n), h_n)
        loss_n = outp(z_n, y_n)

        loss_chunks[n] = loss_n.detach()
        loss_n = loss_n.sum(1).mean(0)

        for a, b in zip(struct_flatten(h_n_plus_1), struct_flatten(h_chunks[n + 1])):
            assert torch.allclose(a, b)

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

        loss = loss + loss_n.item()

    return loss, torch.cat(loss_chunks, 1)