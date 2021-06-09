from typing import List
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def add_vector( x: torch.Tensor, v: torch.Tensor, dim: int) -> torch.Tensor:
    sh = [1]*x.ndim
    sh[dim] = len(v)
    return x + v.view(*sh)


def div_vector( x: torch.Tensor, v: torch.Tensor, dim: int) -> torch.Tensor:
    sh = [1]*x.ndim
    sh[dim] = len(v)
    return x / v.view(*sh)


def mul_vector( x: torch.Tensor, v: torch.Tensor, dim: int) -> torch.Tensor:
    sh = [1]*x.ndim
    sh[dim] = len(v)
    return x * v.view(*sh)


def struct_map(f, x):
    if isinstance(x, list):
        return list(map(lambda xi: struct_map(f, xi), x))
    elif isinstance(x, tuple):
        return tuple(map(lambda xi: struct_map(f, xi), x))
    elif x is None:
        return None
    else:
        try:
            return f(x)
        except:
            raise ValueError("cannot map {} over type type {}".format(f, type(x)))


def grad_of(x):
    return struct_map(lambda xi: xi.grad, x)


def requires_grad(x, grad_required):
    if isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            requires_grad(xi, grad_required)
    else:
        if x is not None:
            x.requires_grad = grad_required


def struct_flatten(x):
    if isinstance(x, list) or (isinstance(x, tuple) and not isinstance(x, PackedSequence)):
        for xi in x:
            yield from struct_flatten(xi)
    elif isinstance(x, dict):
        for xi in x.values():
            yield from struct_flatten(xi)
    else:
        if x is not None:
            yield x


def struct_unflatten(x, proto):
    if isinstance(proto, tuple) and not isinstance(proto, PackedSequence):
        return tuple(struct_unflatten(x, p) for p in proto)
    elif isinstance(proto, list):
        return list(struct_unflatten(x, p) for p in proto)
    elif isinstance(proto, dict):
        return dict(zip(proto.keys(), list(struct_unflatten(x, p) for p in proto.values())))
    elif proto is None:
        return None
    else:
        return next(x)


def struct_equal(a,b):

    if type(a) != type(b):
        return False

    if isinstance(a, torch.Tensor):
        return torch.equal(a,b)

    if isinstance(a, PackedSequence):
        a_sorted_indices = torch.arange(a.batch_sizes[0]) if a.sorted_indices is None else a.sorted_indices
        b_sorted_indices = torch.arange(b.batch_sizes[0]) if b.sorted_indices is None else b.sorted_indices
        a_unsorted_indices = torch.arange(a.batch_sizes[0]) if a.unsorted_indices is None else a.unsorted_indices
        b_unsorted_indices = torch.arange(b.batch_sizes[0]) if b.unsorted_indices is None else b.unsorted_indices
            
        return torch.equal(a.data, b.data) and \
               torch.equal(a.batch_sizes, b.batch_sizes) and \
               torch.equal(a_sorted_indices, b_sorted_indices) and \
               torch.equal(a_unsorted_indices, b_unsorted_indices) 

    if isinstance(a, list) or isinstance(a, tuple):
        return len(a)==len(b) and all(struct_equal( *ab) for ab in zip(a,b))

    if isinstance(a, dict):
        return (a.keys() == b.keys()) and struct_equal(list(a.values()), list(b.values()))

    return a == b



def breakup_packed_sequence( x: PackedSequence, N) -> List[PackedSequence]:
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
    

def mean_of_packed_sequence(x: PackedSequence, keepdim: bool = False) -> torch.Tensor:
    x_, l = pad_packed_sequence(x)
    x_ = x_.sum(0)
    x_ = div_vector(x_, l, dim=0)
    if keepdim:
        x_ = x_.unsqueeze(1)
    return x_


def sum_of_packed_sequence(x: PackedSequence, keepdim: bool = False) -> torch.Tensor:
    x_, l = pad_packed_sequence(x)
    x_ = x_.sum(0)
    if keepdim:
        x_ = x_.unsqueeze(1)
    return x_


def lengths_of_packed_sequence(x: PackedSequence) -> torch.Tensor:
    n_batch = x.batch_sizes[0]
    lengths = (x.batch_sizes.unsqueeze(0)>torch.arange(n_batch).unsqueeze(1)).sum(1)
    return lengths[x.unsorted_indices]


def chunk_lengths(chunks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    cum_chunks = chunks.cumsum(0)[:-1]
    lengths = lengths.view(-1, 1)
    lengths = torch.cat([lengths, lengths - cum_chunks.view(1, -1)], 1)
    lengths = torch.min(torch.max(lengths, torch.zeros_like(lengths)), chunks.view(1, -1))
    return lengths


# class BreakUp(object):

#     def __init__(self, N):
#         self.N = N
#         self.datatype = None 

#     def __call__(self, x):

#         if isinstance(x, torch.Tensor):
#             assert self.datatype in [None, torch.Tensor]
#             self.datatype = torch.Tensor
#             return list(x.chunk( self.N, dim=1))

#         if isinstance(x, PackedSequence):
#             assert self.datatype in [None, PackedSequence]
#             self.datatype = PackedSequence
#             return breakup_packed_sequence(x, self.N)

#         if isinstance(x, tuple):
#             return list( zip( *(self.__call__(xi) for xi in x)) )

#         if isinstance(x, list):
#             return list( list(block) for block in zip( *(self.__call__(xi) for xi in x)) )

#         if isinstance(x, dict):
#             return list( dict(zip(x.keys(),block)) for block in zip( *(self.__call__(xv) for xv in x.values())) )

#         raise ValueError(f'unknown type {type(x)}')

# class Combine(object):

#     def __init__(self, **kwargs):
#         self.N = None
#         self.datatype = None 
#         self.kwargs = kwargs

#     def __call__(self, x):
#         assert isinstance(x,list)

#         #print(x)
#         assert self.N is None or len(x)==self.N
#         self.N = len(x)

#         if isinstance(x[0], torch.Tensor):
#             return torch.cat(x, dim=1)
    
#         if isinstance(x[0], PackedSequence):
#             return combine_packed_sequence(x, **self.kwargs)
    
#         if isinstance(x[0], tuple):
#             return tuple(self.__call__(list(xi)) for xi in zip(*x))

#         if isinstance(x[0], list):
#             return list(self.__call__(list(xi)) for xi in zip(*x))

#         if isinstance(x[0], dict):     
#             return dict(zip(x[0].keys(), tuple(self.__call__(list(xi)) for xi in zip(*(xj.values() for xj in x)))))
