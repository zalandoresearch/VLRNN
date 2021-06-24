from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
from typing import List, Optional, Union

import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence

from .utilities import breakup_packed_sequence, combine_packed_sequence, div_vector, lengths_of_packed_sequence, mean_of_packed_sequence, mul_vector, struct_equal, struct_flatten, requires_grad, grad_of, struct_map, struct_map2, struct_unflatten, sum_of_packed_sequence

class SequenceStruct(object):
    """A conatiner class that either represents a sequence container or a list of sequence containers. Sequence containers can be
    * torch.Tensor for regular, equally sized, batch-first sequences  (num_batch, len_seq, channle_1, channel_2, ...)
    * torch.nn.utils.rnn.PackedSequence

    In a list, all containers ust be of same type and structure regarding batch size, sequence length(s).
    """
    def __init__(self, struct):
        if isinstance( struct_equal, SequenceStruct):
            self.struct = struct.struct
        else:
            self.struct = struct
        self.container_type = None
        self.num_seq = None
        self.seq_lengths = None
        self.sorted_indices = None
        self.unsorted_indices = None

        # self.parent_sorted_indices = None
        self.validate()

    def validate(self, x=None, root=True):
        if root:
            x = self.struct

        if isinstance(x, torch.Tensor):
            # print(x.shape)
            self.assert_and_assign('container_type', torch.Tensor)
            self.assert_and_assign('seq_lengths', torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device))
            self.assert_and_assign('num_seq', x.shape[0]) 
            self.assert_and_assign('sorted_indices', torch.arange(x.shape[0], device=x.device))
            self.assert_and_assign('unsorted_indices', torch.arange(x.shape[0], device=x.device))

        elif isinstance(x, PackedSequence):
            self.assert_and_assign('container_type', PackedSequence)
            self.assert_and_assign('seq_lengths', lengths_of_packed_sequence(x)) 
            self.assert_and_assign('num_seq', x.batch_sizes[0]) 
            self.assert_and_assign('sorted_indices', x.sorted_indices)
            self.assert_and_assign('unsorted_indices', x.unsorted_indices)


        elif isinstance(x, list):
            assert root # no lists of lists allowed!
            for xi in x:
                self.validate(xi, root=False)
        else:
            raise ValueError(f'SequenceStruct cannot contain objets of type {type(x)}')


    def assert_and_assign(self, name, new_value):
        old_value = getattr(self, name)
        # print(name, old_value, new_value, file=sys.stderr)
        if old_value is None:
            setattr(self, name, new_value)
        else:
            assert struct_equal(old_value, new_value)


    def is_compatible_with(self,  y: SequenceStruct) -> bool:
        # same seq_lengths (implying same batch_size for PackedSequences)
        # same container type (Tensor or PackedSequence)
        if self.container_type != y.container_type: 
            return False
        return torch.equal( self.seq_lengths, y.seq_lengths)


    @staticmethod
    def breakup_container(x: Union[torch.Tensor,PackedSequence], N) -> List[Union[torch.Tensor,PackedSequence]]:
        if isinstance(x, torch.Tensor):
            return list(x.chunk( N, dim=1))

        if isinstance(x, PackedSequence):
            return breakup_packed_sequence(x, N)


    def breakup(self, N: int) -> List[SequenceStruct]:
        if isinstance( self.struct, list):
            return [SequenceStruct(list(s)) for s in zip(*(self.breakup_container(x, N) for x in self.struct))]
        else:
            return [SequenceStruct(x) for x in self.breakup_container(self.struct, N)]


    @staticmethod
    def combine_container(x: List[Union[torch.Tensor,PackedSequence]], **kwargs) -> Union[torch.Tensor,PackedSequence]:
        if isinstance(x[0], torch.Tensor):
            return torch.cat(x, dim=1)
    
        if isinstance(x[0], PackedSequence):
            return combine_packed_sequence(x, **kwargs)


    @staticmethod
    def combine(s: List[SequenceStruct], **kwargs) -> SequenceStruct:
        if isinstance( s[0].struct, list):
            return SequenceStruct([SequenceStruct.combine_container(list(x), **kwargs) for x in zip(*[si.struct for si in s])])
        else:
            return SequenceStruct( SequenceStruct.combine_container( [si.struct for si in s], **kwargs))
        

    def seq_mean(self, x=None, root=True) -> SequenceStruct:
        if self.container_type is torch.Tensor:
            flat = (f.mean(1,keepdim=True) for f in struct_flatten(self.struct))
        elif self.container_type is PackedSequence:
            flat = (mean_of_packed_sequence(f,keepdim=True) for f in struct_flatten(self.struct))
        return SequenceStruct(struct_unflatten(flat, self.struct))


    def seq_sum(self, scale=None) -> SequenceStruct:
        if self.container_type is torch.Tensor:
            if scale is None:
                aggr = lambda f: f.sum(1,keepdim=True)
            else:
                aggr = lambda f: div_vector( f.sum(1,keepdim=True), scale, dim=0)

        elif self.container_type is PackedSequence:
            if scale is None:
                aggr = lambda f: sum_of_packed_sequence(f, keepdim=True)
            else:
                aggr = lambda f: div_vector( sum_of_packed_sequence(f, keepdim=True), scale, dim=0)
                
        flat = (aggr(f) for f in struct_flatten(self.struct))
        return SequenceStruct(struct_unflatten(flat, self.struct))

    def mean(self):
        if self.container_type is torch.Tensor:
            flat = [f.mean(1).flatten() for f in struct_flatten(self.struct)]
        elif self.container_type is PackedSequence:
            flat = [mean_of_packed_sequence(f, keepdim=True).flatten() for f in struct_flatten(self.struct)]
        return torch.cat(flat,0).mean()

    def sum(self):
        if self.container_type is torch.Tensor:
            flat = [f.sum() for f in struct_flatten(self.struct)]
        elif self.container_type is PackedSequence:
            flat = [sum_of_packed_sequence(f, keepdim=True).sum() for f in struct_flatten(self.struct)]
        return torch.stack(flat,0).sum()

    def batch_size(self):
        x = next(struct_flatten(self.struct))
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        elif isinstance(x, PackedSequence):
            return x.batch_sizes[0]
        else:
            raise ValueError(f"type {type(x)} cannot occur in SequenceStruct")
 
class BlockRNN(nn.Module, ABC):

    def __init__(self, rnn: nn.Module, out: nn.Module, loss_scaling: str, N: Optional[int]=None):
        
        super().__init__()
        assert loss_scaling in ['MEAN','SUM']
        self.rnn = rnn
        self.out = out
        self.loss_scaling = loss_scaling
        self.N = N

    def zero_grad(self):
        self.rnn.zero_grad()
        self.out.zero_grad()

    def _call_rnn(self, x, h):
        if isinstance(x, SequenceStruct):
            x = x.struct
        if h is not None and isinstance(h, SequenceStruct):
            h = h.struct
        return self.rnn(x, h)

    def _call_out(self, z, y=None, return_y=False):
        if isinstance(z, SequenceStruct):
            z = z.struct
        if y is not None and isinstance(y, SequenceStruct):
            y = y.struct
        return self.out(z, y, return_y=return_y)


    def forward(self, x, h0, y=None, N=None, return_y=None, return_h=None):

        x = SequenceStruct(x)
        if y is not None:
            y = SequenceStruct(y)
            assert x.is_compatible_with(y)

        packed = x.container_type is PackedSequence

        if N is None:
            N = self.N
        assert N is not None and N>0

        n_batch = x.batch_size()
        assert n_batch > 0 

        x_blocks = x.breakup(N)
        if y is not None:
            y_blocks = y.breakup(N)
            assert len(x_blocks) == len(y_blocks)
        else:
            y_blocks = [None] * N


        N = len(x_blocks) # may be different than requested for packed sequences

        h_blocks = [h0] + [None] * N
        h_last = None # to aggregate the last
        with torch.no_grad():
            for n in range(N):
                x_n = x_blocks[n]
                h_n = h_blocks[n]
                
                z, h = self._call_rnn(x_n, h_n)
                h_blocks[n + 1] = h

                if h_last is None or not packed:
                    h_last = h
                else:  
                    def f(h1,h2):
                        res = h1.clone()
                        res[:,:h2.shape[1]] = h2
                        return res                      
                    h_last = struct_map2( f, h_last, h)

        h_last = struct_map(lambda h: h[:,x.unsorted_indices], h_last)
            
        #$# loss_blocks = [None] * N

        loss = 0
        delta_h_n_plus_1 = None
        for n in reversed(range(N)):

            x_n = x_blocks[n]
            y_n = y_blocks[n]
            h_n = h_blocks[n]

            if torch.is_grad_enabled():
                if h_n is not None:
                    # h_n.requires_grad = True
                    requires_grad(h_n, True)

            z_n, h_n_plus_1 = self._call_rnn(x_n, h_n)
            if return_y:
                loss_n, y_n = self._call_out(z_n, y_n, return_y=True)
                y_blocks[n] = SequenceStruct(y_n)
            else:
                loss_n = self._call_out(z_n, y_n)
            loss_n = SequenceStruct(loss_n)

            if self.loss_scaling == "MEAN":
                scale = x.seq_lengths[x.sorted_indices[:loss_n.num_seq]]
                loss_n = loss_n.seq_sum(scale) # leading to (unpacked) sequences of lengths 1
                loss_n = loss_n.sum()/n_batch # now we have a scalar
            else:
                loss_n = loss_n.seq_sum() # leading to (unpacked) sequences of lengths 1
                loss_n = loss_n.sum() # now we have a scalar
            
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

        res = (loss,)

        if return_y:
            res = res + (SequenceStruct.combine(y_blocks, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices).struct,)
        if return_h == 'last':
            res = res + (h_last,)

        if len(res)==1:
            res = res[0]

        return res


class _CheckedModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.enable_checks = True
        self.reset_checks()

    def __call__(self, *args, **kwargs):
        self.input_parameter_check(*args, **kwargs)
        res = super().__call__(*args, **kwargs)
        self.output_parameter_check(res)
        self.reset_checks()
        return res

    def reset_checks(self):
        self.packed = None
        self.n_batch = None
        self.n_seq = None


    def valid_sequence(self, x, name, action):

        if self.packed is None:
            self.packed = isinstance(x, PackedSequence)
            if self.packed:
                self.n_batch = x.batch_sizes[0]
                self.n_seq = lengths_of_packed_sequence(x)
            else:
                if not isinstance(x, torch.Tensor) or x.ndim<2:
                    raise TypeError(f"{action} {name} of forward() must be a sequence Tensor with at least 2 dimensions, but was {x}")
                self.n_batch, self.n_seq = x.shape[:2]
        else:
            if self.packed:
                if not isinstance(x, PackedSequence):
                    raise TypeError(f"{action} {name} of forward() must be PackedSequence, but was {type(x)}")
                n_seq = lengths_of_packed_sequence(x)
                if not torch.equal(n_seq, self.n_seq):
                    raise ValueError(f"{action} {name} of forward() must be PackedSequence with sequence lengths {self.n_seq}, but has lengths {n_seq}")
            else:
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"{action} {name} of forward() must be a sequence Tensor, but was {type(x)}")
                if (x.ndim<2) or (x.shape[0] != self.n_batch) or (x.shape[1]!=self.n_seq):
                    raise ValueError(f"argument {name} of forward() must be ({self.n_batch},{self.n_seq},...) sequence Tensor, but had shape {x.shape}")

    def valid_latent(self, h, name, action):
        if (not isinstance(h, torch.Tensor)):
            raise TypeError(f"hidden state {action} {name} must be batch-first Tensor, but was {h}")                
        if h.shape[0] != self.n_batch:
            raise ValueError(f"hidden state {action} {name} must be batch-first Tensor with batch_size {self.n_batch}, but was shape {h.shape}")


    def check_sequence_arg(self, z, name, action):
        if type(z) not in (torch.Tensor, PackedSequence, tuple):
            raise TypeError(f"{action} {name} of forward() must be a sequence or a tuple of sequences, but was {type(z)}")

        if type(z) is not tuple:
            z = (z,)

        for i,z_ in enumerate(z):
            # ToDo: possibly allow None values in the tuple of sequences
            self.valid_sequence(z_, f"{name}[{i}]", action)


    def check_latent_arg(self, h, name, action):
        if type(h) not in (torch.Tensor, tuple):
            raise TypeError(f"{action} {name} of forward() must be a latent tensor or a tuple of latent tensors, but was {type(h)}")

        if type(h) is not tuple:
            h = (h,)

        for i,h_ in enumerate(h):
            # ToDo: possibly allow None values in the tuple of latent tensors
            self.valid_latent(h_, f"{name}[{i}]", action)


class OutputModule(_CheckedModule):
    
    @abstractmethod
    def forward(self, z, y, **kwargs):
        pass


    def input_parameter_check(self, *args, **kwargs):

        if not self.enable_checks:
            return 

        argspec = inspect.getfullargspec(self.forward)
        callargs = inspect.getcallargs( self.forward, *args, **kwargs)
        
        M = len(argspec.args)-1   

        if M<2:
            raise TypeError(f"forward() must be called with two arguments (and optional kwargs), but had {M} fixed arguments")

        name = argspec.args[1]
        z = callargs[name]
        self.check_sequence_arg(z, name, 'argument')

        if self.packed:
            raise TypeError("forward() cannot be called with packed sequences")

        name = argspec.args[2]
        y = callargs[name]
        if y is not None:
            self.check_sequence_arg(y, name, 'argument')



    def output_parameter_check(self, res):

        if not self.enable_checks:
            return 

        if type(res) is not tuple:
            res = (res,)
        if  len(res)<1:
            raise TypeError(f"forward() must return at least one output, loss, but returned {res}")

        l = res[0]
        self.valid_sequence(l, "loss", "output")
        if (l.ndim != 2) :
            raise ValueError( f"forward() must return loss: Tensor(n_batch={self.n_batch}, n_seq={self.n_seq}), but returned shape {l.shape} ")

        if len(res) > 1:
            extra_seq = res[1]
            if extra_seq is not None:
                self.check_sequence_arg(extra_seq, 'extra_seq', 'output')


class RNNModule(_CheckedModule):

    @abstractmethod
    def forward(self, x, h, **kwargs):
        pass


    def input_parameter_check(self, *args, **kwargs):

        if not self.enable_checks:
            return 

        argspec = inspect.getfullargspec(self.forward)
        callargs = inspect.getcallargs( self.forward, *args, **kwargs)

        M = len(argspec.args)-1   

        if M<2:
            raise TypeError(f"forward() must be called with at least two fixed arguments (and optional args, kwargs), but had {M} fixed arguments {argspec.args[1:M+1]}")
        
        name = argspec.args[1]
        x = callargs[name]
        self.check_sequence_arg( x, name, 'argument')

        name = argspec.args[2]
        h = callargs[name]
        if h is not None:
            self.check_latent_arg( h, name, 'argument')


    def output_parameter_check(self, res):
        
        if not self.enable_checks:
            return 
        if (type(res) is not tuple) or len(res)<2:
            raise TypeError(f"forward() must return two (tuples of) outputs, z and h, but returned {res}")
        z = res[0]
        h = res[1]

        self.check_sequence_arg(z, 'z', 'output')
        if h is not None:
            self.check_latent_arg(h, 'h', 'output')
        if len(res) >= 3:
            extra_seq = res[2]
            if extra_seq is not None:
                self.check_sequence_arg(extra_seq, 'extra_seq', 'output')


class BaseRNN(nn.Module):

    def __init__(self, rnn: RNNModule, out: nn.Module):
        
        super().__init__()
        assert isinstance(rnn, RNNModule)
        self.rnn = rnn
        assert isinstance(out, OutputModule)
        self.out = out


    def flatten(x):
        if isinstance(x, type(None)):
            return ()
        if isinstance(x, (torch.Tensor, PackedSequence)):
            return (x,)
        if isinstance(x, tuple):
            return tuple(xi for xi in x if xi is not None)            


    def ispacked(self, x):
        if isinstance(x, (type(None), PackedSequence)):
            return True
        if isinstance(x, tuple) and all( isinstance(xi, (type(None), PackedSequence)) for xi in x):
            return True
        if any( isinstance(xi, (type(None), PackedSequence)) for xi in x):
            raise TypeError("Some, but not all items in x are PackedSequences.")

        return False


    def open_packed_sequence(self, x):
        if isinstance(x, type(None)):
            return None
        if isinstance(x, PackedSequence):
            return x.data.unsqueeze(0)
        if isinstance(x, tuple):
            return tuple( self.open_packed_sequence(xi) for xi in x)
        raise TypeError("x is not a (tuple of) packed sequence(s)")


    def close_packed_sequence(self, x, example: PackedSequence):
        if isinstance(x, type(None)):
            return None
        if isinstance(x, PackedSequence):
            return PackedSequence( data=x.squeeze(0), batch_sizes=example.batch_sizes, 
            sorted_indices=example.sorted_indices, unsorted_indices=example.unsorted_indices)
        if isinstance(x, tuple):
            return tuple( self.close_packed_sequence(xi, example) for xi in x)
        raise TypeError("x is not a (tuple of) opened packed sequence(s)")



class PlainRNN(BaseRNN):

    def __init__(self, rnn: nn.Module, out: nn.Module, loss_scaling: str):
        
        super().__init__()
        assert loss_scaling in ['MEAN','SUM']
        self.rnn = rnn
        self.out = out
        self.loss_scaling = loss_scaling


    def forward(self, x, y, h0, *args, **kwargs):

        z, h, extra_seq_rnn = self.rnn( x, h0, *args, **kwargs)

        if self.ispacked(x):
            example = self.flatten(x)[0]
            z = self.open_packed_sequence(z)
            assert self.ispacked(y)
            y = self.open_packed_sequence(y)

        loss, extra_seq_out = self.out(z, y, *args, **kwargs) 

        if self.ispacked(x):
            extra_seq_out = self.close_packed_sequence(extra_seq_out, example) 

        return loss, extra_seq_rnn, extra_seq_out, h
