from __future__ import annotations
from abc import abstractmethod
import inspect
from typing import List, Optional, Union


import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from .utilities import *

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
            raise TypeError(f"hidden state {action} {name} must be Tensor, but was type {type(h)}")                
        if h.shape[1] != self.n_batch:
            raise ValueError(f"hidden state {action} {name} must be Tensor with shape (layers, batch_size(={self.n_batch}), ...), but was shape {h.shape}")


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


    def call( self, module, *args, **kwargs):
        f = module.forward
        filtered_kwargs = {
            name:kwargs[name]
            for name,param in inspect.signature(f).parameters.items() if (
                param.kind is inspect.Parameter.KEYWORD_ONLY or
                param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ) and
            name in kwargs
        }
        return module(*args, **filtered_kwargs)
    

    def forward_rnn(self, *args, **kwargs):
        res = self.call( self.rnn, *args, **kwargs)
        z = res[0]
        h = res[1] 

        extra_seq = res[2] if len(res)>2 else None
        extra = res[3] if len(res)>3 else None
        if len(res)>4:
            Warning(f"RNNModule returned more than 4 values. Ignoring spurious values of type(s) {tuple(type(x) for x in res[4:])}")
        return z, h, extra_seq, extra
        

    def forward_out(self, *args, **kwargs):
        res = self.call( self.out, *args, **kwargs)
        if type(res) is not tuple:
            res = (res,)
        loss = res[0]
        extra_seq = res[1] if len(res)>1 else None
        extra = res[2] if len(res)>2 else None
        if len(res)>3:
            Warning(f"RNNModule returned more than 3 values. Ignoring spurious values of type(s) {tuple(type(x) for x in res[3:])}")
        return loss, extra_seq, extra


class PlainRNN(BaseRNN):

    def __init__(self, rnn: nn.Module, out: nn.Module, loss_scaling: str = 'MEAN'):
        
        super().__init__(rnn, out)
        assert loss_scaling in ['MEAN','SUM']
        self.loss_scaling = loss_scaling


    def forward(self, x, y, h0, flatten_output=True, *args, **kwargs):

        z, h, extra_seq_rnn, extra_rnn = self.forward_rnn( x, h0, *args, **kwargs)
        
        if ispacked(x):
            example = flatten(x)[0]
            z = open_packed_sequence(z)
            assert ispacked(y)
            y = open_packed_sequence(y)

        loss_seq, extra_seq_out, extra_out = self.forward_out(z, y, *args, **kwargs)

        if ispacked(x):
            extra_seq_out = close_packed_sequence(extra_seq_out, example) 
            loss_seq = close_packed_sequence(loss_seq, example)
            if self.loss_scaling == 'MEAN':
                loss = mean_of_packed_sequence(loss_seq).mean()
            else:
                loss = sum_of_packed_sequence(loss_seq).sum()
        else:
            loss = loss_seq.mean() if self.loss_scaling == 'MEAN' else loss_seq.sum()

        if flatten_output:
            return flatten((loss, loss_seq, h,  extra_seq_rnn, extra_rnn, extra_seq_out, extra_out))
        else:
            return loss, loss_seq, h,  extra_seq_rnn, extra_rnn, extra_seq_out, extra_out



class VLRNN(PlainRNN):

    def __init__(self, rnn: nn.Module, out: nn.Module, loss_scaling: str = 'MEAN', N: Optional[int] = None):
        
        super().__init__(rnn, out, loss_scaling)
        self.N = N


    @staticmethod
    def breakup_sequence(x: Union[torch.Tensor,PackedSequence], N) -> List[Union[torch.Tensor,PackedSequence]]:
        if isinstance(x, torch.Tensor):
            return list(x.chunk( N, dim=1))

        if isinstance(x, PackedSequence):
            return breakup_packed_sequence(x, N)

    @staticmethod
    def breakup(x, N: int) -> List[Union[torch.Tensor,PackedSequence, tuple]]:
        if type(x) is tuple:
            return list(zip(*(VLRNN.breakup_sequence(xi, N) for xi in x)))
        else:
            return VLRNN.breakup_sequence(x, N)


    @staticmethod
    def combine_sequence(x: List[Union[torch.Tensor,PackedSequence]], example) -> Union[torch.Tensor,PackedSequence]:
        if isinstance(x[0], torch.Tensor):
            return torch.cat(x, dim=1)
    
        if isinstance(x[0], PackedSequence):
            return combine_packed_sequence(x, example.sorted_indices, example.unsorted_indices)


    @staticmethod
    def combine(s: List[Union[torch.Tensor,PackedSequence, tuple]], example) -> Union[torch.Tensor,PackedSequence, tuple]:
        if isinstance( s[0], list):
            return tuple(VLRNN.combine_sequence(list(x), example) for x in zip(*s))
        else:
            return VLRNN.combine_sequence(s, example)
  


    def forward(self, x, y, h0,  N=None, *args, **kwargs):

        example = flatten(x)[0]
        packed = isinstance(example, PackedSequence)
        if packed:
            seq_lengths = lengths_of_packed_sequence(example)
        else:
            seq_lengths = example.shape[1]

        if N is None:
            N = self.N
        assert N is not None and N>0

        #! n_batch = x.batch_size()
        #! assert n_batch > 0 

        x_blocks = self.breakup(x, N)
        if y is not None:
            y_blocks = self.breakup(y, N)
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
                
                z, h, extra_seq_rnn, extra_rnn = self.forward_rnn( x_n, h_n, *args, **kwargs)

                if h_last is None or not packed:
                    h_last = h
                else:  
                    def f(h1,h2):
                        res = h1.clone()
                        res[:,:h2.shape[1]] = h2
                        return res                      
                    h_last = struct_map2( f, h_last, h)

                if n<N-1:
                    if not packed:
                        h_blocks[n+1] = h
                    else:
                        def f(h):
                            return h[:,:len(flatten(x_blocks[n+1])[0].sorted_indices)]
                        h_blocks[n+1] = struct_map(f, h)

        if packed:
            h_last = struct_map(lambda h: h[:,x.unsorted_indices], h_last)

        loss = 0
        delta_h_n_plus_1 = None
        loss_seq_blocks = [None] * N
        extra_seq_rnn_blocks = [None] * N
        extra_seq_out_blocks = [None] * N

        for n in reversed(range(N)):

            x_n = x_blocks[n]
            y_n = y_blocks[n]
            h_n = h_blocks[n]

            if torch.is_grad_enabled():
                if h_n is not None:
                    # h_n.requires_grad = True
                    requires_grad(h_n, True)

            loss_n, loss_seq_n, h_n_plus_1,  extra_seq_rnn_n, extra_rnn, extra_seq_out_n, extra_out = super().forward(x_n, y_n, h_n, False, *args, **kwargs)

            loss_seq_blocks[n] = loss_seq_n
            extra_seq_rnn_blocks[n] = extra_seq_rnn_n
            extra_seq_out_blocks[n] = extra_seq_out_n
        
            if packed:
                loss_n = sum_of_packed_sequence(loss_seq_n)
                if self.loss_scaling == 'MEAN':
                    scale = seq_lengths[example.sorted_indices[:len(loss_n)]]
                    loss_n = (loss_n / scale).sum() / len(seq_lengths)
                else:
                    loss_n = loss_n.sum()
            else:
                if self.loss_scaling == 'MEAN':
                    loss_n = loss_n * flatten(x_n)[0].shape[1] / seq_lengths 

            # else, super().forward computes the loss corectly
            
            if torch.is_grad_enabled():
                if n < N - 1:
                    loss_n.backward(retain_graph=True)

                    # we need to strip h_n_plus_1 from all sequences that don't continue in the next block
                    if packed:
                        def f(h):
                            return h[:,:len(flatten(x_blocks[n+1])[0].sorted_indices)]
                        h_n_plus_1 = struct_map(f, h_n_plus_1)

                    h_n_plus_1_ = list(h_n_plus_1)
                    delta_h_n_plus_1_ = list(delta_h_n_plus_1)

                    torch.autograd.backward(h_n_plus_1_, grad_tensors=delta_h_n_plus_1_)
                else:
                    loss_n.backward()

                if h_n is not None:
                    delta_h_n_plus_1 = grad_of(h_n)
                else:
                    delta_h_n_plus_1 = None

            loss = loss + loss_n.detach() #.item()


        loss_seq = VLRNN.combine( loss_seq_blocks, example)
        extra_seq_rnn = VLRNN.combine( extra_seq_rnn_blocks, example)
        extra_seq_out = VLRNN.combine( extra_seq_out_blocks, example)
        
        if (extra_rnn is not None) or (extra_out is not None):
            Warning("Ignoring and discarding 'extra_rnn' and 'extra_out' outputs in VLRNN.forward !")

        return flatten((loss, loss_seq, h_last, extra_seq_rnn, extra_seq_out))
