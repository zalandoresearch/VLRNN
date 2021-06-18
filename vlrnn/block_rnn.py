from __future__ import annotations
from abc import ABC, abstractmethod
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