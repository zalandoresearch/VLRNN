from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence

from utilities import BreakUp, Combine, breakup_packed_sequence, div_vector, lengths_of_packed_sequence, mean_of_packed_sequence, mul_vector, struct_equal, struct_flatten, requires_grad, grad_of, struct_unflatten, sum_of_packed_sequence

class SequenceStruct(object):

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

        self.parent_sorted_indices = None
        self.validate()

    def validate(self, x=None, root=True):
        if root:
            x = self.struct

        if isinstance(x, torch.Tensor):
            # print(x.shape)
            self.assert_and_assign('container_type', torch.Tensor)
            self.assert_and_assign('seq_lengths', torch.full((x.shape[0],), x.shape[1], dtype=torch.long) )
            self.assert_and_assign('num_seq', x.shape[0]) 
            self.assert_and_assign('sorted_indices', torch.arange(x.shape[0]))
            self.assert_and_assign('unsorted_indices', torch.arange(x.shape[0]))

        elif isinstance(x, PackedSequence):
            self.assert_and_assign('container_type', PackedSequence)
            self.assert_and_assign('seq_lengths', lengths_of_packed_sequence(x)) 
            self.assert_and_assign('num_seq', x.batch_sizes[0]) 
            self.assert_and_assign('sorted_indices', x.sorted_indices)
            self.assert_and_assign('unsorted_indices', x.unsorted_indices)


        elif isinstance(x, tuple) or isinstance(x, list):
            for xi in x:
                self.validate(xi, root=False)

        elif isinstance(x, dict):
            for xi in x.values():
                self.validate(xi, root=False)
        else:
            raise ValueError(f'unknown type {type(x)}')


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


    def breakup(self, N: int) -> List[SequenceStruct]:
        br = BreakUp(N)
        return [SequenceStruct(s) for s in br(self.struct)]


    @staticmethod
    def combine(x: List[SequenceStruct], **kwargs) -> SequenceStruct:
        cm = Combine(**kwargs)
        return SequenceStruct( cm([xi.struct for xi in x]))
        

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




class BlockRNNOutput(nn.Module, ABC):
    pass


class BlockRNN(nn.Module, ABC):

    def __init__(self, rnn: nn.Module, out: BlockRNNOutput, loss_scaling: str, N: Optional[int]=None):
        
        super().__init__()
        assert loss_scaling in ['MEAN','SUM']
        self.rnn = rnn
        self.out = out
        self.loss_scaling = loss_scaling
        self.N = N

    def zero_grad(self):
        self.rnn.zero_grad
        self.out.zero_grad

    def _call_rnn(self, x, h):
        if isinstance(x, SequenceStruct):
            x = x.struct
        if h is not None and isinstance(h, SequenceStruct):
            h = h.struct
        return self.rnn(x, h)

    def _call_out(self, z, y):
        if isinstance(z, SequenceStruct):
            z = z.struct
        if y is not None and isinstance(y, SequenceStruct):
            y = y.struct
        return self.out(z, y)


    def forward(self, x, h0, y=None, N=None):

        x = SequenceStruct(x)
        if y is not None:
            y = SequenceStruct(y)
            assert x.is_compatible_with(y)
            sampling = False
        else:
            sampling = True

        if N is None:
            N = self.N
        assert N is not None and N>0


        x_blocks = x.breakup(N)
        if not sampling:
            y_blocks = y.breakup(N)
            assert len(x_blocks) == len(y_blocks)


        N = len(x_blocks) # may be different than requested for packed sequences

        h_blocks = [h0] + [None] * N
        with torch.no_grad():
            for n in range(N):
                x_n = x_blocks[n]
                h_n = h_blocks[n]

                z, h = self._call_rnn(x_n, h_n)
                h_blocks[n + 1] = h
            
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
            if sampling:
                loss_n, y_n = self._call_out(z_n)
                y_blocks[n] = SequenceStruct(y_n)
            else:
                loss_n = self._call_out(z_n, y_n)
            loss_n = SequenceStruct(loss_n)

            if self.loss_scaling == "MEAN":
                scale = x.seq_lengths[x.sorted_indices[:loss_n.num_seq]]
                loss_n = loss_n.seq_sum(scale) # leading to (unpacked) sequences of lengths 1
                loss_n = loss_n.sum() # now we have a scalar
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

        if sampling:
            return loss, SequenceStruct.combine(y_blocks, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
        else:
            return loss #$#, torch.cat(loss_blocks, 1)
