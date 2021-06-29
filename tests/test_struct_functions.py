
from vlrnn import struct_flatten, struct_unflatten, struct_equal, lengths_of_packed_sequence

import pytest

import torch
torch.manual_seed(0)

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


x1=torch.randn(50,5,4)
x2=torch.randn(100,5,4)
x3=torch.randn(20,5,4)

y1=torch.randn(50,14)
y2=torch.randn(100,14)
y3=torch.randn(20,14)

z1=torch.randn(50)
z2=torch.randn(100)
z3=torch.randn(20)

l = torch.tensor([50,100,20])    


@pytest.fixture(scope='module', params=[
    torch.device('cpu'), 
    pytest.param(torch.device('cuda'), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))], 
    ids=['CPU','CUDA'])
def device(request):
    return request.param

# functions that create a complicated structures, together with the corresponding flat (list) structure
struct_lambdas = [
    lambda x,y,z: (x,                       [x]),
    lambda x,y,z: ([x, y, z],               [x, y, z]),
]

struct_lambdas_bad = [
    lambda x,y,z: ((x, [y,z]),              [x, y, z]),
    lambda x,y,z: ({'x':x, 'y':y, 'z':[z]}, [x, y, z]),    
    lambda x,y,z: ([x, y, [z]],             [x, y, z]),
    lambda x,y,z: ([x, y, 'z'],             [x, y]),
]


def make_seq(mode, device):
    xp = pack_sequence([x1,x2,x3], enforce_sorted=False)
    yp = pack_sequence([y1,y2,y3], enforce_sorted=False)
    zp = pack_sequence([z1,z2,z3], enforce_sorted=False)

    if mode=='PACKED':
        return xp.to(device), yp.to(device), zp.to(device)
    x = pad_packed_sequence(xp)[0].transpose(0,1)
    y = pad_packed_sequence(yp)[0].transpose(0,1)
    z = pad_packed_sequence(zp)[0].transpose(0,1)
    if mode=='FIXED':
        return x.to(device),y.to(device),z.to(device)

    # MIXED
    return x.to(device), y.to(device), zp.to(device)

@pytest.fixture(scope='module', params=['PACKED', 'FIXED','MIXED'])
def seq(request, device):
    return make_seq(request.param, device)
    
@pytest.fixture(scope='module', params=['PACKED', 'FIXED'])
def valid_seq(request, device):
    return make_seq(request.param, device)
    
@pytest.fixture(scope='module', params=['PACKED'])
def packed_seq(request, device):
    return make_seq(request.param, device)
    
@pytest.fixture(scope='module', params=['MIXED'])
def mixed_seq(request, device):
    return make_seq(request.param, device)
    
@pytest.fixture(scope='module', params=['FIXED'])
def fixed_seq(request, device):
    return make_seq(request.param, device)
    

def test_legth_of_packed_sequence(request, packed_seq):
    x,y,z = packed_seq
    assert torch.equal( lengths_of_packed_sequence(x), l.to(x.data.device))
    assert torch.equal( lengths_of_packed_sequence(y), l.to(y.data.device))
    assert torch.equal( lengths_of_packed_sequence(z), l.to(z.data.device))


@pytest.fixture(scope='module', params=struct_lambdas)
def make_struct(request):
    return request.param

@pytest.fixture(scope='module', params=struct_lambdas)
def make_other_struct(request):
    return request.param

@pytest.fixture(scope='module', params=struct_lambdas_bad)
def make_struct_bad(request):
    return request.param

def test_struct_equal(request, seq, make_struct, make_other_struct):
    x,y,z = seq
    s1, s1_flat = make_struct(x,y,z)
    s2, s2_flat = make_other_struct(x,y,z)
    
    assert struct_equal(s1,s2) == (make_struct==make_other_struct)


def test_struct_flatten(seq, make_struct):
    x,y,z = seq
    s, s_flat = make_struct(x,y,z)
    assert struct_equal( list(struct_flatten(s)), s_flat)

def test_struct_unflatten(seq, make_struct):
    x,y,z = seq
    s, s_flat = make_struct(x,y,z)
    assert struct_equal( struct_unflatten(iter(s_flat), s), s)


def test_struct_flatten_unflatten(seq, make_struct):
    x,y,z = seq

    s1, s1_flat = make_struct(x,y,z)
    f1 = struct_flatten(s1)
    s2 = struct_unflatten( f1, s1)
    assert struct_equal(s1,s2)
