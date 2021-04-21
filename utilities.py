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
    if isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            yield from struct_flatten(xi)
    else:
        if x is not None:
            yield x


def struct_unflatten(x, proto):
    if isinstance(proto, tuple):
        return tuple(struct_unflatten(x, p) for p in proto)
    elif isinstance(proto, list):
        return list(struct_unflatten(x, p) for p in proto)
    elif proto is None:
        return None
    else:
        return next(x)
