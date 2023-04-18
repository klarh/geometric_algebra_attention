import functools

import torch as pt

class CustomNorm(pt.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = pt.linalg.norm(x, axis=-1, keepdims=True)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        y = custom_norm(x)
        eps = pt.as_tensor(1e-19, dtype=x.dtype, device=x.device)
        return grad_output * (x / pt.maximum(y, eps))

custom_norm = CustomNorm.apply

@functools.lru_cache
def _bivec_dual_mult(dtype, device):
    mult = pt.as_tensor([1, -1, 1, -1], dtype=dtype, device=device)
    return mult

def bivec_dual(b):
    """scalar + bivector -> vector + trivector

    Calculates the dual of an input value, expressed as (scalar,
    bivector) with basis (1, e12, e13, e23).

    """
    return pt.flip(b, [-1])*_bivec_dual_mult(b.dtype, b.device).detach()

@functools.lru_cache
def _trivec_dual_mult(dtype, device):
    mult = pt.as_tensor([-1, 1, -1, 1], dtype=dtype, device=device)
    return mult

def trivec_dual(b):
    """vector + trivector -> scalar + bivector

    Calculates the dual of an input value, expressed as (scalar,
    bivector) with basis (1, e12, e13, e23).

    """
    return pt.flip(b, [-1])*_trivec_dual_mult(b.dtype, b.device).detach()

@functools.lru_cache
def _vecvec_swizzle(dtype, device):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    swizzle = pt.as_tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
    ], dtype=dtype, device=device)
    return swizzle

def vecvec(a, b):
    """vector*vector -> scalar + bivector

    Calculates the product of two vector inputs with basis (e1, e2,
    e3). Produces a (scalar, bivector) output with basis (1, e12, e13,
    e23).

    """
    products = a[..., None]*b[..., None, :]
    old_shape = products.shape
    new_shape = list(old_shape[:-2]) + [9]
    products = pt.reshape(products, new_shape)
    swizzle = _vecvec_swizzle(products.dtype, products.device).detach()
    return pt.tensordot(products, swizzle, 1)

def vecvec_invariants(p):
    """Calculates rotation-invariant attributes of a (scalar, bivector) quantity.

    Returns a 2D output: the scalar and norm of the bivector.

    """
    result = [p[..., :1], custom_norm(p[..., 1:4])]
    return pt.cat(result, axis=-1)

def vecvec_covariants(p):
    """Calculates rotation-covariant attributes of a (scalar, bivector) quantity.

    Converts the bivector to a vector by taking the dual.

    """
    dual = bivec_dual(p)
    return dual[..., :3]

@functools.lru_cache
def _bivecvec_swizzle(dtype, device):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = pt.as_tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
    ], dtype=dtype, device=device)
    return swizzle

def bivecvec(p, c):
    """(scalar + bivector)*vector -> vector + trivector

    Calculates the product of a (scalar + bivector) and a vector. The
    two inputs are expressed in terms of the basis (1, e12, e13, e23)
    and (e1, e2, e3); the output is expressed in terms of the basis
    (e1, e2, e3, e123).

    """
    products = p[..., None]*c[..., None, :]
    old_shape = products.shape
    new_shape = list(old_shape[:-2]) + [12]
    products = pt.reshape(products, new_shape)
    swizzle = _bivecvec_swizzle(products.dtype, products.device).detach()
    return pt.tensordot(products, swizzle, 1)

def bivecvec_invariants(q):
    """Calculates rotation-invariant attributes of a (vector, trivector) quantity.

    Returns a 2D output: the norm of the vector and the trivector.

    """
    result = [custom_norm(q[..., :3]), q[..., 3:4]]
    return pt.cat(result, axis=-1)

def bivecvec_covariants(q):
    """Calculates rotation-covariant attributes of a (vector, trivector) quantity.

    Returns the vector.

    """
    return q[..., :3]

@functools.lru_cache
def _trivecvec_swizzle(dtype, device):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = pt.as_tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
    ], dtype=dtype, device=device)
    return swizzle

def trivecvec(q, d):
    """(vector + trivector)*vector -> scalar + bivector

    Calculates the product of a (vector + trivector) and a vector. The
    two inputs are expressed in terms of the basis (e1, e2, e3, e123)
    and (e1, e2, e3); the output is expressed in terms of the basis
    (1, e12, e13, e23).

    """
    products = q[..., None]*d[..., None, :]
    old_shape = products.shape
    new_shape = list(old_shape[:-2]) + [12]
    products = pt.reshape(products, new_shape)
    swizzle = _trivecvec_swizzle(products.dtype, products.device).detach()
    return pt.tensordot(products, swizzle, 1)

trivecvec_invariants = vecvec_invariants

trivecvec_covariants = vecvec_covariants

def vec2trivec(v):
    """vector -> vector + trivector(0)

    This function simply appends a 0 in the appropriate location to
    treat a lone vector as a vector-trivector combination with basis
    (e1, e2, e3, e123).

    """
    trivec = pt.zeros_like(v[..., :1])
    return pt.cat([v, trivec], axis=-1)

@functools.lru_cache
def _mvec_dual_mult(dtype, device):
    mult = pt.as_tensor([
        -1, -1, 1, -1, 1, -1, 1, 1], dtype=dtype, device=device)
    return mult

def mvec_dual(m):
    return pt.flip(m, [-1])*_mvec_dual_mult(m.dtype, m.device).detach()

@functools.lru_cache
def _mvecmvec_swizzle(dtype, device):
    swizzle = pt.as_tensor([
        [ 1,  0,  0,  0,  0,  0,  0,  0], # 0
        [ 0,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1],
        [ 0,  1,  0,  0,  0,  0,  0,  0], # 8
        [ 1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  1,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1],
        [ 0,  0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0], # 16
        [ 0,  0,  0,  0, -1,  0,  0,  0],
        [ 1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0],
        [ 0, -1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0, -1],
        [ 0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0, -1,  0,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0], # 24
        [ 0,  0,  0,  0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0,  0, -1,  0],
        [ 1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1],
        [ 0, -1,  0,  0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0], # 32
        [ 0,  0, -1,  0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1],
        [-1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0, -1,  0],
        [ 0,  0,  0,  0,  0,  1,  0,  0],
        [ 0,  0,  0, -1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  1,  0,  0], # 40
        [ 0,  0,  0, -1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0, -1],
        [ 0,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0],
        [-1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -1,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0], # 48
        [ 0,  0,  0,  0,  0,  0,  0,  1],
        [ 0,  0,  0, -1,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0],
        [-1,  0,  0,  0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1], # 56
        [ 0,  0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0, -1,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0,  0,  0,  0],
        [-1,  0,  0,  0,  0,  0,  0,  0],
    ], dtype=dtype, device=device)
    return swizzle

def mvecmvec(a, b):
    """multivector*multivector -> multivector

    Calculates the product of two full multivector inputs with basis
    (1, e1, e2, e3, e12, e13, e23, e123). Produces a multivector output
    with basis (1, e1, e2, e3, e12, e13, e23, e123).

    """
    products = a[..., None]*b[..., None, :]
    old_shape = products.shape
    new_shape = list(old_shape[:-2]) + [64]
    products = pt.reshape(products, new_shape)
    swizzle = _mvecmvec_swizzle(products.dtype, products.device).detach()
    return pt.tensordot(products, swizzle, 1)

def mvecmvec_invariants(p):
    """Calculates rotation-invariant attributes of a multivector quantity.

    Returns a 4D output: the scalar, trivector and norms of the vector
    and bivector components.

    """
    result = [p[..., :1], custom_norm(p[..., 1:4]),
              custom_norm(p[..., 4:7]), p[..., 7:8]]
    return pt.cat(result, axis=-1)

def mvecmvec_covariants(p):
    """Calculates rotation-covariant attributes of a multivector quantity.

    Returns two sets of vectors, generated by the vector component and the
    dual of the bivector component.

    """
    vec = p[..., 1:4]
    # grab a phony scalar component from the input just to reuse the
    # bivec_dual function
    scalar_and_bivec = p[..., 3:7]
    dual = bivec_dual(scalar_and_bivec)[..., :3]
    return (vec, dual)
