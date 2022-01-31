import tensorflow as tf

@tf.custom_gradient
def custom_norm(x):
    """Calculate the norm of a set of vector-like quantities, with some
    numeric stabilization applied to the gradient."""
    y = tf.linalg.norm(x, axis=-1, keepdims=True)

    def grad(dy):
        y = custom_norm(x)
        return dy * (x / tf.maximum(y, 1e-19))

    return y, grad

def bivec_dual(b):
    """scalar + bivector -> vector + trivector

    Calculates the dual of an input value, expressed as (scalar,
    bivector) with basis (1, e12, e13, e23).

    """
    swizzle = tf.constant([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=b.dtype)
    return tf.tensordot(b, swizzle, 1)

def vecvec(a, b):
    """vector*vector -> scalar + bivector

    Calculates the product of two vector inputs with basis (e1, e2,
    e3). Produces a (scalar, bivector) output with basis (1, e12, e13,
    e23).

    """
    products = a[..., tf.newaxis]*b[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [9]], -1)
    products = tf.reshape(products, new_shape)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    swizzle = tf.constant([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)

def vecvec_invariants(p):
    """Calculates rotation-invariant attributes of a (scalar, bivector) quantity.

    Returns a 2D output: the scalar and norm of the bivector.

    """
    result = [p[..., :1], custom_norm(p[..., 1:4])]
    return tf.concat(result, axis=-1)

def vecvec_covariants(p):
    """Calculates rotation-covariant attributes of a (scalar, bivector) quantity.

    Converts the bivector to a vector by taking the dual.

    """
    dual = bivec_dual(p)
    return dual[..., :3]

def bivecvec(p, c):
    """(scalar + bivector)*vector -> vector + trivector

    Calculates the product of a (scalar + bivector) and a vector. The
    two inputs are expressed in terms of the basis (1, e12, e13, e23)
    and (e1, e2, e3); the output is expressed in terms of the basis
    (e1, e2, e3, e123).

    """
    products = p[..., tf.newaxis]*c[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [12]], -1)
    products = tf.reshape(products, new_shape)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = tf.constant([
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
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)

def bivecvec_invariants(q):
    """Calculates rotation-invariant attributes of a (vector, trivector) quantity.

    Returns a 2D output: the norm of the vector and the trivector.

    """
    result = [custom_norm(q[..., :3]), q[..., 3:4]]
    return tf.concat(result, axis=-1)

def bivecvec_covariants(q):
    """Calculates rotation-covariant attributes of a (vector, trivector) quantity.

    Returns the vector.

    """
    return q[..., :3]

def trivecvec(q, d):
    """(vector + trivector)*vector -> scalar + bivector

    Calculates the product of a (vector + trivector) and a vector. The
    two inputs are expressed in terms of the basis (e1, e2, e3, e123)
    and (e1, e2, e3); the output is expressed in terms of the basis
    (1, e12, e13, e23).

    """
    products = q[..., tf.newaxis]*d[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [12]], -1)
    products = tf.reshape(products, new_shape)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = tf.constant([
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
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)

trivecvec_invariants = vecvec_invariants

trivecvec_covariants = vecvec_covariants

def mvecmvec(a, b):
    """multivector*multivector -> multivector

    Calculates the product of two full multivector inputs with basis
    (1, e1, e2, e3, e12, e13, e23, e123). Produces a multivector output
    with basis (1, e1, e2, e3, e12, e13, e23, e123).

    """
    products = a[..., tf.newaxis]*b[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [64]], -1)
    products = tf.reshape(products, new_shape)
    swizzle = tf.constant([
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
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)

def mvecmvec_invariants(p):
    """Calculates rotation-invariant attributes of a multivector quantity.

    Returns a 4D output: the scalar, trivector and norms of the vector
    and bivector components.

    """
    result = [p[..., :1], custom_norm(p[..., 1:4]),
              custom_norm(p[..., 4:7]), p[..., 7:8]]
    return tf.concat(result, axis=-1)

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
