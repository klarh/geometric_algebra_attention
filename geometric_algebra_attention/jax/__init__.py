
"""The JAX API provides layers (currently only in the style of stax)
for use in jax. For the most part the layers work as standard stax
layers; the main caveats concern the functions used for generating
scores, values, and scaling factors:

- These functions are passed in in the stax form of layers: a tuple of `(init(rng, input_shape), apply(params, x))` functions.
- Given functions become "owned" by the parent `VectorAttention` or other layer objects. Their initialization and evaluation functions are incorporated into the main layer's initialization and evaluation.

"""

from . import geometric_algebra
from .LabeledMultivectorAttention import LabeledMultivectorAttention
from .LabeledVectorAttention import LabeledVectorAttention
from .MultivectorAttention import MultivectorAttention
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention
from .Multivector2Vector import Multivector2Vector
from .VectorAttention import VectorAttention
from .Vector2Multivector import Vector2Multivector
from .Vector2VectorAttention import Vector2VectorAttention
