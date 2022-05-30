import torch as pt

from .. import base
from . import geometric_algebra

def keepdims_decorator(f):
    def wrapped(*args, **kwargs):
        if 'keepdims' in kwargs:
            kwargs['keepdim'] = kwargs.pop('keepdims')
        return f(*args, **kwargs)
    return wrapped

class AttentionBase:
    algebra = geometric_algebra

    math = base.Namespace(
        all=pt.all,
        any=pt.any,
        asarray=pt.as_tensor,
        bool_to_int=lambda x: x.to(pt.int8),
        clip=pt.clip,
        concat=pt.cat,
        logical_and=pt.logical_and,
        pow=pt.pow,
        product=keepdims_decorator(pt.prod),
        reshape=pt.reshape,
        shape=lambda x: x.shape,
        softmax=pt.softmax,
        sqrt=pt.sqrt,
        sum=keepdims_decorator(pt.sum),
        tensordot=pt.tensordot,
        where=pt.where,
        zeros_like=pt.zeros_like,
    )

    def init(self):
        """Initialize the weights for this layer."""
        weight_sets = self._build_weight_definitions(self.n_dim)
        for (name, defs) in weight_sets.groups.items():
            weights = pt.nn.ParameterList([
                pt.nn.Parameter(pt.normal(0, pt.ones(*def_.shape)*def_.stdev)) for def_ in defs])
            setattr(self, name, weights)

        for (name, def_) in weight_sets.singles.items():
            weight = pt.nn.Parameter(pt.normal(0, pt.ones(*def_.shape)*def_.stdev))
            setattr(self, name, weight)

    def _calculate_attention(self, scores, values, old_shape):
        dims, reduce_axes = self._get_reduction()

        shape = list(old_shape[:dims]) + [old_shape[dims:].numel()]
        scores = self.math.reshape(scores, shape)
        attention = self.math.reshape(self.math.softmax(scores, -1), old_shape)
        if reduce_axes:
            output = self.math.sum(attention*values, reduce_axes)
        else:
            output = attention*values

        return attention, output

    def forward(self, inputs):
        """Evaluate the attention calculation for this layer."""
        return self._evaluate(inputs).output
