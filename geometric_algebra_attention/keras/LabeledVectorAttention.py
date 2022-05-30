
from tensorflow import keras

from .. import base
from .Vector2VectorAttention import Vector2VectorAttention

class LabeledVectorAttention(base.LabeledVectorAttention, Vector2VectorAttention):
    __doc__ = base.LabeledVectorAttention.__doc__

    def build(self, input_shape):
        modified_shape = input_shape[1]
        return super().build(modified_shape)

    def compute_mask(self, inputs, mask=None):
        """Calculate the output mask of this layer given input shapes and masks."""
        if mask is None:
            return mask

        (child_mask, other_mask) = mask
        return child_mask

keras.utils.get_custom_objects()['LabeledVectorAttention'] = LabeledVectorAttention
