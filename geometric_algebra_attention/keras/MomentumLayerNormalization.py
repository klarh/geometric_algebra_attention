
import tensorflow as tf
from tensorflow import keras

from ..tensorflow.geometric_algebra import custom_norm

class MomentumLayerNormalization(keras.layers.Layer):
    """Exponential decay normalization.

    Calculates a running average of the L2 norm and scales inputs to
    have length (over the last axis) 1, on average.

    :param momentum: Momentum of moving average, from 0 to 1
    :param epsilon: Minimum norm for normalization scaling factor

    """
    def __init__(self, momentum=.99, epsilon=1e-7, *args, **kwargs):
        self.momentum = momentum
        self.epsilon = epsilon
        self.supports_masking = True
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = [1]

        self.norm = self.add_weight(
            name = 'norm', shape=shape, initializer='ones', trainable=False)

    def call(self, inputs, training=False, mask=None):
        if training:
            norm = custom_norm(inputs)
            norm = tf.math.reduce_mean(norm, keepdims=False)
            self.norm.assign(self.momentum*self.norm + (1 - self.momentum)*norm)

        result = inputs/tf.maximum(self.norm, self.epsilon)
        if mask is not None:
            return tf.where(mask, result, inputs)
        return result

    def get_config(self):
        result = super().get_config()
        result['momentum'] = self.momentum
        result['epsilon'] = self.epsilon
        return result

keras.utils.get_custom_objects()['MomentumLayerNormalization'] = MomentumLayerNormalization
