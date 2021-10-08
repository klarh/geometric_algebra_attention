
import tensorflow as tf
from tensorflow import keras

class MomentumNormalization(keras.layers.Layer):
    """Exponential decay normalization.

    Computes the mean and standard deviation all axes but the last and
    normalizes values to have mean 0 and variance 1; suitable for
    normalizing a vector of real-valued quantities with differing
    units.

    """
    def __init__(self, momentum=.99, epsilon=1e-7, use_mean=True,
                 use_std=True, *args, **kwargs):
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_mean = use_mean
        self.use_std = use_std
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = [input_shape[-1]]

        self.mu = self.add_weight(
            name='mu', shape=shape, initializer='zeros', trainable=False)
        self.sigma = self.add_weight(
            name='sigma', shape=shape, initializer='ones', trainable=False)

    def call(self, inputs, training=False):
        if training:
            axes = range(len(inputs.shape) - 1)
            mean = tf.math.reduce_mean(inputs, axis=axes, keepdims=False)
            std = tf.math.reduce_std(inputs, axis=axes, keepdims=False)
            self.mu.assign(self.momentum*self.mu + (1 - self.momentum)*mean)
            self.sigma.assign(self.momentum*self.sigma + (1 - self.momentum)*std)

        mu = self.mu*tf.cast(self.use_mean, tf.float32)
        use_std = tf.cast(self.use_std, tf.float32)
        denominator = use_std*(self.sigma + self.epsilon) + (1 - use_std)*1.
        return (inputs - mu)/denominator

    def get_config(self):
        result = super().get_config()
        result['momentum'] = self.momentum
        result['epsilon'] = self.epsilon
        result['use_mean'] = self.use_mean
        result['use_std'] = self.use_std
        return result