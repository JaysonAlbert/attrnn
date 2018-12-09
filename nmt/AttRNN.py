import tensorflow as tf
from tensorflow.python.util.tf_export import tf_export
from tensorflow.contrib.rnn import (LayerRNNCell)
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.rnn_cell_impl import _concat
from tensorflow.python.framework.tensor_util import constant_value_as_shape, constant_value
from tensorflow.contrib.framework import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""

    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        size = tf.random_uniform(c, dtype=dtype)
        return size

    return nest.map_structure(get_state_shape, state_size)


class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


@tf_export("nn.rnn_cell.AttRNNCell")
class AttRNNCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(AttRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = tf.keras.activations.get(activation)
        else:
            self._activation = tf.math.tanh
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._layer_normalization = LayerNormalization(self._num_units)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _add_kernel(self,name, shape):
        return self.add_variable(
            "kernel/%s" % name,
            shape=shape,
            initializer=self._kernel_initializer
        )

    # def zero_state(self, batch_size, dtype):
    #     return tf.random.uniform([batch_size, self._num_units])

    def zero_state(self, batch_size, dtype):
        # shape = constant_value_as_shape(_concat(batch_size, self._num_units))
        if not hasattr(self, '_init_state'):
            # self._init_state = self.add_variable(
            #     'init_state',
            #     shape=shape,
            #     initializer=self._kernel_initializer
            # )
            self._init_state = _zero_state_tensors(self.state_size, batch_size, dtype=dtype)
        return self._init_state

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        input_depth = inputs_shape[-1]
        batch_size = inputs_shape[0]

        self._q_kernel = self._add_kernel('q', [input_depth, 2 * self._num_units])
        self._k_kernel = self._add_kernel('k', [input_depth, self._num_units])
        self._v_kernel = self._add_kernel('v', [input_depth, self._num_units])

        # self.zero_state(batch_size, self._dtype)

        self.built = True

    def call(self, inputs, state):
        q = tf.matmul(inputs, self._q_kernel)
        k = tf.matmul(state, self._k_kernel)
        v = tf.matmul(state, self._v_kernel)

        q = tf.reshape(q,[-1,self._num_units])
        candidate = tf.matmul(q, k, transpose_b=True)
        candidate = tf.nn.softmax(candidate, name="attention_weights")
        candidate = tf.matmul(candidate,v)
        candidate = self._layer_normalization(candidate)

        u, candidate = tf.split(value=candidate, num_or_size_splits=2, axis=0)

        c = self._activation(candidate)
        u = tf.sigmoid(u)

        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "activation": tf.keras.activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(AttRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
