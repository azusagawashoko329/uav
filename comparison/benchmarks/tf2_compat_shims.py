#!/usr/bin/env python3
"""TF1-contrib API shims for running the official Future Frame / MLEP graphs on TF2.

The two official repos are TensorFlow 1.x graph-mode code.  Their TF1.15 wheels
are pinned to CUDA 10.0 / cuDNN 7.6.5, which has no usable cuBLAS/cuDNN kernels
for the RTX 5880 Ada (sm_89): session creation and elementwise ops work, but
matmul/conv hang.  This module keeps the model code unchanged and supplies the
small set of TF1 contrib APIs it imports, implemented with TF2 raw ops.

Only API plumbing is replaced:
  * tensorflow.contrib.layers.{conv2d,conv3d,conv2d_transpose,max_pool2d,
    flatten,dropout,fully_connected,instance_norm,l2_regularizer}
  * tensorflow.contrib.framework.arg_scope
  * tensorflow.contrib.rnn.ConvLSTMCell

The ConvLSTMCell math and variable shapes follow TF1's contrib implementation
(kernel + biases, four gates, forget bias); MLEP's parameter count matches the
TF1 graph exactly (12,560,835), which is the acceptance check for this shim.

This is a *runtime adaptation*, not the original TF1.15 runtime.  Timing from
it must be labelled runtime-adapted.
"""
from __future__ import annotations

import types

_TF = None
_SCOPE_STACK = []


class _ArgScope:
    def __init__(self, scopes):
        self.scopes = scopes

    def __enter__(self):
        _SCOPE_STACK.append(self.scopes)
        return self

    def __exit__(self, *exc):
        _SCOPE_STACK.pop()
        return False


def arg_scope(list_ops_or_scope, **kwargs):
    if isinstance(list_ops_or_scope, _ArgScope):
        scopes = dict(list_ops_or_scope.scopes)
    else:
        scopes = {op: dict(kwargs) for op in list_ops_or_scope}
    return _ArgScope(scopes)


def _defaults(fn):
    out = {}
    for scopes in _SCOPE_STACK:
        for op, kw in scopes.items():
            if op is fn:
                out.update(kw)
    return out


_UNSET = object()


def _pair(v):
    return list(v) if isinstance(v, (list, tuple)) else [v, v]


def _triple(v):
    return list(v) if isinstance(v, (list, tuple)) else [v, v, v]


def instance_norm(inputs, center=True, scale=True, epsilon=1e-6,
                  activation_fn=None, params=None, reuse=None, scope=None, **kw):
    tf = _TF
    with tf.compat.v1.variable_scope(scope, 'InstanceNorm', [inputs], reuse=reuse):
        rank = inputs.shape.ndims
        axes = list(range(1, rank - 1))
        mean, var = tf.nn.moments(inputs, axes=axes, keepdims=True)
        x = (inputs - mean) * tf.math.rsqrt(var + epsilon)
        ch = int(inputs.shape[-1])
        if scale:
            gamma = tf.compat.v1.get_variable(
                'gamma', [ch], initializer=tf.compat.v1.ones_initializer())
            x = x * gamma
        if center:
            beta = tf.compat.v1.get_variable(
                'beta', [ch], initializer=tf.compat.v1.zeros_initializer())
            x = x + beta
        if activation_fn is not None:
            x = activation_fn(x)
    return x


def l2_regularizer(scale):
    tf = _TF

    def _reg(x):
        loss = scale * tf.nn.l2_loss(x)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, loss)
        return loss

    return _reg


def conv2d(inputs, num_outputs, kernel_size=_UNSET, stride=_UNSET, padding=_UNSET,
           activation_fn=_UNSET, normalizer_fn=_UNSET, normalizer_params=_UNSET,
           weights_initializer=_UNSET, biases_initializer=None,
           weights_regularizer=_UNSET, scope=None, reuse=None, **kw):
    tf = _TF
    opts = _defaults(conv2d)
    kernel_size = opts.get('kernel_size', 3) if kernel_size is _UNSET else kernel_size
    stride = opts.get('stride', 1) if stride is _UNSET else stride
    padding = opts.get('padding', 'SAME') if padding is _UNSET else padding
    activation_fn = opts.get('activation_fn', tf.nn.relu) if activation_fn is _UNSET else activation_fn
    normalizer_fn = opts.get('normalizer_fn', None) if normalizer_fn is _UNSET else normalizer_fn
    normalizer_params = opts.get('normalizer_params', None) if normalizer_params is _UNSET else normalizer_params
    weights_initializer = opts.get('weights_initializer', None) if weights_initializer is _UNSET else weights_initializer
    weights_regularizer = opts.get('weights_regularizer', None) if weights_regularizer is _UNSET else weights_regularizer
    k = _pair(kernel_size)
    s = _pair(stride)
    in_ch = int(inputs.shape[-1])
    with tf.compat.v1.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
        init = weights_initializer or tf.compat.v1.glorot_uniform_initializer()
        w = tf.compat.v1.get_variable(
            'weights', k + [in_ch, num_outputs], initializer=init)
        if weights_regularizer is not None:
            weights_regularizer(w)
        x = tf.nn.conv2d(inputs, w, [1, s[0], s[1], 1], padding.upper())
        if normalizer_fn is not None:
            x = normalizer_fn(x, **(normalizer_params or {}))
        else:
            b = tf.compat.v1.get_variable(
                'biases', [num_outputs], initializer=tf.compat.v1.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        if activation_fn is not None:
            x = activation_fn(x)
    return x


def conv3d(inputs, num_outputs, kernel_size=_UNSET, stride=_UNSET, padding=_UNSET,
           activation_fn=_UNSET, normalizer_fn=_UNSET, normalizer_params=_UNSET,
           weights_initializer=_UNSET, biases_initializer=None,
           weights_regularizer=_UNSET, scope=None, reuse=None, **kw):
    tf = _TF
    opts = _defaults(conv3d)
    kernel_size = opts.get('kernel_size', 3) if kernel_size is _UNSET else kernel_size
    stride = opts.get('stride', 1) if stride is _UNSET else stride
    padding = opts.get('padding', 'SAME') if padding is _UNSET else padding
    activation_fn = opts.get('activation_fn', tf.nn.relu) if activation_fn is _UNSET else activation_fn
    normalizer_fn = opts.get('normalizer_fn', None) if normalizer_fn is _UNSET else normalizer_fn
    normalizer_params = opts.get('normalizer_params', None) if normalizer_params is _UNSET else normalizer_params
    weights_initializer = opts.get('weights_initializer', None) if weights_initializer is _UNSET else weights_initializer
    weights_regularizer = opts.get('weights_regularizer', None) if weights_regularizer is _UNSET else weights_regularizer
    k = _triple(kernel_size)
    s = _triple(stride)
    in_ch = int(inputs.shape[-1])
    with tf.compat.v1.variable_scope(scope, 'Conv3D', [inputs], reuse=reuse):
        init = weights_initializer or tf.compat.v1.glorot_uniform_initializer()
        w = tf.compat.v1.get_variable(
            'weights', k + [in_ch, num_outputs], initializer=init)
        if weights_regularizer is not None:
            weights_regularizer(w)
        x = tf.nn.conv3d(inputs, w, [1, s[0], s[1], s[2], 1], padding.upper())
        if normalizer_fn is not None:
            x = normalizer_fn(x, **(normalizer_params or {}))
        else:
            b = tf.compat.v1.get_variable(
                'biases', [num_outputs], initializer=tf.compat.v1.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        if activation_fn is not None:
            x = activation_fn(x)
    return x


def conv2d_transpose(inputs, num_outputs, kernel_size=_UNSET, stride=_UNSET, padding=_UNSET,
                     activation_fn=_UNSET, normalizer_fn=_UNSET, normalizer_params=_UNSET,
                     weights_initializer=_UNSET, biases_initializer=None,
                     weights_regularizer=_UNSET, scope=None, reuse=None, **kw):
    tf = _TF
    opts = _defaults(conv2d_transpose)
    kernel_size = opts.get('kernel_size', 3) if kernel_size is _UNSET else kernel_size
    stride = opts.get('stride', 2) if stride is _UNSET else stride
    padding = opts.get('padding', 'SAME') if padding is _UNSET else padding
    activation_fn = opts.get('activation_fn', tf.nn.relu) if activation_fn is _UNSET else activation_fn
    normalizer_fn = opts.get('normalizer_fn', None) if normalizer_fn is _UNSET else normalizer_fn
    normalizer_params = opts.get('normalizer_params', None) if normalizer_params is _UNSET else normalizer_params
    weights_initializer = opts.get('weights_initializer', None) if weights_initializer is _UNSET else weights_initializer
    weights_regularizer = opts.get('weights_regularizer', None) if weights_regularizer is _UNSET else weights_regularizer
    k = _pair(kernel_size)
    s = _pair(stride)
    in_ch = int(inputs.shape[-1])
    with tf.compat.v1.variable_scope(scope, 'Conv2dTranspose', [inputs], reuse=reuse):
        init = weights_initializer or tf.compat.v1.glorot_uniform_initializer()
        w = tf.compat.v1.get_variable(
            'weights', k + [num_outputs, in_ch], initializer=init)
        if weights_regularizer is not None:
            weights_regularizer(w)
        shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1] * s[0],
                          tf.shape(inputs)[2] * s[1], num_outputs])
        x = tf.nn.conv2d_transpose(inputs, w, shape, [1, s[0], s[1], 1], padding.upper())
        if normalizer_fn is not None:
            x = normalizer_fn(x, **(normalizer_params or {}))
        else:
            b = tf.compat.v1.get_variable(
                'biases', [num_outputs], initializer=tf.compat.v1.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        if activation_fn is not None:
            x = activation_fn(x)
    return x


def max_pool2d(inputs, kernel_size, stride=None, padding='VALID', scope=None, **kw):
    tf = _TF
    k = _pair(kernel_size)
    s = _pair(stride if stride is not None else kernel_size)
    with tf.compat.v1.variable_scope(scope, 'MaxPool', [inputs]):
        return tf.nn.max_pool2d(
            inputs, [1, k[0], k[1], 1], [1, s[0], s[1], 1], padding.upper())


def flatten(inputs, scope=None, **kw):
    tf = _TF
    with tf.compat.v1.variable_scope(scope, 'Flatten', [inputs]):
        return tf.reshape(inputs, [tf.shape(inputs)[0], -1])


def dropout(inputs, keep_prob=0.5, is_training=True, scope=None, **kw):
    tf = _TF
    opts = _defaults(dropout)
    keep_prob = opts.get('keep_prob', keep_prob)
    is_training = opts.get('is_training', is_training)
    with tf.compat.v1.variable_scope(scope, 'Dropout', [inputs]):
        if not is_training:
            return inputs
        return tf.nn.dropout(inputs, rate=1.0 - keep_prob)


def fully_connected(inputs, num_outputs, activation_fn=_UNSET,
                    weights_initializer=_UNSET, biases_initializer=None,
                    weights_regularizer=_UNSET, scope=None, reuse=None, **kw):
    tf = _TF
    opts = _defaults(fully_connected)
    activation_fn = opts.get('activation_fn', tf.nn.relu) if activation_fn is _UNSET else activation_fn
    weights_initializer = opts.get('weights_initializer', None) if weights_initializer is _UNSET else weights_initializer
    weights_regularizer = opts.get('weights_regularizer', None) if weights_regularizer is _UNSET else weights_regularizer
    in_ch = int(inputs.shape[-1])
    with tf.compat.v1.variable_scope(scope, 'FullyConnected', [inputs], reuse=reuse):
        init = weights_initializer or tf.compat.v1.glorot_uniform_initializer()
        w = tf.compat.v1.get_variable('weights', [in_ch, num_outputs], initializer=init)
        if weights_regularizer is not None:
            weights_regularizer(w)
        b = tf.compat.v1.get_variable(
            'biases', [num_outputs], initializer=tf.compat.v1.zeros_initializer())
        x = tf.matmul(inputs, w) + b
        if activation_fn is not None:
            x = activation_fn(x)
    return x


class _LSTMStateTuple(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        return tuple.__new__(cls, args)

    @property
    def c(self):
        return self[0]

    @property
    def h(self):
        return self[1]


class ConvLSTMCell:
    """TF1 contrib ConvLSTMCell (2-D) reimplemented with raw TF ops."""

    def __init__(self, conv_ndims, input_shape, output_channels, kernel_shape,
                 use_bias=True, skip_connection=False, forget_bias=1.0,
                 initializers=None, name='conv_lstm_cell'):
        if conv_ndims != 2:
            raise ValueError('This shim only implements conv_ndims=2')
        self._conv_ndims = conv_ndims
        self._input_shape = list(input_shape)
        self._output_channels = output_channels
        self._kernel_shape = list(kernel_shape)
        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._skip_connection = skip_connection
        state_shape = self._input_shape[:-1] + [self._output_channels]
        self._state_size = (_TF.TensorShape(state_shape), _TF.TensorShape(state_shape))
        self._output_size = _TF.TensorShape(state_shape)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        shape = [batch_size] + self._input_shape[:-1] + [self._output_channels]
        return _LSTMStateTuple(_TF.zeros(shape, dtype), _TF.zeros(shape, dtype))

    def __call__(self, inputs, state, scope=None):
        tf = _TF
        cell, hidden = state
        with tf.compat.v1.variable_scope(scope or 'conv_lstm_cell'):
            in_ch = int(inputs.shape[-1]) + int(hidden.shape[-1])
            kernel = tf.compat.v1.get_variable(
                'kernel', self._kernel_shape + [in_ch, 4 * self._output_channels])
            x = tf.concat([inputs, hidden], axis=-1)
            gates = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
            if self._use_bias:
                bias = tf.compat.v1.get_variable(
                    'biases', [4 * self._output_channels],
                    initializer=tf.compat.v1.zeros_initializer())
                gates = tf.nn.bias_add(gates, bias)
            i, j, f, o = tf.split(gates, 4, axis=3)
            new_cell = (tf.sigmoid(f + self._forget_bias) * cell +
                        tf.sigmoid(i) * tf.tanh(j))
            output = tf.tanh(new_cell) * tf.sigmoid(o)
            if self._skip_connection:
                output = tf.concat([output, inputs], axis=-1)
        return output, _LSTMStateTuple(new_cell, output)


def install(tf):
    """Install the contrib shims on the given TF module.  Returns the modules."""
    global _TF
    _TF = tf

    # TF2.20 dropped many TF1 module-level aliases; the repos use these directly.
    for name in [
        'variable_scope', 'placeholder', 'trainable_variables',
        'global_variables_initializer', 'Session', 'Graph', 'get_variable',
        'add_to_collection', 'GraphKeys', 'random_normal_initializer',
        'truncated_normal_initializer', 'constant_initializer',
        'zeros_initializer', 'ones_initializer', 'AUTO_REUSE', 'name_scope',
        'control_dependencies', 'get_collection', 'assign',
        'initialize_all_variables',
    ]:
        if not hasattr(tf, name) and hasattr(tf.compat.v1, name):
            setattr(tf, name, getattr(tf.compat.v1, name))

    layers_mod = types.ModuleType('tensorflow.contrib.layers')
    for name, fn in [
        ('conv2d', conv2d), ('conv3d', conv3d),
        ('conv2d_transpose', conv2d_transpose), ('max_pool2d', max_pool2d),
        ('flatten', flatten), ('dropout', dropout),
        ('fully_connected', fully_connected), ('instance_norm', instance_norm),
        ('l2_regularizer', l2_regularizer),
    ]:
        setattr(layers_mod, name, fn)

    framework_mod = types.ModuleType('tensorflow.contrib.framework')
    framework_mod.arg_scope = arg_scope

    rnn_mod = types.ModuleType('tensorflow.contrib.rnn')
    rnn_mod.ConvLSTMCell = ConvLSTMCell

    contrib = types.ModuleType('tensorflow.contrib')
    contrib.layers = layers_mod
    contrib.framework = framework_mod
    contrib.rnn = rnn_mod

    import sys
    sys.modules['tensorflow.contrib'] = contrib
    sys.modules['tensorflow.contrib.layers'] = layers_mod
    sys.modules['tensorflow.contrib.framework'] = framework_mod
    sys.modules['tensorflow.contrib.rnn'] = rnn_mod
    setattr(tf, 'contrib', contrib)

    if not hasattr(tf.nn, 'dynamic_rnn'):
        tf.nn.dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
    if not hasattr(tf.image, 'resize_nearest_neighbor'):
        tf.image.resize_nearest_neighbor = tf.compat.v1.image.resize_nearest_neighbor
    if not hasattr(tf.image, 'resize_bilinear'):
        tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear

    return layers_mod, framework_mod, rnn_mod
