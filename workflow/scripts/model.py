import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, ReLU, LayerNormalization


def standard(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs)  # 250 7 relu
    layer = BatchNormalization()(layer)
    layer = Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = Dropout(0.1)(layer)
    layer = Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(300, activation='sigmoid')(layer)  # 300
    layer = Dropout(0.3)(layer)
    layer = Dense(200, activation='sigmoid')(layer)  # 300
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return (model)


def simplified(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(100, kernel_size=8, strides=1, activation='softmax', name="conv1")(inputs)  # 250 7 relu
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = Dropout(0.1)(layer)
    layer = Conv1D(100, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 3, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(110, activation='sigmoid')(layer)  # 300
    layer = Dropout(0.3)(layer)
    layer = Dense(110, activation='sigmoid')(layer)  # 300
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return (model)


def saluki(input_shape, output_shape, stochatisc_shift=False, num_layers=5):
    augment_shift = 3
    dropout = 0.3
    kernel_size = 5
    ln_epsilon = 0.007
    l2_scale = 0.0001
    filters = 64
    bn_momentum = 0.90
    initializer = 'he_normal'

    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")

    layer = inputs
    if stochatisc_shift:
        layer = StochasticShift(augment_shift, symmetric=False)(inputs)

    layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid',
                   kernel_initializer=initializer, use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_scale), name="conv")(layer)

    for num in range(num_layers):
        layer = LayerNormalization(epsilon=ln_epsilon, name="layerNorm%d" % num)(layer)
        layer = ReLU(name="relu%d" % num)(layer)
        layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid',
                       kernel_initializer=initializer,
                       kernel_regularizer=tf.keras.regularizers.l2(l2_scale), name="conv%d" % (num+1))(layer)
        layer = Dropout(dropout, name="dropout%d" % num)(layer)
        layer = MaxPooling1D(name="maxpool%d" % num)(layer)

    layer = LayerNormalization(epsilon=ln_epsilon, name="layerNormAggr")(layer)
    layer = ReLU(name="reluAggr")(layer)

    #layer = BatchNormalization(momentum=bn_momentum, name="batchNorm")(layer)

    # penultimate
    layer = Dense(filters, kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(layer)
    layer = Dropout(dropout)(layer)

    # final representation
    layer = BatchNormalization(momentum=bn_momentum, name="batchNorm")(layer)
    layer = ReLU()(layer)

    # compile model
    predictions = Dense(output_shape[1], kernel_initializer=initializer)(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return (model)


class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max+1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                        maxval=len(self.augment_shifts))
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                                lambda: shift_sequence(seq_1hot, shift),
                                lambda: seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shift_max': self.shift_max,
            'symmetric': self.symmetric,
            'pad': self.pad
        })
        return config


def shift_sequence(seq, shift, pad_value=0):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tensorflow.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError('input sequence should be rank 3')
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(tf.greater(shift, 0),
                   lambda: _shift_right(seq),
                   lambda: _shift_left(seq))
    sseq.set_shape(input_shape)

    return sseq
