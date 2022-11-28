from tensorflow.keras import Model, activations, layers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, ReLU, LayerNormalization

from tensorflow.keras.regularizers import L2


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


def saluki(input_shape, output_shape, stochatisc_shift=False, conv_size=6):
    augment_shift = 3
    dropout = 0.3
    epsilon = 0.007
    l2_scale = 0.0001
    filters = 64
    bn_momentum = 0.90

    def addLayer(input, num):
        layer = Conv1D(filters, kernel_size=5, kernel_regularizers=L2(l2=l2_scale),
                       strides=1, activation=None, name="conv%d" % num)(input)
        layer = Dropout(dropout, name="dropout%d" % num)(layer)
        layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
        layer = LayerNormalization(epsilon=epsilon, name="layerNorm%d" % num)(layer)
        layer = ReLU(name="relu%d" % num)(layer)
        return (layer)

    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    if stochatisc_shift:
        layer = StochasticShift(augment_shift, symmetric=False)(inputs)
    else:
        layer = inputs
    for num in range(1, conv_size+1):
        layer = addLayer(layer, num)

    layer = Dense(64,)(layer)
    layer = Dropout(dropout, name="dropout%d" % conv_size+1)(layer)
    layer = BatchNormalization(momentum=bn_momentum, name="batchNorm")(layer)
    layer = ReLU(name="relu%d " % conv_size+1)(layer)
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return (model)


class StochasticShift(tensorflow.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tensorflow.range(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = tensorflow.range(0, self.shift_max+1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tensorflow.random.uniform(shape=[], minval=0, dtype=tensorflow.int64,
                                                maxval=len(self.augment_shifts))
            shift = tensorflow.gather(self.augment_shifts, shift_i)
            sseq_1hot = tensorflow.cond(tensorflow.not_equal(shift, 0),
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
    shift: signed shift value (tensorflow.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tensorflow.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError('input sequence should be rank 3')
    input_shape = seq.shape

    pad = pad_value * tensorflow.ones_like(seq[:, 0:tensorflow.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tensorflow.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tensorflow.concat([sliced_seq, pad], axis=1)

    sseq = tensorflow.cond(tensorflow.greater(shift, 0),
                           lambda: _shift_right(seq),
                           lambda: _shift_left(seq))
    sseq.set_shape(input_shape)

    return sseq
