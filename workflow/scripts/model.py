from tensorflow.keras import Model, activations, layers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling1D, add
from tensorflow.keras.regularizers import l2

def standard(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs) # 250 7 relu
    layer = BatchNormalization()(layer)
    layer = Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer) # 250 8 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = Dropout(0.1)(layer)
    layer = Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer) # 250 3 softmax
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer) # 100 3 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(300, activation='sigmoid')(layer) # 300
    layer = Dropout(0.3)(layer)
    layer = Dense(200, activation='sigmoid')(layer) # 300
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)


def simplified(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(100, kernel_size=8, strides=1, activation='softmax', name="conv1")(inputs) # 250 7 relu
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 8, strides=1, activation='softmax', name="conv2")(layer) # 250 8 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = Dropout(0.1)(layer)
    layer = Conv1D(100, 3, strides=1, activation='softmax', name="conv3")(layer) # 250 3 softmax
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 3, strides=1, activation='softmax', name="conv4")(layer) # 100 3 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(110, activation='sigmoid')(layer) # 300
    layer = Dropout(0.3)(layer)
    layer = Dense(110, activation='sigmoid')(layer) # 300
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)