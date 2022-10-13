import numpy as np
import pandas as pd
import gzip
import click

import os

import tensorflow as tf


# options
@click.command()
@click.option(
    "--seq-train",
    "seq_train",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Training sequences",
)
@click.option(
    "--seq-validation",
    "seq_validation",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Validation sequences",
)
@click.option(
    "--labels-train",
    "labels_train",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Labels training",
)
@click.option(
    "--labels-validation",
    "labels_validation",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Labels validation",
)
@click.option(
    "--model",
    "model_file",
    required=True,
    type=click.Path(writable=True),
    help="Model output file",
)
@click.option(
    "--weights",
    "weights_file",
    required=True,
    type=click.Path(writable=True),
    help="Weights output file",
)
@click.option(
    "--fit-log",
    "fit_log_file",
    required=True,
    type=click.Path(writable=True),
    help="Accuracy validation output file",
)
@click.option(
    "--val-acc",
    "acc_file",
    required=True,
    type=click.Path(writable=True),
    help="Accuracy validation output file",
)
@click.option(
    "--val-pred",
    "pred_file",
    required=False,
    type=click.Path(writable=True),
    help="Prediction validation output file",
)
@click.option(
    "--batch-size",
    "batch_size",
    required=False,
    default=32,
    type=int,
    help="Batch size",
)
@click.option(
    "--label-threshold",
    "label_threshold",
    required=False,
    default=0.8,
    type=float,
    help="Positive label threshold",
)
@click.option("--epochs", "epochs", required=True, type=int, help="Number of epochs")
@click.option(
    "--learning-rate", "learning_rate", required=True, type=float, help="Learning rate"
)
@click.option(
    "--use-learning-rate-sheduler/--no-learning-rate-sheduler",
    "learning_rate_sheduler",
    default=False,
    help="Learning rate sheduler",
)
@click.option(
    "--use-early-stopping/--no-early-stopping",
    "early_stopping",
    default=False,
    help="Learning rate",
)
@click.option(
    "--loss",
    "loss",
    type=click.Choice(
        [
            "MSE",
            "Huber",
            "Poission",
            "CategoricalCrossentropy",
        ],
        case_sensitive=False,
    ),
    default="MSE",
    help="Choise of loss function.",
)
@click.option('--seed',
              'seed',
              required=False,
              type=int,
              default=None,
              help='seed for randomness.'
)

def cli(seq_train, seq_validation, labels_train, labels_validation, fit_log_file, model_file, weights_file, acc_file, pred_file, batch_size, label_threshold, epochs, learning_rate, learning_rate_sheduler, early_stopping, loss, seed):
    """
    Train a model for the given sequences and labels.
    """
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # load data

    def loadSingleData(regions,labels):
        X = []
        Y = []
        for region in regions:
            for sequence in utils.SequenceIO.readFasta(region,skipN=False):
                X.append(Encoder.one_hot_encode_along_channel_axis(sequence.getSequence()))
        for label in labels:
            with gzip.open(label, 'rt') as f:
                for line in f:
                    lineSplit = line.split("\t")[6:]
                    if line.startswith("#"):
                        labelNames=lineSplit
                    else:
                        Y.append(list(map(lambda x: 1.0 if float(x)>=label_threshold else 0.0,lineSplit)))

        return(np.array(X),np.array(Y))

    strategy = tf.distribute.MirroredStrategy(devices=None)

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        if epoch > 10:
            learning_rate = 0.00001

        return learning_rate

    # train model
    X_train, Y_train = loadSingleData(seq_train,labels_train)
    X_validation, Y_validation = loadSingleData(seq_validation,labels_validation)

    with strategy.scope():

        inputs = Input(shape=(300,4), name="input")
        layer = Conv1D(250, kernel_size=7, strides=1,activation='relu', name="conv1")(inputs)
        layer = BatchNormalization()(layer)
        layer = Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
        layer = Dropout(0.1)(layer)
        layer = Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)
        layer = BatchNormalization()(layer)
        layer = Conv1D(100, 2, strides=1, activation='softmax',name="conv4")(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
        layer = Dropout(0.1)(layer)
        layer = Flatten()(layer)
        layer = Dense(300, activation='sigmoid')(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(200, activation='sigmoid')(layer)
        predictions = Dense(np.shape(train_Y)[1], activation='linear')(layer)
        model = Model(inputs=inputs, outputs=predictions)

        # defining callbacks
        call_backs = []

        csvLogger = callbacks.CSVLogger(fit_log_file, separator="\t", append=False)

        call_backs.append(csvLogger)

        lr_callback = callbacks.LearningRateScheduler(lr_schedule)
        if learning_rate_sheduler:
            call_backs.append(lr_callback)

        earlyStopping_callback = callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        if early_stopping:
            call_backs.append(earlyStopping_callback)

        optimizer = optimizers.Adam(learning_rate=learning_rate)

        if loss == "Poission":  # use_poisson_loss:
            model.compile(
                loss=tf.keras.losses.Poisson(),
                metrics=["mse", "mae", "mape", "acc"],
                optimizer=optimizer,
            )
        elif loss == "Huber":  # use_huber_loss:
            model.compile(
                loss=tf.keras.losses.huber(),
                metrics=["mse", "mae", "mape", "acc"],
                optimizer=optimizer,
            )
        elif loss == "CategoricalCrossentropy":  # use_cat_ent:
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["mse", "mae", "mape", "acc", "categorical_accuracy"],
                optimizer=optimizer,
            )
        else:
            model.compile(
                loss="mean_squared_error",
                metrics=["mse", "mae", "mape", "acc"],
                optimizer=optimizer,
            )

        print("Fit model")

        result = model.fit(
            X_train,
            Y_train,
            validation_data=(X_validation, Y_validation),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=2,
            callbacks=call_backs,
        )

        print("Save_model")

        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(weights_file)

        if pred_file:
            print("Final prediction")
            preds = model.predict(X_validation)
            pd.DataFrame(preds).to_csv(pred_file, sep="\t", index=False)

        print("Final evaluation")
        eval = model.evaluate(X_validation, Y_validation),
        pd.DataFrame(eval).to_csv(acc_file, sep="\t", index=False, header=None)

        print("Done")
