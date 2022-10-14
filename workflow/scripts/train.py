import numpy as np
import pandas as pd
import click

from sequence import SeqCNNDataLoader1D

from model import standard, simplified

import tensorflow as tf

model_type = {
    "standard": standard,
    "simplified": simplified,
}

# options
@click.command()
@click.option(
    "--fasta-file",
    "fasta_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Genome Fasta file",
)
@click.option(
    "--intervals-train",
    "intervals_train",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Intervals and labels for training",
)
@click.option(
    "--intervals-validation",
    "intervals_validation",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Intervals and labels for validation",
)
@click.option(
    "--model-type",
    "model_type_str",
    default="standard",
    type=click.Choice(model_type.keys(), case_sensitive=False),
    help="The model that should be used.",
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
def cli(
    fasta_file, intervals_train, intervals_validation, model_type_str, fit_log_file, model_file, weights_file, acc_file, pred_file, batch_size, epochs, learning_rate, learning_rate_sheduler, early_stopping, loss, seed
):
    """
    Train a model for the given sequences and labels.
    """
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    strategy = tf.distribute.MirroredStrategy(devices=None)

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        if epoch > 10:
            learning_rate = 0.00001

        return learning_rate

    dl = SeqCNNDataLoader1D(intervals_file=intervals_train, fasta_file=fasta_file, label_dtype=int)

    dl_val = SeqCNNDataLoader1D(intervals_file=intervals_validation, fasta_file=fasta_file, label_dtype=int)

    # train model
    train_data = dl.load_all()
    val_data = dl_val.load_all()

    with strategy.scope():

        model = model_type[model_type_str](
            val_data["inputs"].shape, val_data["targets"].shape
        )

        # defining callbacks
        call_backs = []

        csvLogger = tf.keras.callbacks.CSVLogger(fit_log_file, separator="\t", append=False)

        call_backs.append(csvLogger)

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        if learning_rate_sheduler:
            call_backs.append(lr_callback)

        earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        if early_stopping:
            call_backs.append(earlyStopping_callback)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
            train_data["inputs"],
            train_data["targets"],
            validation_data=(val_data["inputs"], val_data["targets"]),
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
            preds = model.predict(val_data["inputs"])
            pd.DataFrame(preds).to_csv(pred_file, sep="\t", index=False)

        print("Final evaluation")
        eval = model.evaluate(val_data["inputs"], val_data["targets"])
        pd.DataFrame(eval).to_csv(acc_file, sep="\t", index=False, header=None)

        print("Done")

if __name__ == "__main__":
    cli()