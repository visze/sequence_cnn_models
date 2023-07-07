import click
import pandas as pd

from lib.sequence import SeqFastaLoader1D, SeqRegressionDataLoader1D

import tensorflow as tf

# options


@click.command()
@click.option('--test-fasta-file',
              'test_fasta_file',
              required=False,
              type=click.Path(exists=True, readable=True),
              help='Test Fasta sequences')
@click.option('--test-file',
              'test_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Test sequences')
@click.option('--model',
              'model_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Model file')
@click.option('--weights',
              'weights_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Weights file')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option("--sequence-length",
              "sequence_length",
              required=True,
              type=int,
              help="Length of the sequence"
              )
@click.option(
    "--use-augmentation/--no-augmentation",
    "use_augmentation",
    required=False,
    default=False,
    help="Augment data using reverse complement",
)
@click.option(
    "--augment-on",
    "augment_on",
    required=False,
    type=(int, int),
    help="Augment data using reverse complement from given start to stop.",
)
@click.option('--prediction-name',
              'prediction_names',
              required=True,
              multiple=True,
              help='Name of the prediction column')
def cli(test_fasta_file, test_file, model_file, weights_file, augment_on, use_augmentation, output_file, sequence_length, prediction_names):

    strategy = tf.distribute.MirroredStrategy(devices=None)

    if (test_fasta_file):
        dl_test = SeqFastaLoader1D(test_fasta_file, length=sequence_length, augment=use_augmentation, augment_on=augment_on,)
    else:
        dl_test = SeqRegressionDataLoader1D(tsv_file=test_file, label_dtype=float, augment=use_augmentation, augment_on=augment_on, ignore_targets=True)

    test_data = dl_test.load_all()

    with strategy.scope():

        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        print(f"Model loaded from: {model_file} and {weights_file}")

        print("Final prediction")
        preds = model.predict(test_data["inputs"])
        preds = pd.DataFrame(preds, test_data["metadata"]["id"])
        preds.index.name = "ID"
        for col in preds.columns:
            preds = preds.rename(columns={col: prediction_names[col]})
        print(preds.head())
        agg = {}
        for col in preds.columns:
            agg[col] = 'mean'
        preds = preds.groupby(level=0)
        print(preds.head())
        preds = preds.agg(agg)
        print(preds.head())
        preds.to_csv(output_file, sep='\t', header=True, index=True)


if __name__ == '__main__':
    cli()
