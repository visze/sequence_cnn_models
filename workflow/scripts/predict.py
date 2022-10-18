import click
import pandas as pd

from sequence import SeqFastaLoader1D

import tensorflow as tf

# options


@click.command()
@click.option('--test',
              'test_fasta_file',
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
def cli(test_fasta_file, model_file, weights_file, output_file, sequence_length):

    strategy = tf.distribute.MirroredStrategy(devices=None)

    dl_test = SeqFastaLoader1D(test_fasta_file, length=sequence_length)

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
        pd.DataFrame(preds, test_data["metadata"]["id"]).to_csv(output_file, sep='\t', header=None, index=True)


if __name__ == '__main__':
    cli()
