import click
import pandas as pd
import numpy as np

from sequence import SeqRegressionDataLoader1D

import tensorflow as tf

# options


@click.command()
@click.option('--test',
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
@click.option('--prediction-name',
                'prediction_name',
                required=False,
                default='prediction',
                help='Name of the prediction column')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
def cli(test_file, model_file, weights_file, output_file, prediction_name):

    strategy = tf.distribute.MirroredStrategy(devices=None)

    dl_test = SeqRegressionDataLoader1D(tsv_file=test_file, label_dtype=float, augment=True, augment_on=(16,215), ignore_targets=True)

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
        # FIXME strange workaround. why saluki model has 3 outputs. Whar are they?
        if len(np.shape(preds)) == 3:
            preds = preds[:,0,0]
        preds = pd.DataFrame(preds, test_data["metadata"]["id"])
        preds.index.name = "ID"
        preds = preds.rename(columns={0: prediction_name})
        print(preds.head())
        preds = preds.groupby(level=0)
        print(preds.head())
        preds = preds.agg({prediction_name:'mean'})
        print(preds.head())
        preds.to_csv(output_file, sep='\t', header=True, index=True)


if __name__ == '__main__':
    cli()
