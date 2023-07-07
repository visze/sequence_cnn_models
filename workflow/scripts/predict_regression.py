import click
import pandas as pd

from lib.sequence import SeqRegressionDataLoader1D

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
@click.option(
    "--legnet-model/--no-legnet-model",
    "is_legnet_model",
    required=False,
    default=False,
    help="Switch to define if it is a legnet or a classical tensorflwo model",
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
                'prediction_name',
                required=False,
                default='prediction',
                help='Name of the prediction column')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
def cli(test_file, model_file, weights_file, augment_on, use_augmentation, is_legnet_model, output_file, prediction_name):

    

    if is_legnet_model:
        alphabet = "AGCT"
    else:
        alphabet = "ACGT"
    dl_test = SeqRegressionDataLoader1D(tsv_file=test_file, label_dtype=float, augment=use_augmentation, 
                                        augment_on=augment_on, ignore_targets=True, alphabet=alphabet)



    test_data = dl_test.load_all()

    if is_legnet_model:
        from lib.prediction_legnet import predict as legnet_predict
        preds = legnet_predict(model_file, weights_file, test_data["inputs"])
    else:
        from lib.prediction_tf import predict as tf_predict
        preds = tf_predict(model_file, weights_file, test_data["inputs"])
    
    preds = pd.DataFrame(preds, test_data["metadata"]["id"])
    preds.index.name = "ID"
    for col in preds.columns:
        preds = preds.rename(columns={col: "%s.%d" % (prediction_name,col) })
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
