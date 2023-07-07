#!/bin/env python

import tensorflow
import deepexplain

import click
import csv
import logging
import gzip
import numpy as np

# options


@click.command()
@click.option('--sequence',
              'sequence_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Test sequences')
@click.option('--reference',
                'reference_file',
                required=True,
                type=click.Path(exists=True, readable=True),
                help='Reference sequences')
@click.option('--background',
              'background_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='background sequences')
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
@click.option('--input-layer',
              'input_layer',
              required=True,
              type=str,
              help='Name of the input layer')
@click.option('--log',
              'log_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option('--output-intgrad',
              'output_file_intgrad',
              required=True,
              type=click.Path(writable=True),
              help='Integrad output file')
@click.option('--output-deeplift',
              'output_file_deeplift',
              required=True,
              type=click.Path(writable=True),
              help='Deeplift output file')
def cli(sequence_file, reference_file, background_file, model_file, weights_file, input_layer, log_file, output_file_intgrad, output_file_deeplift):

    from lib.sequence import SeqFastaLoader1D, SeqClassificationDataLoader1D

    from tensorflow.keras import Model
    import tensorflow as tf
    
    task = 0
    n_tasks = 2

    logging.basicConfig(filename=log_file, level=logging.DEBUG)



    dl_backgorund = SeqClassificationDataLoader1D(intervals_file=background_file, fasta_file=reference_file, label_dtype=int, ignore_targets=True)
    dl_tests = SeqFastaLoader1D(sequence_file, length=230)

    background = dl_backgorund.load_all()
    x_tests = dl_tests.load_all()

    from keras.models import Model
    from keras import backend as K

    from keras.utils import multi_gpu_model

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                        intra_op_parallelism_threads=25,
                                        inter_op_parallelism_threads=25))

    # TF_CONFIG =  K.tf.ConfigProto(intra_op_parallelism_threads=25,
    #                           inter_op_parallelism_threads=25)
    #K.tf.Session(config=TF_CONFIG)
    # K.set_session(sess)

    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights_file)

    print(model.summary())

    try:
        keras_model = multi_gpu_model(model, gpus=2)
        logging.debug("Using multiple GPUs..")
    except ValueError:
        keras_model = model
        logging.debug("Using single GPU or CPU..")

    y_tests = []
    for test in x_tests["inputs"]:
        y = np.zeros(n_tasks, dtype=np.int8)
        y[task] = 1
        y_tests.append(y)
    y_tests = np.array(y_tests)

    with deepexplain.tensorflow.DeepExplain(session=sess) as de:
        input_tensor = keras_model.get_layer(input_layer).input
        fModel = Model(inputs=input_tensor, outputs=keras_model.get_layer('dense_3').output)
        target_tensor = fModel(input_tensor)
        print(x_tests["inputs"].shape)
        i = 0
        elements = 100
        attributions = {}
        while (i < x_tests["inputs"].shape[0]):
            xs = x_tests["inputs"][i:i+elements, :, :]
            print(xs.shape)
            ys = y_tests[i:i+elements, :]

            result = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys, baseline=background["inputs"])
            if ('deeplift' in attributions):
                attributions['deeplift'] = np.append(attributions['deeplift'], result, axis=0)
            else:
                attributions['deeplift'] = result
            result = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys, baseline=background["inputs"])
            if ('intgrad' in attributions):
                attributions['intgrad'] = np.append(attributions['intgrad'], result, axis=0)
            else:
                attributions['intgrad'] = result

            i += elements

    logging.info("Writing scores for task for task %d" % task)

    for method_name in attributions.keys():
        file_path = output_file_deeplift if method_name == 'deeplift' else output_file_intgrad
        with gzip.open(file_path, 'wt') as score_file:
            for idx in range(np.shape(attributions[method_name])[0]):
                scores = attributions[method_name][idx]

                score_writer = csv.writer(score_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                score_writer.writerow(scores.flatten().tolist())

if __name__ == "__main__":
    cli()
