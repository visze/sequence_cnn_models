import click
import csv
import logging
import gzip
import numpy as np
import deeplift
from sequence import SeqFastaLoader1D, SeqClassificationDataLoader1D
# options

methods = [
    'grad_times_inp',
    'guided_backprop',
    'integrated_gradients10',
    'rescale_all_layers', 'revealcancel_all_layers',
    'rescale_conv_revealcancel_fc',
    'rescale_conv_revealcancel_fc_multiref_10'
]

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
@click.option('--output',
              'output_file',
              required=True,
              multiple=True,
              type=(click.Choice(methods.keys(), case_sensitive=True),click.Path(writable=True)),
              help='Integrad output file')
def cli(sequence_file, reference_file, background_file, model_file, weights_file, input_layer, log_file, output_file):

    import tensorflow as tf

    task_idx = 0
    n_tasks = 2

    logging.basicConfig(filename=str(log_file), level=logging.DEBUG)

    # load onhot data
    logging.debug("Load sequence data")
    dl_backgorund = SeqClassificationDataLoader1D(intervals_file=background_file,
                                                  fasta_file=reference_file, label_dtype=int, ignore_targets=True)
    dl_tests = SeqFastaLoader1D(sequence_file, length=230)

    background = dl_backgorund.load_all()
    onehot_data = dl_tests.load_all()

    # load the keras model
    logging.debug("Load keras model")

    strategy = tf.distribute.MirroredStrategy(devices=None)

    with strategy.scope():

        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(weights_file)

        print(model.summary())

        # Prepare the deeplift models
        # Model conversion
        logging.debug("Convert Keras model to deepfift model")
        from deeplift.layers import NonlinearMxtsMode
        import deeplift.conversion.kerasapi_conversion as kc
        from collections import OrderedDict

        print("check3")

        method_to_model = OrderedDict()
        for method_name, nonlinear_mxts_mode in [
            # The genomics default = rescale on conv layers, revealcance on fully-connected
            ('rescale_conv_revealcancel_fc', NonlinearMxtsMode.DeepLIFT_GenomicsDefault),
            ('rescale_all_layers', NonlinearMxtsMode.Rescale),
            ('revealcancel_all_layers', NonlinearMxtsMode.RevealCancel),
            ('grad_times_inp', NonlinearMxtsMode.Gradient),
            ('guided_backprop', NonlinearMxtsMode.GuidedBackprop)
        ]:
            method_to_model[method_name] = kc.convert_model_from_saved_files(
                h5_file=json_file,
                json_file=model_file,
                nonlinear_mxts_mode=nonlinear_mxts_mode)
        # sanity checks
        # make sure predictions are the same as the original model
        logging.debug("Sanity checks of deeplift model")
        from deeplift.util import compile_func
        model_to_test = method_to_model['rescale_conv_revealcancel_fc']
        print(model_to_test.get_name_to_layer())
        logging.debug(model_to_test.get_name_to_layer())
        deeplift_prediction_func = compile_func([model_to_test.get_name_to_layer(
        )[input_layer].get_activation_vars()], model_to_test.get_name_to_layer()['dense_3_0'].get_activation_vars())
        original_model_predictions = model.predict(onehot_data["inputs"], batch_size=200)
        converted_model_predictions = deeplift.util.run_function_in_batches(
            input_data_list=[onehot_data["inputs"]],
            func=deeplift_prediction_func,
            batch_size=200,
            progress_update=None)
        logging.info("maximum difference in predictions: %f " % np.max(
            np.array(converted_model_predictions)-np.array(original_model_predictions)))
        assert np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)) < 10**-5
        predictions = converted_model_predictions

        # Compute importance scores
        # Compile various scoring functions

        logging.info("Compiling scoring functions")
        method_to_scoring_func = OrderedDict()
        for method, model in method_to_model.items():
            logging.info("Compiling scoring function for: "+method)
            method_to_scoring_func[method] = model.get_target_contribs_func(find_scores_layer_name=input_layer,
                                                                            pre_activation_target_layer_name='dense_3_0')

        # To get a function that just gives the gradients, we use the multipliers of the Gradient model
        gradient_func = method_to_model['grad_times_inp'].get_target_multipliers_func(
            find_scores_layer_name=input_layer, pre_activation_target_layer_name='dense_3_0')
        logging.info("Compiling integrated gradients scoring functions")
        integrated_gradients10_func = deeplift.util.get_integrated_gradients_function(
            gradient_computation_function=gradient_func,
            num_intervals=10)
        method_to_scoring_func['integrated_gradients10'] = integrated_gradients10_func

        # Call scoring functions on the data
        logging.debug("Call scoring functions on data")
        # background = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3)])

        from collections import OrderedDict
        method_to_task_to_scores = OrderedDict()
        for method_name, score_func in method_to_scoring_func.items():
            logging.info("on method %s" % method_name)
            method_to_task_to_scores[method_name] = OrderedDict()
            # scores = np.array(score_func(
            #     task_idx=task_idx,
            #     input_data_list=[[onehot_data["inputs"]]],
            #     input_references_list=[
            #         np.array([background['A'],
            #                   background['C'],
            #                   background['G'],
            #                   background['T']])[None, None, :]],
            #     batch_size=200,
            #     progress_update=None))
            scores = np.array(score_func(
                task_idx=task_idx,
                input_data_list=[[onehot_data["inputs"]]],
                input_references_list=[np.array(background["inputs"])[None, :, :]],
                batch_size=200,
                progress_update=None))
            assert scores.shape[2] == 4
            # The sum over the ACGT axis in the code below is important! Recall that DeepLIFT
            # assigns contributions based on difference-from-reference; if
            # a position is [1,0,0,0] (i.e. 'A') in the actual sequence and [0.3, 0.2, 0.2, 0.3]
            # in the reference, importance will be assigned to the difference (1-0.3)
            # in the 'A' channel, (0-0.2) in the 'C' channel,
            # (0-0.2) in the G channel, and (0-0.3) in the T channel. You want to take the importance
            # on all four channels and sum them up, so that at visualization-time you can project the
            # total importance over all four channels onto the base that is actually present (i.e. the 'A'). If you
            # don't do this, your visualization will look very confusing as multiple bases will be highlighted at
            # every position and you won't know which base is the one that is actually present in the sequence!
            scores = np.sum(scores, axis=2)
            method_to_task_to_scores[method_name] = scores

        logging.debug("Using multiple shuffled references")
        from deeplift.util import get_shuffle_seq_ref_function
        #from deeplift.util import randomly_shuffle_seq
        from deeplift.dinuc_shuffle import dinuc_shuffle  # function to do a dinucleotide shuffle

        rescale_conv_revealcancel_fc_many_refs_func = get_shuffle_seq_ref_function(
            # score_computation_function is the original function to compute scores
            score_computation_function=method_to_scoring_func['rescale_conv_revealcancel_fc'],
            # shuffle_func is the function that shuffles the sequence
            # technically, given the background of this simulation, randomly_shuffle_seq
            # makes more sense. However, on real data, a dinuc shuffle is advisable due to
            # the strong bias against CG dinucleotides
            shuffle_func=dinuc_shuffle,
            one_hot_func=None)

        num_refs_per_seq = 10  # number of references to generate per sequence
        method_to_task_to_scores['rescale_conv_revealcancel_fc_multiref_'+str(num_refs_per_seq)] = OrderedDict()

        # The sum over the ACGT axis in the code below is important! Recall that DeepLIFT
        # assigns contributions based on difference-from-reference; if
        # a position is [1,0,0,0] (i.e. 'A') in the actual sequence and [0, 1, 0, 0]
        # in the reference, importance will be assigned to the difference (1-0)
        # in the 'A' channel, and (0-1) in the 'C' channel. You want to take the importance
        # on all channels and sum them up, so that at visualization-time you can project the
        # total importance over all four channels onto the base that is actually present (i.e. the 'A'). If you
        # don't do this, your visualization will look very confusing as multiple bases will be highlighted at
        # every position and you won't know which base is the one that is actually present in the sequence!
        method_to_task_to_scores['rescale_conv_revealcancel_fc_multiref_'+str(num_refs_per_seq)] =\
            np.sum(rescale_conv_revealcancel_fc_many_refs_func(
                task_idx=task_idx,
                input_data_sequences=onehot_data["inputs"],
                num_refs_per_seq=num_refs_per_seq,
                batch_size=200,
                progress_update=1000,
            ), axis=2)
        import csv

        logging.info("Writing scores for task for task %d" % task_idx)

        for method_name, file_path in output_file:
            scores = method_to_task_to_scores[method_name]
            with gzip.open(file_path, 'wt') as score_file:
                for idx in range(np.shape(scores)[0]):
                    scores_for_idx = scores[idx]
                    original_onehot = onehot_data["inputs"][idx]
                    scores_for_idx = original_onehot*scores_for_idx[:, None]

                    score_writer = csv.writer(score_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    score_writer.writerow(scores_for_idx.flatten().tolist())

if __name__ == "__main__":
    cli()
