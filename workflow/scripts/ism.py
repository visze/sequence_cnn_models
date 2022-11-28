#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import click
import numpy as np
import os
import h5py


"""
saluki_ism_tfr.py

Compute in silico mutagenesis of sequences in tfrecords.
"""

################################################################################
# cli
################################################################################


@click.command()
@click.option('--sequence',
              'sequence_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Test sequences')
@click.option('--sequence-length',
              'sequence_length',
              required=True,
              type=int,
              help='Length of the original sequence in FASTA file')
@click.option('--mutation-length',
              'mutation_length',
              required=True,
              type=int,
              help='Length of the mutated sequence')
@click.option('--mutation-start',
              'mutation_start',
              required=True,
              type=int,
              help='Start of the mutated sequence')
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
@click.option('--scores-output',
              'scores_h5_file',
              required=True,
              type=click.Path(writable=True),
              help='Scores h5 file')
def cli(sequence_file, sequence_length, mutation_length, mutation_start, model_file, weights_file, scores_h5_file):

    print("importing tensorflow")
    from sequence import SeqFastaLoader1D
    import tensorflow as tf

    #######################################################
    # model
    print("load model")
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights_file)

    #######################################################
    # ISM sequences
    print("load sequences")
    # construct dataset
    dl_eval = SeqFastaLoader1D(sequence_file, length=sequence_length)

    eval_data = dl_eval.load_all()

    num_seqs = len(eval_data["inputs"])

    print("ISM sequences")
    # make sequence generator
    seqs_gen = satmut_gen(eval_data, mutation_start, mutation_length)

    seqs = np.array([i for i in seqs_gen], 'float16')

    #################################################################
    # predict scores, write output
    print("predict scores")
    strategy = tf.distribute.MirroredStrategy(devices=None)

    with strategy.scope():

        preds = model.predict(seqs)

    num_targets = preds.shape[1]

    #################################################################
    # setup output

    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, 'w')
    scores_h5.create_dataset('seqs', dtype='bool',
                             shape=(num_seqs, mutation_length, 4))
    scores_h5.create_dataset('ref', dtype='float16',
                             shape=(num_seqs, num_targets))
    scores_h5.create_dataset('ism', dtype='float16',
                             shape=(num_seqs, mutation_length, 4, num_targets))

    # sequence index
    si = 0

    # predictions index
    pi = 0

    print("combine scores")
    for seq_1hotc in eval_data["inputs"]:

        # write reference sequence
        seq_mut_len = mutation_length
        seq_mut_start = mutation_start-1
        seq_1hot_mut = seq_1hotc[seq_mut_start:seq_mut_start+seq_mut_len]
        scores_h5['seqs'][si, -seq_mut_len:, :] = seq_1hot_mut.astype('bool')

        # initialize scores
        seq_scores = np.zeros((seq_mut_len, 4, num_targets), dtype='float32')

        # collect reference prediction
        preds_mut0 = preds[pi]
        pi += 1

        # for each mutated position
        for mi in range(seq_mut_len):
            # if position as nucleotide
            if seq_1hot_mut[mi].max() < 1:
                # reference score
                seq_scores[mi, :, :] = preds_mut0
            else:
                # for each nucleotide
                for ni in range(4):
                    if seq_1hot_mut[mi, ni]:
                        # reference score
                        seq_scores[mi, ni, :] = preds_mut0
                    else:
                        # collect and set mutation score
                        seq_scores[mi, ni, :] = preds[pi]
                        pi += 1

        # normalize
        seq_scores -= seq_scores.mean(axis=1, keepdims=True)

        # write to HDF5
        scores_h5['ref'][si] = preds_mut0.astype('float16')
        scores_h5['ism'][si, -seq_mut_len:, :, :] = seq_scores.astype('float16')

        # increment sequence
        si += 1
    # close output HDF5
    scores_h5.close()


def satmut_gen(eval_data, mut_start, mut_len):
    """Construct generator for 1 hot encoded saturation
       mutagenesis DNA sequences."""

     # set mutation boundaries
    mut_start = mut_start - 1
    mut_end = mut_start + mut_len

    for seq_1hotc in eval_data["inputs"]:
        yield seq_1hotc

        # for mutation positions
        for mi in range(mut_start, mut_end):
            # if position as nucleotide
            if seq_1hotc[mi].max() == 1:
                # for each nucleotide
                for ni in range(4):
                    # if non-reference
                    if seq_1hotc[mi, ni] == 0:
                        # copy and modify
                        seq_mut_1hotc = np.copy(seq_1hotc)
                        seq_mut_1hotc[mi, :4] = 0
                        seq_mut_1hotc[mi, ni] = 1

                        yield seq_mut_1hotc


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    cli()
