# this run in tf114-gpu envir

from modisco.visualization import viz_sequence
import modisco
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
from collections import OrderedDict
import click
import matplotlib
matplotlib.use('pdf')

######################################################################
# Script modified from one written by Han Yuan, Calico Labs
######################################################################

##########
# inputs #
##########


@click.command()
@click.option('--scores',
              'score_files',
              required=True,
              multiple=True,
              type=click.Path(exists=True, readable=True),
              help='h5 ISM score file')
@click.option('--target-id',
              'target_id',
              required=True,
              type=int,
              default=0,
              help='target task id')
@click.option('--mspmc',
              'mspmc',
              required=True,
              type=int,
              default=20000,
              help='maximum seqlets per metacluster')
@click.option('--config',
              'config',
              required=True,
              type=click.Choice(['default', 'saluki'], case_sensitive=False),
              default="saluki",
              help='configuration of modisco')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Output file')
def cli(score_files, target_id, mspmc, config, output_file):

    seqs = []
    scores = []

    for f in score_files:
        h5 = h5py.File(f, 'r')
        seqs.append(h5['seqs'][:])
        try:
            scores.append(h5['scores'][:, :, :, target_id])
        except:
            scores.append(h5['ism'][:, :, :, target_id])
        h5.close()

    seqs = np.concatenate(seqs)
    scores = np.concatenate(scores)

    # run tfmodisco
    target_importance = OrderedDict()
    target_importance["task"] = seqs * scores
    target_hypothetical = OrderedDict()
    target_hypothetical["task"] = scores

    ##########################
    # run tfmodisco workflow #
    ##########################
    if config == "saluki":
        tfm_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
            sliding_window_size=8,
            flank_size=8,
            min_metacluster_size=20,
            target_seqlet_fdr=0.1,
            max_seqlets_per_metacluster=mspmc,  # don't put constrain on this
            seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                n_cores=16,  # use 16 cores
                trim_to_window_size=30,
                # min_num_to_trim_to=15, #added later
                initial_flank_to_add=10,
                kmer_len=8, num_gaps=3,
                num_mismatches=2,
                final_min_cluster_size=30)
        )(
            task_names=["task"],
            contrib_scores=target_importance,
            hypothetical_contribs=target_hypothetical,
            revcomp=False,
            one_hot=seqs)  # seqs
    else:
        tfm_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
            max_seqlets_per_metacluster=mspmc,  # don't put constrain on this
            seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                n_cores=16)
        )(
            task_names=["task"],
            contrib_scores=target_importance,
            hypothetical_contribs=target_hypothetical,
            one_hot=seqs)  # seqs

    h5_out = h5py.File(output_file, 'w')
    tfm_results.save_hdf5(h5_out)
    h5_out.close()


if __name__ == '__main__':
    cli()
