
import modiscolite

import numpy as np
import h5py
import click

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

    click.echo('Loading data...')
    for f in score_files:
        h5 = h5py.File(f, 'r')
        seqs.append(h5['seqs'][:])
        try:
            scores.append(h5['scores'][:, :, :, target_id])
        except:
            scores.append(h5['ism'][:, :, :, target_id])
        h5.close()

    seqs = np.concatenate(seqs)
    scores = np.concatenate(scores).astype(np.float32)

    print(scores.shape)
    print(seqs.shape)

    click.echo('Running tfmodisco-lite')
    print(scores.dtype)
    print(seqs.dtype)
    ##########################
    # run tfmodisco-lite workflow #
    ##########################

    if config == "saluki":
        pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        sliding_window_size=8,
        flank_size=8,
        min_metacluster_size=20,
        target_seqlet_fdr=0.1,
        hypothetical_contribs=scores,
        one_hot=seqs,
        max_seqlets_per_metacluster=mspmc,
        trim_to_window_size=30,
        n_leiden_runs=2,
        initial_flank_to_add=10,
        final_min_cluster_size=30,
        verbose=True)
    else:
        pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=scores,
        one_hot=seqs,
        max_seqlets_per_metacluster=mspmc,
        verbose=True)

    

    modiscolite.io.save_hdf5(output_file, pos_patterns, neg_patterns)


if __name__ == '__main__':
    cli()
