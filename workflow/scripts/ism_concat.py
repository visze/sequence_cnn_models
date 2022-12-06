import click
import h5py
import numpy as np
import os

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
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Output file')
def cli(score_files, output_file):

    seqs = []
    ref = []
    scores = []
    for f in score_files:
        h5 = h5py.File(f,'r')
        if len(seqs) == 0:
            seqs.append(h5['seqs'][:])
            ref.append(h5['ref'][:])
        scores.append([h5['ism'][:]])
        h5.close()

    seqs = np.concatenate(seqs)
    ref = np.concatenate(ref)
    scores = np.mean(scores,axis=0)[0,:,:,:]

    if os.path.isfile(output_file):
        os.remove(output_file)
    scores_h5 = h5py.File(output_file, 'w')
    scores_h5['ref'] = ref
    scores_h5['seqs'] = seqs
    scores_h5['ism'] = scores
    scores_h5.close()

if __name__ == '__main__':
    cli()