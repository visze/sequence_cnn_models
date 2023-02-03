import click
import h5py
import numpy as np
import os

##########
# inputs #
##########


"""
This script combines the ISM scores from multiple splits into one file. It averages the obverlap of scors between splits.
"""


@click.command()
@click.option('--input',
              'input_file',
              required=True,
              multiple=False,
              type=click.Path(exists=True, readable=True),
              help='h5 ISM score file')
@click.option('--overlap',
              'overlap',
              required=True,
              type=int,
              help='Overlap between splits')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Output file')
def cli(input_file, overlap, output_file):

    seqs = []
    ref = []
    scores = []
    h5 = h5py.File(input_file, 'r')
    ref = h5['ref'][:]
    for i in range(h5['seqs'].shape[0]):
        if i == 0:
            seqs.append(h5['seqs'][i, :, :])
            scores.append(h5['ism'][i, :-overlap, :, :])
        else:
            seqs.append(h5['seqs'][i, overlap:, :])
            scores.append(np.mean([h5['ism'][i-1, -overlap:, :, :],
                          h5['ism'][i, :overlap, :, :]], axis=0))
            if (i == h5['seqs'].shape[0] - 1):
                # Add the last missing overlap part for scores
                scores.append(h5['ism'][i, overlap:, :, :])
            else: 
                # Add center for middle intervals
                scores.append(h5['ism'][i, overlap:-overlap, :, :])

    h5.close()

    seqs = np.concatenate(seqs)
    scores = np.concatenate(scores)

    if os.path.isfile(output_file):
        os.remove(output_file)
    scores_h5 = h5py.File(output_file, 'w')
    scores_h5['ref'] = ref
    scores_h5['seqs'] = seqs
    scores_h5['ism'] = scores
    scores_h5.close()


if __name__ == '__main__':
    cli()
