import h5py
import click
import numpy as np
import pandas as pd


@click.command()
@click.option('--score',
              'score_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Score h5 file sequences')
@click.option('--satmut',
              'satmut_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='SatMut score h5 file sequences')
@click.option('--target',
              'target_idx',
              required=False,
              default=0,
              type=int,
              help='Prediction index of scores to plot')
@click.option('--start',
              'start_pos',
              required=False,
              default=0,
              type=int,
              help='X axis start position')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='TSV file of variants with ISM and satmut scores')
def cli(score_file, satmut_file, output_file, target_idx, start_pos):
    # take the scores
    click.echo('Loading data...')

    

    h5 = h5py.File(satmut_file, 'r')
    seqs_satmut = h5['seqs'][:][0]
    seqlen = len(seqs_satmut[:])
    scores_satmut = h5['satmut'][:, :, :, 0][0]
    p_values =  h5['satmut'][:, :, :, 1][0]
    bcs = h5['satmut'][:, :, :, 2][0]
    h5.close()

    h5 = h5py.File(score_file, 'r')
    seqs_score = h5['seqs'][:seqlen]
    scores_ism = h5['ism'][:seqlen, :, target_idx]
    h5.close()

    if not np.any(seqs_score==seqs_satmut):
        raise ValueError('Sequences do not match')


    ism_ref_scores = scores_ism[seqs_satmut]
    ism_ref_scores = np.repeat(ism_ref_scores[:, np.newaxis], 4, axis=1)

    # difference from reference
    ism_delta_ti = scores_ism - ism_ref_scores

    # get satMut sequence
    ref_sequence = []
    alt_sequence = []
    positions = []
    # for mutation positions
    for mi in range(seqlen):
        # if position as nucleotide
        if seqs_satmut[mi].max() == 1:
            if seqs_satmut[mi,0]:
                ref = "A"
            elif seqs_satmut[mi,1]:
                ref = "C"
            elif seqs_satmut[mi,2]:
                ref = "G"
            elif seqs_satmut[mi,3]:
                ref = "T"

            # for each nucleotide
            for ni in range(4):
                # if non-reference
                positions.append(start_pos+mi)
                if ni == 0:
                    alt = "A"
                elif ni == 1:
                    alt = "C"
                elif ni == 2:
                    alt = "G"
                elif ni == 3:
                    alt = "T"
                ref_sequence.append(ref)
                alt_sequence.append(alt)


    df = pd.DataFrame({
        "Position": positions,
        "Ref": ref_sequence,
        "Alt": alt_sequence,
        'ISM_delta': ism_delta_ti.flatten(), 
        'satmut_coefficient': scores_satmut.flatten(), 
        'pvalue': p_values.flatten(),
        "BCs" : bcs.flatten()})

    df = df[df["Ref"]!= df["Alt"]]
    
    # saving as tsv file
    df.to_csv(output_file, sep="\t", index=False)



if __name__ == '__main__':
    cli()
