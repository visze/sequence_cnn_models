import click
import h5py
import numpy as np
import os

##########
# inputs #
##########


@click.command()
@click.option('--satmut',
              'satmut_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='satmut file in TSV format')
@click.option('--start-position',
              'start_pos',
              required=True,
              type=int,
              help='Start position in the human genome.')
@click.option('--sequence',
              'sequence_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Fasta sequence of reference file')
@click.option('--sequence-length',
              'sequence_length',
              required=True,
              type=int,
              help='Length of the original sequence in FASTA file')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Output file')
def cli(satmut_file, sequence_file, start_pos, sequence_length, output_file):

    from sequence import SeqFastaLoader1D, StringFastaLoader1D
    import pandas as pd

    #######################################################
    # ISM sequences
    print("load sequences")
    # construct dataset
    dl_eval = SeqFastaLoader1D(sequence_file, length=sequence_length)

    eval_data = dl_eval.load_all()

    dl_eval_str = StringFastaLoader1D(sequence_file, length=sequence_length)

    seqs_str = dl_eval_str.load_all()

    num_seqs = len(eval_data["inputs"])

    print("satMut sequences")
    # make sequence generator
    seqs_gen = satmut_gen(eval_data, 1, sequence_length)


    
    #########################################
    # Satmut scores
    satmut_df = pd.read_csv(satmut_file, header='infer', sep='\t')
    
    preds = []
    for position in (range(0, sequence_length)):
        genome_position = start_pos + position
        for alt in "ACGT":
            if alt == seqs_str["inputs"][0][position]:
                continue
            search = satmut_df[["Value","P-Value","Tags"]].where(satmut_df["Position"] == genome_position).where(satmut_df["Alt"] == alt).dropna()
            #
            if search.empty:
                preds.append([None,None,None])
            else:
                preds.append([search.iloc[0]["Value"], search.iloc[0]["P-Value"], search.iloc[0]["Tags"]])


    num_targets = 3
    if os.path.isfile(output_file):
        os.remove(output_file)
    scores_h5 = h5py.File(output_file, 'w')
    scores_h5.create_dataset('seqs', dtype='bool',
                             shape=(num_seqs, sequence_length, 4))
    scores_h5.create_dataset('satmut', dtype='float16',
                             shape=(num_seqs, sequence_length, 4, num_targets))

    # sequence index
    si = 0

    # predictions index
    pi = 0

    print("combine scores")
    for seq_1hotc in eval_data["inputs"]:

        # write reference sequence
        seq_mut_len = sequence_length
        seq_mut_start = 0
        seq_1hot_mut = seq_1hotc[seq_mut_start:seq_mut_start+seq_mut_len]
        scores_h5['seqs'][si, -seq_mut_len:, :] = seq_1hot_mut.astype('bool')

        # initialize scores
        seq_scores = np.zeros((seq_mut_len, 4, num_targets), dtype='float32')


        # for each mutated position
        for mi in range(seq_mut_len):
            # if position as nucleotide
            if seq_1hot_mut[mi].max() < 1:
                # reference score
                seq_scores[mi, :, :] = np.nan
            else:
                # for each nucleotide
                for ni in range(4):
                    if seq_1hot_mut[mi, ni]:
                        # reference score
                        seq_scores[mi, ni, :] = np.nan
                    else:
                        # collect and set mutation score
                        seq_scores[mi, ni, :] = preds[pi]
                        pi += 1

        # normalize
        # seq_scores -= seq_scores.mean(axis=1, keepdims=True)

        # write to HDF5
        scores_h5['satmut'][si, -seq_mut_len:, :, :] = seq_scores.astype('float16')

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

if __name__ == '__main__':
    cli()
