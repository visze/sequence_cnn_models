import argparse
import json
import h5py
import time
import click
import numpy as np
import tensorflow as tf
import deeplift.dinuc_shuffle as ds
from basenji import dna_io
try:
    import rnann
except:
    from basenji import rnann
# from basenji import trainer

if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()


# python spikeinmotif.py --region 5utr --seqs ISM/f3_c2_scores.h5 train_gru/params.json \
#   train_gru/f8_c4/train/model0_best.h5 >spikeinmotif_predictions_5utr_f8c4.txt

# python spikeinmotif.py --codons --frame 0 --region orf --seqs ISM/f3_c2_scores.h5 train_gru/params.json \
#   train_gru/f8_c4/train/model0_best.h5 >spikeinmotif_predictions_orf_codons_frame0.txt

# for x in {0..9}; do { for y in {0..4}; do { REG="5utr"; echo $x, $y; sbatch --mem 20000 -J S$REG$x$y -e spikeinmotif/splice$REG\_f$x\_c$y.err -o spikeinmotif/splice$REG\_f$x\_c$y.txt --wrap=". /home/drk/anaconda3/etc/profile.d/conda.sh; conda activate tf2.6-rna; python spikeinmotif.py --addsplicesites --region $REG --seqs ISM/f$x\_c$y\_scores.h5 train_gru/params.json train_gru/f$x\_c$y/train/model0_best.h5"; } done } done


##########
# inputs #
##########
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
@click.option('--motif',
              'motifs',
              required=True,
              multi=True,
              type=str,
              help='Motif')
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
def cli(sequence_file, sequence_length, mutation_length, mutation_start, motifs, model_file, weights_file, scores_h5_file):

    frame = args.frame
    codons = args.codons
    stopcodons = args.stopcodons
    shuf = False
    input_seqs = args.seqs
    region = args.region
    splicesites = args.splicesites

   # model
    print("load model")
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights_file)

    #######################################################

    # print(input_seqs)
    h5 = h5py.File(input_seqs)
    seqs = h5['seqs']

    # print('%s\t%s\t%s\t%s\t%s' % ("seq", "shuf_num", "motif", "preds"))
    # print(seqs.shape[0])

    def left_justify(data, txstart, txend):
        tmp = np.copy(data[txstart:])
        data[:, :] = 0
        data[0:(txend-txstart), :] = tmp
        return data

    bins = 50
    numshufs = 1

    count = 0
    for i in range(seqs.shape[0]):  # iterate through all sequences
        if shuf:
            numshufs = 10
            if count == 10:
                break
        txend = mutation_start+mutation_length-1
        txstart=mutation_start-1
        idxs = np.arange(txstart, txend)
        count += 1
        if shuf:
            myshufs = ds.dinuc_shuffle(seqs[i, idxs], numshufs)  # 10 dinuc shuffles
        else:
            myshufs = seqs[i, idxs]
        for j in range(numshufs):
            wt = seqs[i, ]
            if shuf:
                wt[idxs] = myshufs[j]
            for m in motifs:
                batch = np.zeros((bins+1, txend, 6))
                data = wt.astype('float32')
                data = left_justify(data, txstart, txend)
                batch[0] = np.expand_dims(data, 0)
                for p in range(bins):
                    shuffseq = np.copy(wt)
                    regionpos = int(p/bins * (idxs[-1]-idxs[0]))
                    pos = idxs[0] + regionpos
                    
                    shuffseq[pos:(pos+len(m))] = dna_io.dna_1hot(m)  # replace with motif

                    # print(dna_io.hot1_dna(shuffseq[idxs]))
                    data = shuffseq.astype('float32')
                    data = left_justify(data, txstart, txend)  # left justify
                    batch[p+1] = np.expand_dims(data, 0)
                pred = model.predict(batch)
                print('seq%s\t%s\t%s' % (count, j, m), end='')
                for p in pred:
                    print('\t%s' % (p[0]), end='')
                print('')
    h5.close()


if __name__ == '__main__':
    cli()
