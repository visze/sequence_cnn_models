import h5py
import click
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from basenji import plots
from basenji import dna_io


@click.command()
@click.option('--score',
              'score_files',
              required=True,
              multiple=True,
              type=click.Path(exists=True, readable=True),
              help='Score h5 file sequences')
@click.option('--target',
              'target_idx',
              required=True,
              type=int,
              help='Prediction index of scores to plot')
@click.option('--n-plots',
              'num_plots',
              required=False,
              default=20,
              type=int,
              help='Number of Plots to generate. 0 means all')
@click.option('--output',
              'output_folder',
              required=True,
              type=click.Path(writable=True),
              help='Output Folder')
def cli(score_files, output_folder, target_idx, num_plots):
    # take the scores
    click.echo('Loading data...')
    seqs = []
    scores = []

    for f in score_files:
        h5 = h5py.File(f, 'r')
        seqs.append(h5['seqs'][:])
        try:
            scores.append(h5['scores'][:, :, :, target_idx])
        except:
            scores.append(h5['ism'][:, :, :, target_idx])
        h5.close()

    seqs = np.concatenate(seqs)
    scores = np.concatenate(scores)

    def plot_seq(si, outfile=None):
        seqlen = 200
        plot_start = 0
        plot_end = seqlen

        seqs_tmp = seqs[si, plot_start:plot_end]
        scores_tmp = scores[si, plot_start:plot_end]
        ref_scores = scores_tmp[seqs_tmp]
        ref_scores = np.repeat(ref_scores[:, np.newaxis], 4, axis=1)

        # difference from reference
        delta_ti = scores_tmp - ref_scores

        # compute loss and gain
        delta_mean = delta_ti.mean(axis=1)
        delta_loss = delta_ti.min(axis=1)
        delta_gain = delta_ti.max(axis=1)

        print(si)
        print(dna_io.hot1_dna(seqs_tmp))

        f, axs = plt.subplots(figsize=(40, 10), nrows=6, ncols=1)
        plot_seqlogo(axs[0], seqs_tmp, delta_mean, pseudo_pct=0)  # plot seqlogo, mean
        plot_seqlogo(axs[1], seqs_tmp, -delta_loss, pseudo_pct=0)  # plot seqlogo, loss
        plot_seqlogo(axs[2], seqs_tmp, delta_gain, pseudo_pct=0)  # plot seqlogo, gain
        plot_sad(axs[3], delta_loss, delta_gain)  # plot loss and gain
        plot_heat(axs[4], delta_ti.T, 0.01)  # , cbar=False plot heatmap
        plot_heat(axs[5], delta_ti.T, 0.01, cbar=True)  # , cbar=False plot heatmap
        plt.tight_layout()

        if outfile is not None:
            f.savefig(outfile, format="pdf")
            plt.close()

    # 10 random examples
    if num_plots > 0:
        np.random.seed(5)
        ids = np.random.choice(scores.shape[0], num_plots, replace=False)
    else:
        ids = range(scores.shape[0])
    # ids = [136]
    print(ids)
    for i in ids:
        plot_seq(i, outfile=output_folder + '/example_%d.pdf' % i)
        # plot_seq(i, 725, outfile='png/example_%d.pdf'%i)


def plot_heat(ax, sat_delta_ti, min_limit, cbar=False):
    vlim = max(min_limit, np.nanmax(np.abs(sat_delta_ti)))
    sns.heatmap(
        sat_delta_ti,
        linewidths=0,
        cmap='RdBu_r',
        vmin=-vlim,
        vmax=vlim,
        xticklabels=False, cbar=cbar,
        ax=ax)
    ax.yaxis.set_ticklabels('ACGT', rotation='horizontal')


def plot_sad(ax, sat_loss_ti, sat_gain_ti):
    """ Plot loss and gain SAD scores.
      Args:
          ax (Axis): matplotlib axis to plot to.
          sat_loss_ti (L_sm array): Minimum mutation delta across satmut length.
          sat_gain_ti (L_sm array): Maximum mutation delta across satmut length.
      """

    rdbu = sns.color_palette('RdBu_r', 10)

    ax.plot(-sat_loss_ti, c=rdbu[0], label='loss', linewidth=1)
    ax.plot(sat_gain_ti, c=rdbu[-1], label='gain', linewidth=1)
    ax.set_xlim(0, len(sat_loss_ti))
    ax.legend()
    # ax_sad.grid(True, linestyle=':')

    ax.xaxis.set_ticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)


def plot_seqlogo(ax, seq_1hot, sat_score_ti, pseudo_pct=0.05):
    """ Plot a sequence logo for the loss/gain scores.
      Args:
          ax (Axis): matplotlib axis to plot to.
          seq_1hot (Lx4 array): One-hot coding of a sequence.
          sat_score_ti (L_sm array): Minimum mutation delta across satmut length.
          pseudo_pct (float): % of the max to add as a pseudocount.
      """
    sat_score_cp = sat_score_ti.copy()
    satmut_len = len(sat_score_ti)

    # add pseudocounts
    sat_score_cp += pseudo_pct * sat_score_cp.max()

    # expand
    sat_score_4l = expand_4l(sat_score_cp, seq_1hot)

    plots.seqlogo(sat_score_4l, ax)


def expand_4l(sat_lg_ti, seq_1hot):
    """ Expand
      In:
          sat_lg_ti (l array): Sat mut loss/gain scores for a single sequence and
          target.
          seq_1hot (Lx4 array): One-hot coding for a single sequence.
      Out:
          sat_loss_4l (lx4 array): Score-hot coding?
      """

    # determine satmut length
    satmut_len = sat_lg_ti.shape[0]

    # jump to satmut region in one hot coded sequence
    ssi = int((seq_1hot.shape[0] - satmut_len) // 2)

    # filter sequence for satmut region
    seq_1hot_sm = seq_1hot[ssi:ssi + satmut_len, :]

    # tile loss scores to align
    sat_lg_tile = np.tile(sat_lg_ti, (4, 1)).T

    # element-wise multiple
    sat_lg_4l = np.multiply(seq_1hot_sm, sat_lg_tile)

    return sat_lg_4l


if __name__ == '__main__':
    cli()
