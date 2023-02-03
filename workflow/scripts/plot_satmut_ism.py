import h5py
import click
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from basenji import plots
from basenji import dna_io
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
@click.option('--p-value',
              'pvalue_threshold',
              required=False,
              default=1.00,
              type=float,
              help='P-value threshold')
@click.option('--num-bcs',
              'num_bcs',
              required=False,
              default=0,
              type=int,
              help='Number of barcodes required to be significant')
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
@click.option('--output-heatmap',
              'output_heatmap',
              required=True,
              type=click.Path(writable=True),
              help='Heatmap output file')
@click.option('--output-scatter',
              'output_scatter',
              required=True,
              type=click.Path(writable=True),
              help='Scatterplot output file')
def cli(score_file, satmut_file, output_heatmap, pvalue_threshold, num_bcs, output_scatter, target_idx, start_pos):
    # take the scores
    click.echo('Loading data...')

    

    h5 = h5py.File(satmut_file, 'r')
    seqs_satmut = h5['seqs'][:][0]
    seqlen = len(seqs_satmut[:])
    scores_satmut = h5['satmut'][:, :, :, 0][0]
    p_value = h5['satmut'][:, :, :, 1][0] <= pvalue_threshold
    bcs = h5['satmut'][:, :, :, 2][0] >= num_bcs
    scores_satmut[~p_value] = np.nan
    scores_satmut[~bcs] = np.nan
    h5.close()

    h5 = h5py.File(score_file, 'r')
    seqs_score = h5['seqs'][:seqlen]
    scores_ism = h5['ism'][:seqlen, :, target_idx]
    h5.close()

    if not np.any(seqs_score==seqs_satmut):
        raise ValueError('Sequences do not match')

    
    plot_start = 0
    plot_end = seqlen

    seqs_tmp = seqs_satmut[plot_start:plot_end]
    satmut_scores_tmp = scores_satmut[plot_start:plot_end]
    ism_scores_tmp = scores_ism[plot_start:plot_end]

    p_value_tmp = p_value[plot_start:plot_end]
    bcs_tmp = bcs[plot_start:plot_end]

    ism_ref_scores = ism_scores_tmp[seqs_tmp]
    ism_ref_scores = np.repeat(ism_ref_scores[:, np.newaxis], 4, axis=1)

    # difference from reference
    ism_delta_ti = ism_scores_tmp - ism_ref_scores
    ism_delta_ti[~p_value_tmp] = np.nan
    ism_delta_ti[~bcs_tmp] = np.nan

    # compute loss and gain
    satmut_delta_mean = np.nan_to_num(satmut_scores_tmp).mean(axis=1)
    satmut_loss = np.nan_to_num(satmut_scores_tmp).min(axis=1)
    satmut_gain = np.nan_to_num(satmut_scores_tmp).max(axis=1)

    ism_delta_mean = np.nan_to_num(ism_delta_ti).mean(axis=1)
    ism_loss = np.nan_to_num(ism_delta_ti).min(axis=1)
    ism_gain = np.nan_to_num(ism_delta_ti).max(axis=1)

    print(dna_io.hot1_dna(seqs_tmp))


    f, axs = plt.subplots(figsize=(40, 10), nrows=10, ncols=1)
    plot_seqlogo(axs[0], seqs_tmp, satmut_delta_mean, pseudo_pct=0)  # plot seqlogo, mean
    plot_seqlogo(axs[1], seqs_tmp, ism_delta_mean, pseudo_pct=0)  # plot seqlogo, mean
    plot_seqlogo(axs[2], seqs_tmp, -satmut_loss, pseudo_pct=0)  # plot seqlogo, loss
    plot_seqlogo(axs[3], seqs_tmp, -ism_loss, pseudo_pct=0)  # plot seqlogo, loss
    plot_seqlogo(axs[4], seqs_tmp, satmut_gain, pseudo_pct=0)  # plot seqlogo, gain
    plot_seqlogo(axs[5], seqs_tmp, ism_gain, pseudo_pct=0)  # plot seqlogo, gain
    plot_sad(axs[6], satmut_loss, satmut_gain)  # plot loss and gain
    plot_sad(axs[7], ism_loss, ism_gain)  # plot loss and gain
    plot_heat(axs[8], satmut_scores_tmp.T, 0.01)  # , cbar=False plot heatmap
    plot_heat(axs[9], ism_delta_ti.T, 0.01)  # , cbar=False plot heatmap
    
    plt.tight_layout()

    
    f.savefig(output_heatmap, format="pdf")
    plt.close()

    df = pd.DataFrame({'SatMut': satmut_scores_tmp.flatten(), 'ISM': ism_delta_ti.flatten()})
    p_corr = df.corr()
    f = sns.scatterplot(x="SatMut", y="ISM", data=df, size=1,linewidth=0, legend=False)
    plt.text(-0.2, 0.6, "Pearson corr. %.2f" % p_corr.iloc[0,1], 
        horizontalalignment='center', verticalalignment="top", fontsize=12, 
        color='black', fontweight='bold')
    f.figure.savefig(output_scatter, format="pdf")
    plt.close()


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
    # ax.yaxis.set_ticklabels('ACGT', rotation='vertical')
    ax.set_yticks(np.arange(4) + 0.5, 'ACGT', rotation='horizontal')


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
