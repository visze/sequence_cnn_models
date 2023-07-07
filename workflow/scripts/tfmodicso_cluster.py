import h5py
import numpy as np
import click
import logging
from lib.sequence import SeqClassificationDataLoader1D


@click.command()
@click.option('--ism',
              'ism_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Test sequences')
@click.option('--input',
              'input_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='h5 score file of motdisco')
@click.option('--background',
              'background_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='background sequences')
@click.option('--reference',
              'reference_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Reference sequences')
@click.option('--output-heatmap',
              'output_heatmap',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option('--output-log',
              'output_log_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option('--output-motifs',
              'output_motifs_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option("--output-figures-dir",
              "output_figures_dir",
              required=True,
              type=str,
              help="Length of the sequence"
              )
def cli(input_file, ism_file, background_file, reference_file, output_heatmap, output_figures_dir, output_log_file, output_motifs_file):

    from modisco.visualization import viz_sequence
    from matplotlib import pyplot as plt

    import modisco.affinitymat.core
    import modisco.cluster.phenograph.core
    import modisco.cluster.phenograph.cluster
    import modisco.cluster.core
    import modisco.aggregator

    logging.basicConfig(filename=output_log_file, level=logging.INFO, format="%(message)s")

    hdf5_results = h5py.File(input_file, "r")

    # Metaclusters heatmap
    import seaborn as sns
    activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
        np.array(
            [x[0] for x in sorted(
                enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
                key=lambda x: x[1])])]
    sns.heatmap(activity_patterns, center=0)
    plt.savefig(output_heatmap)

    metacluster_names = [
        x.decode("utf-8") for x in
        list(hdf5_results["metaclustering_results"]
             ["all_metacluster_names"][:])]

    dl_backgorund = SeqClassificationDataLoader1D(intervals_file=background_file,
                                                  fasta_file=reference_file, label_dtype=int, ignore_targets=True)

    background = dl_backgorund.load_all()

    background = np.mean(np.mean(background["inputs"], axis=0), axis=0)

    def writeMotif(file, motif, cluster, name):
        file.write("%s.%s\n" % (cluster, name))
        file.write("A: "+" ".join(list(map(str, motif.tolist()[0]))) + "\n")
        file.write("C: "+" ".join(list(map(str, motif.tolist()[1]))) + "\n")
        file.write("G: "+" ".join(list(map(str, motif.tolist()[2]))) + "\n")
        file.write("T: "+" ".join(list(map(str, motif.tolist()[3]))) + "\n")

    # from modisco.tfmodisco_workflow import workflow

    # Predictions integrad gradients
    scores = h5py.File(ism_file, 'r')

    # Predictions

    predictions = scores["ism"][:, :, :, 0]
    onehot_data = scores["seqs"][:]

    # Normalize pedictions for hypothetical score
    predictions_normed = np.nan_to_num(np.transpose(np.transpose(predictions) - np.transpose(predictions.mean(axis=2))))

    from collections import OrderedDict
    task_to_scores = OrderedDict()
    task_to_scores["task0"] = predictions
    task_to_hyp_scores = OrderedDict()
    task_to_hyp_scores["task0"] = predictions_normed

    track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
        task_names=['task0'],
        contrib_scores=task_to_scores,
        hypothetical_contribs=task_to_hyp_scores,
        one_hot=onehot_data)

    with open(output_motifs_file, 'w') as motifsOut:
        for metacluster_name in metacluster_names:
            logging.info("%s", metacluster_name)
            metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                               [metacluster_name])
            logging.info("activity pattern %s:", metacluster_grp["activity_pattern"][:])

            patterns = modisco.util.load_patterns(metacluster_grp["seqlets_to_patterns_result"]["patterns"], track_set)
            if (len(patterns) == 0):
                logging.info("No motifs found for this activity pattern: %s", metacluster_name)
            i = 0
            for pattern in patterns:
                pattern_name = str(i)

                logging.info("%s pattern %s", metacluster_name, pattern_name)
                logging.info("%s patter %s total seqlets: %d", metacluster_name,
                             pattern_name, len(pattern.seqlets_and_alnmts.unique_seqlets))
                # Task 0 hypothetical scores
                viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"].fwd)
                plt.savefig("%s/%s.%s.hypotheticalContribs.fwd.png" %
                            (output_figures_dir, metacluster_name, pattern_name))
                # Task 0 actual importance scores
                viz_sequence.plot_weights(pattern["task0_contrib_scores"].fwd)
                plt.savefig("%s/%s.%s.importanceScore.fwd.png" % (output_figures_dir, metacluster_name, pattern_name))
                # onehot, fwd and rev
                viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"].fwd),
                                                                background=background))
                plt.savefig("%s/%s.%s.oneHot.fwd.png" % (output_figures_dir, metacluster_name, pattern_name))
                viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"].rev),
                                                                background=background))
                plt.savefig("%s/%s.%s.oneHot.rev.png" % (output_figures_dir, metacluster_name, pattern_name))

                try:
                    trimmed_pattern = pattern.trim_by_ic(ppm_track_name="sequence",
                                                                        background=background,
                                                                        threshold=0.5)

                    viz_sequence.plot_weights(viz_sequence.ic_scale(trimmed_pattern["sequence"].fwd, background=background))
                    plt.savefig("%s/%s.%s.oneHotTrimmed.fwd.png" % (output_figures_dir, metacluster_name, pattern_name))
                    writeMotif(motifsOut, np.transpose(viz_sequence.ic_scale(
                        trimmed_pattern["sequence"].fwd, background=background)), metacluster_name, pattern_name)
                except IndexError as err:
                    logging.info("%s pattern %s: IndexError error: {0}".format(err), metacluster_name, pattern_name)

                i = i+1

    hdf5_results.close()


if __name__ == '__main__':
    cli()
