import numpy as np
import h5py
import modisco
import h5py
import click


@click.command()
@click.option('--input',
              'input_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='h5 score file of ism')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option("--output-figures-dir",
              "output_figures_dir",
              required=True,
              type=str,
              help="Length of the sequence"
              )
def cli(input_file, output_file, output_figures_dir):

    scores = h5py.File(input_file, 'r')

    # Predictions

    predictions = scores["ism"][:]
    onehot_data = scores["seqs"][:]

    # Normalize pedictions for hypothetical score
    predictions_normed = np.nan_to_num(np.transpose(np.transpose(predictions) - np.transpose(predictions.mean(axis=2))))

    # enerat ethe input. Needing ordererd dicts
    from collections import OrderedDict
    task_to_scores = OrderedDict()
    print(np.shape(predictions[:, :, :, 0]))
    task_to_scores["task0"] = predictions[:, :, :, 0]
    task_to_hyp_scores = OrderedDict()
    task_to_hyp_scores["task0"] = predictions_normed[:, :, :, 0]
    print(np.shape(predictions_normed[:, :, :, 0]))
    exit()

    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow()(
        task_names=["task0"],
        contrib_scores=task_to_scores,
        hypothetical_contribs=task_to_hyp_scores,
        one_hot=onehot_data,
        just_return_seqlets=False,
        plot_save_dir=output_figures_dir,
        null_per_pos_scores=null_per_pos_scores)

    grp = h5py.File(output_file, 'w')
    tfmodisco_results.save_hdf5(grp)
    grp.close()


if __name__ == '__main__':
    cli()
