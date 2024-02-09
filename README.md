# MPRAnn - A Snakemake workflow

[![Snakemake](https://img.shields.io/badge/snakemake-≥7.15.2-brightgreen.svg)](https://snakemake.bitbucket.io)
![Tests](https://github.com/visze/sequence_cnn_models/workflows/Tests/badge.svg)
[![Build Status](https://travis-ci.org/snakemake-workflows/sequence_cnn_models.svg?branch=master)](https://travis-ci.org/snakemake-workflows/sequence_cnn_models)

This is a workflow to generate DNA sequence-based CNN models from genomic DNA sequences predicting MPRA activity for e.g.,log(RNA/DNA) values.

The workflow includes respective folders as follows:
```
    sequence_cnn_models/
        ├── config (contains workflow config files and sample files)
        ├── docs (for documentation)
        ├── resources (demo data, reference files, etc.)
        └── workflow (snakemake workflow and scripts)
```
Codes are in respective folders, i.e. `scripts`, `rules`, and `envs` (in `workflow\`). The workflow is in the `workflow\Snakefile` and the main configuration is in the `config\config.yml` file. Please review this file and adjust parameters accordingly.

## Authors

* Max Schubach (@visze), Berlin Institute of Health (BIH), [Computational Genome Biology](https://kircherlab.bihealth.org)
* Pyaree Mohan Dash (@vpyareedash), Berlin Institute of Health (BIH), [Computational Genome Biology](https://kircherlab.bihealth.org)

## Usage

If you use this workflow in a paper, don't forget to give credits to the authors by citing the URL of this (original) repository and, if available, its DOI (see above).

## Dependencies

Snakemake manages dependencies automatically via conda, please update `workflow\envs\` files accordingly.

Please download a copy of [Snakemake Wrappers](https://github.com/snakemake/snakemake-wrappers) in the `resources\` directory or update the `config.yml` file accordingly.

The workflow also requires a reference genome in 1. .genome format, and 2. .fasta format. 
Please download the reference genome in the `resources\` directory or update the `config.yml` file accordingly.

### Step 1: Obtain a copy of this workflow

1. Create a new github repository using this workflow [as a template](https://help.github.com/en/articles/creating-a-repository-from-a-template).
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the newly created repository to your local system, into the place where you want to perform the data analysis.

### Step 2: Configure workflow

Configure the workflow according to your needs via editing the files in the `config/` folder. Adjust `config.yml` to configure the workflow execution, and `samples.tsv` to specify your sample setup.

### Step 3: Install Snakemake

Install Snakemake using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):

    conda create -c bioconda -c conda-forge -n snakemake snakemake

For installation details, see the [instructions in the Snakemake documentation](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).
A typical installation takes ~ 5 minutes in a normal desktop computer.

### Step 4: Execute workflow

Activate the conda environment:

    conda activate snakemake

Test your configuration by performing a dry-run via

    snakemake --use-conda -n

Execute the workflow locally via

    snakemake --use-conda --cores $N

using `$N` cores or run it in a cluster environment via

    snakemake --use-conda --cluster qsub --jobs 100

or

    snakemake --use-conda --drmaa --jobs 100

If you not only want to fix the software stack but also the underlying OS, use

    snakemake --use-conda --use-singularity

in combination with any of the modes above.
See the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/executable.html) for further details.

### Step 5: Investigate results

After successful execution, you can create a self-contained interactive HTML report with all results via:

    snakemake --report report.html

This report can, e.g., be forwarded to your collaborators.
An example (using some trivial test data) can be seen [here](https://cdn.rawgit.com/snakemake-workflows/rna-seq-kallisto-sleuth/master/.test/report.html).

### Step 6: Commit changes

Whenever you change something, don't forget to commit the changes back to your github copy of the repository:

    git commit -a
    git push

### Step 7: Obtain updates from upstream

Whenever you want to synchronize your workflow copy with new developments from upstream, do the following.

1. Once, register the upstream repository in your local copy: `git remote add -f upstream git@github.com:snakemake-workflows/sequence_cnn_models.git` or `git remote add -f upstream https://github.com/snakemake-workflows/sequence_cnn_models.git` if you do not have setup ssh keys.
2. Update the upstream version: `git fetch upstream`.
3. Create a diff with the current version: `git diff HEAD upstream/master workflow > upstream-changes.diff`.
4. Investigate the changes: `vim upstream-changes.diff`.
5. Apply the modified diff via: `git apply upstream-changes.diff`.
6. Carefully check whether you need to update the config files: `git diff HEAD upstream/master config`. If so, do it manually, and only where necessary, since you would otherwise likely overwrite your settings and samples.


### Step 8: Contribute back

In case you have also changed or added steps, please consider contributing them back to the original repository:

1. [Fork](https://help.github.com/en/articles/fork-a-repo) the original repo to a personal or lab account.
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the fork to your local system, to a different place than where you ran your analysis.
3. Copy the modified files from your analysis to the clone of your fork, e.g., `cp -r workflow path/to/fork`. Make sure to **not** accidentally copy config file contents or sample sheets. Instead, manually update the example config files if necessary.
4. Commit and push your changes to your fork.
5. Create a [pull request](https://help.github.com/en/articles/creating-a-pull-request) against the original repository.

## Testing

Test cases are in the subfolder `.test`. They are automatically executed via continuous integration with [Github Actions](https://github.com/features/actions).

## Demo

Let's try to run the workflow with the demo data provided in the `resources\` directory. (i.e., `resources\demo\`)

### Input files

The input files are in the `resources\demo\` directory. The input files are as follows:
    
    1. `resources\demo\example_sequences.fa` - contains the DNA sequences in fasta format
    
    2. `resources\demo\example_labels.tsv` - contains 3 columns i.e., 'BIN', 'ID', and, 'MEAN' (value1*) in tsv format. 

The 'BIN' column contains the bin number (1-10) and the 'ID' column contains the sequence ID. The 'ID' column should match the sequence ID in the fasta file. The 'MEAN' column contains the mean value of the log(RNA/DNA). Adding more columns is possible, as additional values which starts a multi-task learning.

### Configure config.yml

1. Add the path of snakemake wrapper directory as follows:
```
...
wrapper_directory: resources/snakemake_wrappers
...
```

2. Please add the path of the reference genome files as follows:
```
...
reference:
  genome: resources/example.fa.genome # genome file .genome
  fasta: resources/example.fa # genome file .fa
...  
```

3. Add the path of the input files as follows:
```
...
input:
  fasta: resources/demo/example_sequences.fa
  labels: resources/demo/example_labels.tsv
...
```  

### Run the workflow 

Run as follows (if the device has GPU, snakemake will automatically detect it and run the workflow on GPU):
```bash
snakemake --snakefile workflow/Snakefile --configfile config/config.yml -c 1 --use-conda -p
```
Typical runtime on a non-GPU device is ~ 1 hour and 30 minutes. (1 hour 15 minutes in a GPU enabled device)

A successful run ends up with the following output:
```html
...
Finished job 0.
n of n steps (100%) done
Complete log: .snakemake/log/20XX-XX-27T155007.853000.snakemake.log
```

### Expected Output

The output files are now in the `results\` directory. 
```html
    sequence_cnn_models/results/
        ├── correlation
            ├── regression.MEAN.tsv.gz (correlation between predicted and observed values)
        ├── predictions
            ├── finalConcat.labels.cleaned.tsv.gz
            ...
        ├── regression_input (Train, test and validation input files used for training)
        └── training (Performance of fitted models, log.tsv files, *model.json*, *model.h5*, etc.) 
```

The file `results/predictions/finalConcat.labels.cleaned.tsv.gz` contains the predicted values for the input sequences. 

The file `results/correlation/regression.MEAN.tsv.gz` contains the correlation between predicted and observed values.

All models are saved in the `results/training/` directory as `.json` and `.h5` files.

### Run the workflow with MPRAnn best model

To run the workflow with the best model of MPRAnn (introduced in [Agarwal et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36945371/)), please download the model files from [Zenodo](https://zenodo.org/records/8219231) and update the `config.yml` file accordingly or use `config/config_mprann.yml` file.

Note: Please modify cell type names depending on the model or model files used.



