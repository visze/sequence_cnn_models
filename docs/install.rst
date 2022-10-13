.. _Installation:

=====================
Installation
=====================

Installation should take less than 5 minutes

System Requirements
===================

CentOS Linux 7 or above

Required packages
==================

.. code-block:: bash

  	conda (mamba) 4.6 or above

Download here: https://docs.conda.io/en/latest/miniconda.html

.. code-block:: bash

    snakemake 7.4,1 or above

Download here: https://snakemake.readthedocs.io/


Clone repository
=================

Download here: https://github.com/kircherlab/sequence_cnn_models.git


Set up conda with snakemake environment
==========================================

The whole pipeline is set up to run on a Linux system.

Install the the conda environment. The general conda environment is called ``snakemake``.

.. code-block:: bash

    cd sequence_cnn_models
    conda create -c bioconda -c conda-forge -n snakemake snakemake
    
    # activate snakemake
    conda activate snakemake

To deactivate the environment, use:

.. code-block:: bash

    conda deactivate



Quick test
============

.. code-block:: bash

    conda activate snakemake
    snakemake --help
    
