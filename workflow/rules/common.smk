################################
#### Global functions       ####
################################
from snakemake.workflow import srcdir

SCRIPTS_DIR = srcdir("../scripts")
RESOURCES_DIR = srcdir("../../resources")


def getScript(name):
    return "%s/%s" % (SCRIPTS_DIR, name)


from snakemake.utils import validate
import pandas as pd


# this container defines the underlying OS for each job when using the workflow
# with --use-conda --use-singularity
container: "docker://continuumio/miniconda3"


##### load config and sample sheets #####

# preferrred to use --configfile instead of hard-coded config file
# configfile: "config/config.yaml"


def isRegression():
    """
    Check if we are running a regression test.
    """
    return config["regression"]


if not isRegression():
    validate(config, schema="../schemas/config.schema.yaml")

    samples = pd.read_csv(config["samples"], sep="\t").set_index("sample", drop=False)
    samples.index.names = ["sample_id"]
    validate(samples, schema="../schemas/samples.schema.yaml")

    tests = pd.read_csv(config["tests"], sep="\t").set_index("test", drop=False)
    tests.index.names = ["test_id"]
    validate(tests, schema="../schemas/tests.schema.yaml")


def getWrapper(wrapper):
    """
    Get directory for snakemake wrappers.
    """
    return "file:%s/%s/wrapper.py" % (
        config["wrapper_directory"],
        wrapper,
    )


def getTestFolds(bins):
    """
    Get test folds for a given number of bins.
    """
    output = []
    for i in range(1, bins + 1):
        output += [i] * (bins - 1)
    return output


def getValidationFolds(bins):
    """
    Get validation folds for a given number of bins.
    """
    output = []
    for i in range(1, bins + 1):
        possible = list(range(1, bins + 1))
        possible.remove(i)
        output += possible

    return output


def getValidationFoldsForTest(test_fold, bins):
    output = list(range(1, bins + 1))
    output.remove(int(test_fold))
    return output
