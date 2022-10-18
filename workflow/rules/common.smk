################################
#### Global functions       ####
################################
from snakemake.workflow import srcdir

SCRIPTS_DIR = srcdir("../scripts")


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
