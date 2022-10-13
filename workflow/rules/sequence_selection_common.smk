def getRegionFileFromSample(sample_id):
    """
    Return a region file from a sample.
    """
    return samples.loc[sample_id]["region_file"]


def getRegionFiles():
    """
    Get a list of all region files for all samples
    """
    return [getRegionFileFromSample(sample_id) for sample_id in samples.index]
