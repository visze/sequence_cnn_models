def getRegionFileFromSample(sample_id):
    """
    Return a region file from a sample.
    """
    print(sample_id)
    print(samples[sample_id])
    return samples[sample_id]["region_file"]


def getRegionFiles():
    """
    Get a list of all region files for all samples
    """
    return [getRegionFileFromSample(sample_id) for sample_id in samples.index]
