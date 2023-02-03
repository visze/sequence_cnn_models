satmut = {
    "LDLR": {
        "file": "GRCh38_LDLR_11089283-11089483.SNVs.tsv.gz",
        "startPos": 11089283,
        "contig": "chr19",
    },
    "SORT1": {
        "file": "GRCh38_SORT1.tsv.gz",
        "startPos": 109274652,
        "endPos": 109275251,
        "contig": "chr1",
    },
}


def getSatMutData(region):
    """
    Return a satmut data for a given region
    """
    return "%s/satmut_examples/%s" % (RESOURCES_DIR, satmut[region]["file"])


def getSatMutStartPos(region):
    """
    Return the start position of a satmut data for a given region
    """
    return satmut[region]["startPos"]


def getSatMutEndPos(region):
    """
    Return the end position of a satmut data for a given region
    """
    return satmut[region]["endPos"]


def getSatMutContig(region):
    """
    Return the contig of a satmut data for a given region
    """
    return satmut[region]["contig"]
