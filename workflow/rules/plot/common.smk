satmut = {
    "F9": {
        "file": "GRCh38_F9.tsv.gz",
        "startPos": 139530463,
        "endPos": 139530765,
        "contig": "chrX",
    },
    "PKLR_24": {
        "file": "GRCh38_PKLR_24.tsv.gz",
        "startPos": 155301395,
        "endPos": 155301864,
        "contig": "chr1",
    },
    "PKLR_48": {
        "file": "GRCh38_PKLR_48.tsv.gz",
        "startPos": 155301395,
        "endPos": 155301864,
        "contig": "chr1",
    },
    "TERT_HEK": {
        "file": "GRCh38_TERT_HEK.tsv.gz",
        "startPos": 1294989,
        "endPos": 1295247,
        "contig": "chr5",
    },
    "TERT_GBM": {
        "file": "GRCh38_TERT_GBM.tsv.gz",
        "startPos": 1294989,
        "endPos": 1295247,
        "contig": "chr5",
    },
    "LDLR": {
        "file": "GRCh38_LDLR.tsv.gz",
        "startPos": 11089231,
        "endPos": 11089548,
        "contig": "chr19",
    },
    "LDLR_2": {
        "file": "GRCh38_LDLR_2.tsv.gz",
        "startPos": 11089231,
        "endPos": 11089548,
        "contig": "chr19",
    },
    "SORT1": {
        "file": "GRCh38_SORT1.tsv.gz",
        "startPos": 109274652,
        "endPos": 139530765,
        "endPos": 109275251,
        "contig": "chr1",
    },
    "SORT1_2": {
        "file": "GRCh38_SORT1_2.tsv.gz",
        "startPos": 109274652,
        "endPos": 139530765,
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
