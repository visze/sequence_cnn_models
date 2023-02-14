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
        "endPos": 109275251,
        "contig": "chr1",
    },
    "SORT1_2": {
        "file": "GRCh38_SORT1_2.tsv.gz",
        "startPos": 109274652,
        "endPos": 109275251,
        "contig": "chr1",
    },
    "FOXE1": {
        "file": "GRCh38_FOXE1.tsv.gz",
        "startPos": 97853255,
        "endPos": 97853854,
        "contig": "chr9",
    },
    "GP1BB": {
        "file": "GRCh38_GP1BA.tsv.gz",
        "startPos": 19723266,
        "endPos": 19723650,
        "contig": "chr22",
    },
    "HBB": {
        "file": "GRCh38_HBB.tsv.gz",
        "startPos": 5227022,
        "endPos": 5227208,
        "contig": "chr11",
    },
    "HNF4A": {
        "file": "GRCh38_HNF4A.tsv.gz",
        "startPos": 44355520,
        "endPos": 44355804,
        "contig": "chr20",
    },
    "MSMB": {
        "file": "GRCh38_MSMB.tsv.gz",
        "startPos": 46046244,
        "endPos": 46046834,
        "contig": "chr10",
    },
    "BCL11A": {
        "file": "GRCh38_BCL11A.tsv.gz",
        "startPos": 60494940,
        "endPos": 60495539,
        "contig": "chr2",
    },
    "IRF4": {
        "file": "GRCh38_IRF4.tsv.gz",
        "startPos": 396143,
        "endPos": 396593,
        "contig": "chr6",
    },
    "IRF6": {
        "file": "GRCh38_IRF6.tsv.gz",
        "startPos": 209815790,
        "endPos": 209816390,
        "contig": "chr1",
    },
    "MYC_rs6983267": {
        "file": "GRCh38_MYC_rs6983267.tsv.gz",
        "startPos": 127400829,
        "endPos": 127401428,
        "contig": "chr8",
    },
    "MYC_rs11986220": {
        "file": "GRCh38_MYC_rs11986220.tsv.gz",
        "startPos": 127519270,
        "endPos": 127519732,
        "contig": "chr8",
    },
    "RET": {
        "file": "GRCh38_RET.tsv.gz",
        "startPos": 43086479,
        "endPos": 43087078,
        "contig": "chr10",
    },
    "TCF7L2": {
        "file": "GRCh38_TCF7L2.tsv.gz",
        "startPos": 112998240,
        "endPos": 112998839,
        "contig": "chr10",
    },
    "UC88": {
        "file": "GRCh38_UC88.tsv.gz",
        "startPos": 161238408,
        "endPos": 161238997,
        "contig": "chr2",
    },
    "ZFAND3": {
        "file": "GRCh38_ZFAND3.tsv.gz",
        "startPos": 37807499,
        "endPos": 37808077,
        "contig": "chr6",
    },
    "ZRS_13": {
        "file": "GRCh38_ZRSh-13.tsv.gz",
        "startPos": 156791119,
        "endPos": 156791603,
        "contig": "chr7",
    },
    "ZRS_13h2": {
        "file": "GRCh38_ZRSh-13h2.tsv.gz",
        "startPos": 156791119,
        "endPos": 156791603,
        "contig": "chr7",
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
