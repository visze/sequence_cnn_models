if isRegression():

    rule correlation_regression:
        input:
            a="results/predictions/finalConcat.labels.tsv.gz",
        output:
            "results/correlation/regression.tsv.gz",
        params:
            value_a="MEAN",
            value_b="MEAN_prediction",
        log:
            "logs/correlation/regression.log",
        wrapper:
            getWrapper("evaluate/correlation")
