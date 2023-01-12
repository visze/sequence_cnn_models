if isRegression():

    rule correlation_regression:
        input:
            a="results/predictions/finalConcat.labels.tsv.gz",
        output:
            "results/correlation/regression.{prediction}.tsv.gz",
        params:
            value_a=lambda wc: wc.prediction,
            value_b=lambda wc: "%d.MEAN_prediction"
            % config["prediction"]["output_names"].index(wc.prediction),
        log:
            "logs/correlation/regression.{prediction}.log",
        wrapper:
            getWrapper("evaluate/correlation")
