

rule correlation_classification:
    input:
        a="results/test_predictions/predictions/{test_name}.labels.tsv.gz",
    output:
        "results/correlation/classification.{test_name}.{label}.{prediction}.tsv.gz",
    params:
        value_a=lambda wc: wc.label,
        value_b=lambda wc: wc.prediction,
    log:
        "logs/correlation/classification.{test_name}.{label}.{prediction}.log",
    wrapper:
        getWrapper("evaluate/correlation")


if isRegression():

    use rule correlation from wrappers as correlation_regression with:
        input:
            script=getWrapperPath("evaluate/correlation/correlate.py"),
            files=["results/predictions/finalConcat.labels.tsv.gz"],
        output:
            "results/correlation/regression.{prediction}.tsv.gz",
        params:
            value_a=lambda wc: wc.prediction,
            value_b=lambda wc: "%d.MEAN_prediction"
            % config["prediction"]["output_names"].index(wc.prediction),
        log:
            "logs/correlation/regression.{prediction}.log",

    use rule correlation from wrappers as correlation_prediction_regression with:
        input:
            script=getWrapperPath("evaluate/correlation/correlate.py"),
            files=[
                "results/test_predictions/predictions/finalMeanLabels.{test_name}.tsv.gz"
            ],
        output:
            "results/test_predictions/correlation/correlation.{test_name}.{prediction}.tsv.gz",
        params:
            value_a=lambda wc: wc.prediction,
            value_b=lambda wc: "%s.MEAN_prediction" % wc.prediction,
        log:
            "logs/correlation/prediction_regression.{test_name}.{prediction}.log",


# rule correlation_regression:
#     input:
#         a="results/predictions/finalConcat.labels.tsv.gz",
#     output:
#         "results/correlation/regression.{prediction}.tsv.gz",
#     params:
#         value_a=lambda wc: wc.prediction,
#         value_b=lambda wc: "%d.MEAN_prediction"
#         % config["prediction"]["output_names"].index(wc.prediction),
#     log:
#         "logs/correlation/regression.{prediction}.log",
#     wrapper:
#         getWrapper("evaluate/correlation")
