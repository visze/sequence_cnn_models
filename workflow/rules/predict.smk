if not isRegression():

    rule predict_fromFasta:
        input:
            test_file=lambda wc: tests.loc[wc.test_name]["file"],
            model="results/training/model.json",
            weights="results/training/weights.h5",
            script=getScript("predict.py"),
        output:
            "results/predictions/{test_name}.tsv.gz",
        log:
            "logs/predict/fromFasta.{test_name}.log",
        params:
            length=lambda wc: config["input"]["length"],
        conda:
            "../envs/tensorflow.yml"
        threads: 1
        shell:
            """
            python {input.script} \
            --test {input.test_file} \
            --model {input.model} --weights {input.weights} \
            --sequence-length {params.length} \
            --output {output} > {log}
            """

    rule predict_add_labels:
        conda:
            "../envs/default.yml"
        input:
            "results/predictions/{test_name}.tsv.gz",
        output:
            "results/predictions/{test_name}.labeled.tsv.gz",
        params:
            labels="\\t".join(["id"] + list(samples.index) + ["gc_matched_background"]),
        log:
            "logs/predict/add_labels.{test_name}.log",
        shell:
            """
            (echo -e "{params.labels}"; zcat {input}) | \
            gzip -c > {output}  2> {log}
            """


if isRegression():

    rule predict_regression:
        input:
            test_file="results/regression_input/regression.test.{test_fold}.{validation_fold}.tsv.gz",
            model="results/training/model.regression.{test_fold}.{validation_fold}.json",
            weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
            script=getScript("predict_regression.py"),
        output:
            temp("results/predictions/prediction.{test_fold}.{validation_fold}.tsv.gz"),
        params:
            prediction_name=lambda wc: wc.validation_fold,
        log:
            "logs/predict/regression.{test_fold}.{validation_fold}.log",
        conda:
            "../envs/tensorflow.yml"
        threads: 1
        shell:
            """
            python {input.script} \
            --test {input.test_file} \
            --model {input.model} --weights {input.weights} \
            --prediction-name {params.prediction_name} \
            --output {output} &> {log}
            """

    rule predict_regression_concat:
        input:
            lambda wc: expand(
                "results/predictions/prediction.{{test_fold}}.{validation_fold}.tsv.gz",
                validation_fold=getValidationFoldsForTest(wc.test_fold, 10),
            ),
        output:
            temp("results/predictions/concat.{test_fold}.tsv.gz"),
        params:
            index="ID",
        log:
            "logs/predict/regression_concat.{test_fold}.log",
        wrapper:
            getWrapper("file_manipulation/concat")

    rule predict_regression_mean:
        input:
            "results/predictions/concat.{test_fold}.tsv.gz",
        output:
            "results/predictions/mean.{test_fold}.tsv.gz",
        params:
            columns=lambda wc: getValidationFoldsForTest(wc.test_fold, 10),
            new_columns=["MEAN_prediction", "STD_prediction"],
            operations=["mean", "std"],
        log:
            "logs/predict/regression_mean.{test_fold}.log",
        wrapper:
            getWrapper("file_manipulation/summarize_columns")

    rule predict_regression_finalConcat:
        input:
            expand(
                "results/predictions/mean.{test_fold}.tsv.gz", test_fold=range(1, 11)
            ),
        output:
            "results/predictions/finalConcat.tsv.gz",
        log:
            "logs/predict/regression_finalConcat.log",
        wrapper:
            getWrapper("file_manipulation/concat")

    rule predict_regression_labels:
        input:
            labels=config["input"]["labels"],
            prediction="results/predictions/finalConcat.tsv.gz",
        output:
            "results/predictions/finalConcat.labels.tsv.gz",
        params:
            index="ID",
        log:
            "logs/predict/regression_labels.log",
        wrapper:
            getWrapper("file_manipulation/concat")
