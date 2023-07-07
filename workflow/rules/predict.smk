import numpy as np


rule predict_input_combine_labels:
    """
    Combine labels file data with sequences from fasta file
    """
    input:
        fasta=lambda wc: config["prediction"]["samples"][wc.test_name]["fasta"],
        labels=lambda wc: config["prediction"]["samples"][wc.test_name]["labels"],
    output:
        "results/test_predictions/inputs/{test_name}.tsv.gz",
    log:
        "logs/predict/input_combine_labels.{test_name}.log",
    shell:
        """
        join -t $'\\t' \
        <(
            cat {input.fasta} | \
            awk -v "OFS=\\t" '/^>/ {{printf("%s\\t",substr($1,2));next; }} {{printf("%s\\n",$1);}}' | \
            sed 's/_F\\t/\\t/g' | \
            sort -k1,1;
        ) \
        <(cat {input.labels} | sed 's/\\r//' | awk -v "OFS=\\t" '{{printf "%s\\t", $1; for(i=2; i<=NF; i++) {{ printf "%s\\t", $i}}; printf "\\n"}}' | sort -k1,1) | \
        gzip -c > {output} 2> {log}
        """


if not isRegression():

    rule predict_prediction:
        input:
            test_file=lambda wc: getPredictionTestFile(wc.test_name),
            model=lambda wc: getModelPath()["model"],
            weights=lambda wc: getModelPath()["weights"],
            script=getScript("predict.py"),
        output:
            "results/test_predictions/predictions/{test_name}.tsv.gz",
        log:
            "logs/predict/prediction.{test_name}.log",
        params:
            input_file=lambda wc: "--test-fasta-file"
            if isFastaFile(wc.test_name)
            else "--test-file",
            length=config["input"]["length"],
            augmentation=lambda wc: "--use-augmentation"
            if config["prediction"]["samples"][wc.test_name]["augmentation"]
            or "augment_on" in config["prediction"]["samples"][wc.test_name]
            else "--no-augmentation",
            augment_on=lambda wc: "--augment-on %d %d"
            % (
                config["prediction"]["samples"][wc.test_name]["augment_on"][0],
                config["prediction"]["samples"][wc.test_name]["augment_on"][1],
            )
            if "augment_on" in config["prediction"]["samples"][wc.test_name]
            else "",
            output_names=" ".join(
                [
                    "--prediction-name %s" % i
                    for i in config["prediction"]["output_names"]
                ]
            ),
        conda:
            "../envs/tensorflow.yml"
        threads: 1
        shell:
            """
            python {input.script} \
            {params.input_file} {input.test_file} \
            --model {input.model} --weights {input.weights} \
            --sequence-length {params.length} \
            {params.augmentation} {params.augment_on} \
            {params.output_names} \
            --output {output} &> {log}
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

    rule predict_labels:
        input:
            labels=lambda wc: config["prediction"]["samples"][wc.test_name]["labels"],
            prediction="results/test_predictions/predictions/{test_name}.tsv.gz",
        output:
            "results/test_predictions/predictions/{test_name}.labels.tsv.gz",
        params:
            index="ID",
        log:
            "logs/predict/labels.{test_name}.log",
        wrapper:
            getWrapper("file_manipulation/concat")


if isRegression():

    rule predict_prediction:
        input:
            test_file=lambda wc: getPredictionTestFile(wc.test_name),
            model=lambda wc: getModelPath()["model"],
            weights=lambda wc: getModelPath()["weights"],
            script=getScript("predict.py"),
        output:
            "results/test_predictions/predictions/perModel/{test_name}.{test_fold}.{validation_fold}.tsv.gz",
        log:
            "logs/predict/prediction.{test_name}.{test_fold}.{validation_fold}.log",
        params:
            input_file=lambda wc: "--test-fasta-file"
            if isFastaFile(wc.test_name)
            else "--test-file",
            length=config["input"]["length"],
            augmentation=lambda wc: "--use-augmentation"
            if config["prediction"]["samples"][wc.test_name]["augmentation"]
            or "augment_on" in config["prediction"]["samples"][wc.test_name]
            else "--no-augmentation",
            augment_on=lambda wc: "--augment-on %d %d"
            % (
                config["prediction"]["samples"][wc.test_name]["augment_on"][0],
                config["prediction"]["samples"][wc.test_name]["augment_on"][1],
            )
            if "augment_on" in config["prediction"]["samples"][wc.test_name]
            else "",
            prediction_name=lambda wc: wc.validation_fold,
        conda:
            "../envs/tensorflow.yml"
        threads: 1
        shell:
            """
            python {input.script} \
            {params.input_file} {input.test_file} \
            --model {input.model} --weights {input.weights} \
            --sequence-length {params.length} \
            {params.augmentation} {params.augment_on} \
            --prediction-name {params.prediction_name} \
            --output {output} &> {log}
            """

    rule predict_predict_concat:
        input:
            lambda wc: expand(
                "results/test_predictions/predictions/perModel/{{test_name}}.{{test_fold}}.{validation_fold}.tsv.gz",
                validation_fold=getValidationFoldsForTest(
                    wc.test_fold, config["training"]["folds"]
                ),
            ),
        output:
            temp(
                "results/test_predictions/predictions/concat.{test_name}.{test_fold}.tsv.gz"
            ),
        params:
            index="ID",
        log:
            "logs/predict/predict_concat.{test_name}.{test_fold}.log",
        wrapper:
            getWrapper("file_manipulation/concat")

    rule predict_prediction_mean:
        input:
            "results/test_predictions/predictions/concat.{test_name}.{test_fold}.tsv.gz",
        output:
            "results/test_predictions/predictions/mean.{test_name}.{test_fold}.tsv.gz",
        params:
            columns=lambda wc: getValidationFoldsForTest(
                wc.test_fold, config["training"]["folds"]
            ),
            new_columns=lambda wc: np.array(
                expand(
                    "{output}.{method}",
                    method=["MEAN_prediction", "STD_prediction"],
                    output=wc.test_fold,
                )
            ).reshape(2, config["prediction"]["output_size"]),
            operations=["mean", "std"],
        log:
            "logs/predict/prediction_mean.{test_name}.{test_fold}.log",
        wrapper:
            getWrapper("file_manipulation/summarize_columns")

    rule predict_prediction_finalConcat:
        input:
            expand(
                "results/test_predictions/predictions/mean.{{test_name}}.{test_fold}.tsv.gz",
                test_fold=range(1, config["training"]["folds"] + 1),
            ),
        output:
            "results/test_predictions/predictions/finalConcat.{test_name}.tsv.gz",
        params:
            index="ID",
        log:
            "logs/predict/prediction_finalConcat.{test_name}.log",
        wrapper:
            getWrapper("file_manipulation/concat")

    rule predict_prediction_finalmean:
        input:
            "results/test_predictions/predictions/finalConcat.{test_name}.tsv.gz",
        output:
            "results/test_predictions/predictions/finalMean.{test_name}.tsv.gz",
        params:
            columns=lambda wc: expand(
                "{output}.{method}",
                method=["MEAN_prediction", "STD_prediction"],
                output=range(1, config["training"]["folds"] + 1),
            ),
            new_columns=lambda wc: np.array(
                expand(
                    "{output}.{method}",
                    method=["MEAN_prediction", "STD_prediction"],
                    output=config["prediction"]["output_names"],
                )
            ).reshape(2, config["prediction"]["output_size"]),
            operations=["mean", "std"],
        log:
            "logs/predict/prediction_finalmean.{test_name}.log",
        wrapper:
            getWrapper("file_manipulation/summarize_columns")

    rule predict_prediction_labels:
        input:
            labels=lambda wc: config["prediction"]["samples"][wc.test_name]["labels"],
            prediction="results/test_predictions/predictions/finalMean.{test_name}.tsv.gz",
        output:
            "results/test_predictions/predictions/finalMeanLabels.{test_name}.tsv.gz",
        params:
            index="ID",
        log:
            "logs/predict/regression_labels.{test_name}.log",
        wrapper:
            getWrapper("file_manipulation/concat")

    ##################
    ### using fold ###
    ##################

    rule predict_regression:
        input:
            test_file="results/regression_input/regression.test.{test_fold}.{validation_fold}.tsv.gz",
            model=lambda wc: getModelPath(wc.test_fold, wc.validation_fold)["model"],
            weights=lambda wc: getModelPath(wc.test_fold, wc.validation_fold)["weights"],
            script=getScript("predict_regression.py"),
        output:
            temp("results/predictions/prediction.{test_fold}.{validation_fold}.tsv.gz"),
        params:
            prediction_name=lambda wc: wc.validation_fold,
            augmentation="--use-augmentation"
            if config["prediction"]["augmentation"]
            or "augment_on" in config["prediction"]
            else "--no-augmentation",
            augment_on="--augment-on %d %d"
            % (
                config["prediction"]["augment_on"][0],
                config["prediction"]["augment_on"][1],
            )
            if "augment_on" in config["prediction"]
            else "",
            legnet="--legnet-model"
            if config["training"]["model"] == "legnet"
            else "--no-legnet-model",
        log:
            "logs/predict/regression.{test_fold}.{validation_fold}.log",
        conda:
            "legnet" if config["training"][
            "model"
            ] == "legnet" else "../envs/tensorflow.yml"
        threads: 1
        shell:
            """
            python {input.script} \
            --test {input.test_file} \
            --model {input.model} --weights {input.weights} \
            {params.augmentation} {params.augment_on} \
            {params.legnet} \
            --prediction-name {params.prediction_name} \
            --output {output} &> {log}
            """

    rule predict_regression_concat:
        input:
            lambda wc: expand(
                "results/predictions/prediction.{{test_fold}}.{validation_fold}.tsv.gz",
                validation_fold=getValidationFoldsForTest(
                    wc.test_fold, config["training"]["folds"]
                ),
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
            columns=lambda wc: getColumnsForMean(wc.test_fold),
            new_columns=lambda wc: np.array(
                expand(
                    "{output}.{method}",
                    method=["MEAN_prediction", "STD_prediction"],
                    output=range(0, config["prediction"]["output_size"]),
                )
            ).reshape(2, config["prediction"]["output_size"]),
            operations=["mean", "std"],
        log:
            "logs/predict/regression_mean.{test_fold}.log",
        wrapper:
            getWrapper("file_manipulation/summarize_columns")

    rule predict_regression_finalConcat:
        input:
            expand(
                "results/predictions/mean.{test_fold}.tsv.gz",
                test_fold=range(1, config["training"]["folds"] + 1),
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

    rule predict_regression_labels_rename:
        input:
            "results/predictions/finalConcat.labels.tsv.gz",
        output:
            temp("results/predictions/finalConcat.labels.rename.tsv.gz"),
        params:
            columns=getlabelsForRename(),
        log:
            "logs/predict/regression_labelss_rename.log",
        wrapper:
            getWrapper("file_manipulation/rename")

    rule predict_regression_labels_clean:
        input:
            "results/predictions/finalConcat.labels.rename.tsv.gz",
        output:
            "results/predictions/finalConcat.labels.cleaned.tsv.gz",
        params:
            columns=["ID", "BIN"] + list(getlabelsForRename().values()),
        log:
            "logs/predict/regression_labels_clean.log",
        wrapper:
            getWrapper("file_manipulation/extract_columns")
