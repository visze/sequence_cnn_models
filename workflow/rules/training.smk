if not isRegression():

    rule training_classification:
        input:
            fasta_file=config["reference"]["fasta"],
            train="results/sequence_selection/regions.annotated.training.bed.gz",
            validation="results/sequence_selection/regions.annotated.validation.bed.gz",
            script=getScript("train.py"),
        output:
            model="results/training/model.classification.json",
            weights="results/training/weights.classification.h5",
            validation_acc="results/training/validation_acc.classification.tsv",
            fit_log="results/training/fit_log.classification.tsv",
        log:
            "logs/training/classification.log",
        params:
            epochs=100,
            batchSize=config["training"]["batch_size"],
            learningRate=0.001,
            loss=config["training"]["loss"],
            model=config["training"]["model"],
            lrSheduler="--no-learning-rate-sheduler",
            earlyStopping="--use-early-stopping",
            augmentation="--use-augmentation"
            if config["training"]["augmentation"] or "augment_on" in config["training"]
            else "--no-augmentation",
            augment_on="--augment-on %d %d"
            % (
                config["training"]["augment_on"][0],
                config["training"]["augment_on"][1],
            )
            if "augment_on" in config["training"]
            else "",
        conda:
            "../envs/tensorflow.yml"
        threads: 25
        shell:
            """
            python {input.script} \
            --train-input {input.train} --validation-input {input.validation} \
            --fasta-file {input.fasta_file} \
            --model-type {params.model} \
            --model-mode classification \
            --model {output.model} --weights {output.weights} \
            --val-acc {output.validation_acc} \
            --fit-log {output.fit_log} \
            --loss {params.loss} \
            {params.augmentation} {params.augment_on} \
            --batch-size {params.batchSize} --epochs {params.epochs} \
            --learning-rate {params.learningRate} \
            {params.lrSheduler} \
            {params.earlyStopping} &> {log}
            """

else:

    rule training_regression:
        input:
            train="results/regression_input/regression.training.{test_fold}.{validation_fold}.tsv.gz",
            validation="results/regression_input/regression.validation.{test_fold}.{validation_fold}.tsv.gz",
            script=getScript("train.py"),
        output:
            model="results/training/model.regression.{test_fold}.{validation_fold}.json",
            weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
            validation_acc="results/training/validation_acc.regression.{test_fold}.{validation_fold}.tsv",
            fit_log="results/training/fit_log.regression.{test_fold}.{validation_fold}.tsv",
        log:
            "logs/training/regression.{test_fold}.{validation_fold}.log",
        params:
            epochs=100,
            batchSize=config["training"]["batch_size"],
            learningRate=0.001,
            loss=config["training"]["loss"],
            model=config["training"]["model"],
            lrSheduler="--no-learning-rate-sheduler",
            earlyStopping="--use-early-stopping",
            augmentation="--use-augmentation"
            if config["training"]["augmentation"] or "augment_on" in config["training"]
            else "--no-augmentation",
            augment_on="--augment-on %d %d"
            % (
                config["training"]["augment_on"][0],
                config["training"]["augment_on"][1],
            )
            if "augment_on" in config["training"]
            else "",
        conda:
            "../envs/tensorflow.yml"
        threads: 25
        shell:
            """
            python {input.script} \
            --train-input {input.train} --validation-input {input.validation} \
            --model-type {params.model} \
            --model-mode regression \
            --model {output.model} --weights {output.weights} \
            --val-acc {output.validation_acc} \
            --fit-log {output.fit_log} \
            --loss {params.loss} \
            {params.augmentation} {params.augment_on} \
            --batch-size {params.batchSize} --epochs {params.epochs} \
            --learning-rate {params.learningRate} \
            {params.lrSheduler} \
            {params.earlyStopping} &> {log}
            """
