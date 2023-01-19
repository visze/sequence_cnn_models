if not isRegression():

    rule training_multitask:
        input:
            fasta_file=config["reference"]["fasta"],
            intervals_train="results/sequence_selection/regions.annotated_bidirectional.training.bed.gz",
            intervals_validation="results/sequence_selection/regions.annotated_bidirectional.validation.bed.gz",
            script=getScript("train.py"),
        output:
            model="results/training/model.json",
            weights="results/training/weights.h5",
            validation_acc="results/training/validation_acc.tsv",
            fit_log="results/training/fit_log.tsv",
        log:
            "logs/training/multitask.log",
        params:
            epochs=100,
            batchSize=128,
            learningRate=0.001,
            loss=config["training"]["loss"],
            model=config["training"]["model"],
            lrSheduler="--no-learning-rate-sheduler",
            earlyStopping="--use-early-stopping",
        conda:
            "../envs/tensorflow.yml"
        threads: 25
        shell:
            """
            python {input.script} \
            --fasta-file {input.fasta_file} \
            --train-input {input.intervals_train} --validation-input {input.intervals_validation} \
            --model-mode classification \
            --model-type {params.model} \
            --model {output.model} --weights {output.weights} \
            --val-acc {output.validation_acc} \
            --fit-log {output.fit_log} \
            --loss {params.loss} \
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
            batchSize=128,
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
