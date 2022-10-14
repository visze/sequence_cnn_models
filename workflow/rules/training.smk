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
        lrSheduler="--no-learning-rate-sheduler",
        earlyStopping="--use-early-stopping",
    conda:
        "../envs/tensorflow.yml"
    threads: 25
    shell:
        """
        python {input.script} \
        --fasta-file {input.fasta_file} \
        --intervals-train {input.intervals_train} --intervals-validation {input.intervals_validation} \
        --model {output.model} --weights {output.weights} \
        --val-acc {output.validation_acc} \
        --fit-log {output.fit_log} \
        --batch-size {params.batchSize} --epochs {params.epochs} \
        --learning-rate {params.learningRate} \
        {params.lrSheduler} \
        {params.earlyStopping} 
        """
