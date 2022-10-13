rule training_multitask:
    input:
        seq_training="results/sequence_selection/sequence.annotated_bidirectional.training.fa.gz",
        seq_validation="results/sequence_selection/sequence.annotated_bidirectional.validation.fa.gz",
        labels_training="results/sequence_selection/sequence.annotated_bidirectional.training.bed.gz"
        labels_validation="results/sequence_selection/sequence.annotated_bidirectional.validation.bed.gz"
        script=getScript("train_multitask.py"),
    output:
        model="results/training/model.json",
        weights="results/training/weights.h5",
        validation_acc="results/training/validation_acc.tsv",
        validation_pred="results/training/validation_pred.tsv",
        fit_log="results/training/fit_log.tsv"
    log:
        "results/logs/training/multitask.log"
    params:
        epochs=100,
        batchSize=128,
        positiveLabelThreshold=0.8,
        learningRate=.001,
        lrSheduler="--no-learning-rate-sheduler",
        earlyStopping="--use-early-stopping",
    conda: "envs/gpu.yml"
    threads: 25
    shell: 
    """
    python scripts/train_multitask.py \
    --seq-train {input.seq_training} --seq-validation {input.seq_validation} \
    --labels-train {input.labels_training} --labels-validation {input.labels_validation} \
    --model {output.model} --weights {output.weights} \
    --val-acc {output.validation_acc} -val-pred {output.validation_pred} \
    --fit-log {output.fit_log} \
    --batch-size {params.batchSize} --epochs {params.epochs} \
    --label-threshold {params.positiveLabelThreshold} \
    --learning-rate {params.learningRate} \
    {params.lrSheduler} \
    {params.earlyStopping} \
    """"
