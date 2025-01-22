wildcard_constraints:
    region=r"[^/.]+",
    sequence_length=r"\d+",
    target=r"\d+",


rule plot_ism_get_region:
    """Retrieve the fasta sequence of a region"""
    conda:
        "../envs/samtools.yaml"
    input:
        ref=config["reference"]["fasta"],
    output:
        "results/plot_ism/input/{region}.fa.gz",
    params:
        region=lambda wc: config["ism_regions"][wc.region],
    log:
        "logs/plot_ism/region.{region}.log",
    shell:
        """
        samtools faidx {input.ref} {params.region} | bgzip -c > {output}
        """


rule plot_ism_predict_region:
    """rule to generate ISM scores of a region"""
    input:
        model="results/training/model.regression.{test_fold}.{validation_fold}.json",
        weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
        sequences="results/plot_ism/input/{region}.fa.gz",
        script=getScript("ism.py"),
    output:
        scores="results/plot_ism/input/scores/{region}.scores.{test_fold}.{validation_fold}.h5",
    params:
        sequence_length=config["prediction"]["sequence_length"],
        mutation_length=config["prediction"]["mutation_length"],
        mutation_start=config["prediction"]["mutation_start"],
        mask=" ".join(
            ["--mask %d %d " % (i[0], i[1]) for i in config["prediction"]["mask"]]
        ),
    log:
        "logs/plot_ism/predict_region.{region}.{test_fold}.{validation_fold}.log",
    conda:
        "../envs/tensorflow.yml"
    shell:
        """
        python {input.script} \
        --sequence {input.sequences} --sequence-length {params.sequence_length} --mutation-length {params.mutation_length} --mutation-start {params.mutation_start} \
        --model {input.model} --weights {input.weights} \
        {params.mask} \
        --scores-output {output.scores} &> {log}
        """


rule plot_ism_combine_predictions_val:
    conda:
        "../envs/tfmodisco.yml"
    input:
        scores=lambda wc: expand(
            "results/plot_ism/input/scores/{{region}}.scores.{{test_fold}}.{validation_fold}.h5",
            validation_fold=list(range(1, 11))[: int(wc.test_fold) - 1]
            + list(range(1, 11))[int(wc.test_fold) :],
        ),
        script=getScript("ism_concat.py"),
    output:
        hdf5="results/plot_ism/input/scores/{region}.scores_comb_validation.{test_fold}.h5",
    log:
        "logs/plot_ism/combine_predictions_val.{region}.{test_fold}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --output {output.hdf5}  &> {log}
        """


rule plot_ism_combine_predictions_test:
    conda:
        "../envs/tfmodisco.yml"
    input:
        scores=lambda wc: expand(
            "results/plot_ism/input/scores/{{region}}.scores_comb_validation.{test_fold}.h5",
            test_fold=list(range(1, 11)),
        ),
        script=getScript("ism_concat.py"),
    output:
        hdf5="results/plot_ism/input/{region}.ism_scores.h5",
    log:
        "logs/plot_ism/combine_predictions_test.{region}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --output {output.hdf5}  &> {log}
        """


rule plot_ism_region:
    """Rule to plot ISMs of a region"""
    conda:
        "../envs/plot_ism_regions.yaml"
    input:
        scores="results/plot_ism/input/{region}.ism_scores.h5",
        script=getScript("plot_ism.py"),
    output:
        "results/plot_ism/plots/{region}.{target}/example_0.pdf",
        output_dir=directory("results/plot_ism/plots/{region}.{target}"),
    params:
        target=lambda wc: wc.target,
    log:
        "logs/plot_ism/region.{region}.{target}.log",
    shell:
        """
        python {input.script} --score {input.scores} --target {params.target} --n-plots 0 --output {output.output_dir} &> {log}
        """
