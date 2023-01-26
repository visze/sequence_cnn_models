wildcard_constraints:
    region="[^/.]+",
    sequence_length="\d+",
    target="\d+",


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
        sequence_length=230,
        mutation_length=200,
        mutation_start=16,
    log:
        "logs/plot_ism/predict_region.{region}.{test_fold}.{validation_fold}.log",
    conda:
        "../envs/tensorflow.yml"
    shell:
        """
        python {input.script} \
        --sequence {input.sequences} --sequence-length {params.sequence_length} --mutation-length {params.mutation_length} --mutation-start {params.mutation_start} \
        --model {input.model} --weights {input.weights} \
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


rule plot_ism_satmut_get_region:
    """Retrieve the fasta sequence of a region"""
    conda:
        "../envs/samtools.yaml"
    input:
        ref=config["reference"]["fasta"],
    output:
        "results/plot_ism/input/satmut/{region}.{sequence_length}.fa.gz",
    params:
        region=lambda wc: "%s:%d-%d"
        % (
            getSatMutContig(wc.region),
            getSatMutStartPos(wc.region),
            getSatMutStartPos(wc.region) + int(wc.sequence_length) - 1,
        ),
    log:
        "logs/plot_ism/satmut_region.{region}.{sequence_length}.log",
    shell:
        """
        samtools faidx {input.ref} {params.region} | bgzip -c > {output}
        """


rule plot_ism_satmut_create_scores:
    """Rule to create ISM satmut scores"""
    conda:
        "../envs/tensorflow.yml"
    input:
        satmut=lambda wc: getSatMutData(wc.region),
        sequence="results/plot_ism/input/satmut/{region}.{sequence_length}.fa.gz",
        script=getScript("satMut_toISM.py"),
    output:
        output="results/plot_ism/input/satmut/{region}.satmut.{sequence_length}.h5",
    params:
        startPos=lambda wc: getSatMutStartPos(wc.region),  #11089283
    log:
        "logs/plot_ism/atmut_create_scores.{region}.{sequence_length}.log",
    shell:
        """
        python {input.script} --satmut {input.satmut}  \
        --start-position {params.startPos} --sequence {input.sequence} \
        --sequence-length 200 --output {output} &> {log}
        """


rule plot_ism_satmut_plot_scores:
    """Rule to plot ISM satmut scores"""
    conda:
        "../envs/plot_ism_regions.yaml"
    input:
        satmut="results/plot_ism/input/satmut/{region}.satmut.{sequence_length}.h5",
        scores="results/plot_ism/input/{region}.ism_scores.h5",
        script=getScript("plot_satmut_ism.py"),
    output:
        heatmap="results/plot_ism/plots/{region}.{target}/satmut/heatmap.{sequence_length}.pdf",
        scatter="results/plot_ism/plots/{region}.{target}/satmut/scatter.{sequence_length}.pdf",
    params:
        target=lambda wc: wc.target,
    log:
        "logs/plot_ism/atmut_create_scores.{region}.{target}.{sequence_length}.log",
    # wildcard_constraints:
    #     region="[^.]+",
    shell:
        """
        python {input.script} --satmut {input.satmut} --score {input.scores} \
        --output-heatmap {output.heatmap} --output-scatter {output.scatter} \
        --num-bcs 10 --p-value 1e-5 \
         --target {params.target} &> {log}
        """
