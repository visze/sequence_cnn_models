##### SatMut #####


rule plot_satmut_get_region:
    """Retrieve the fasta sequence of a region"""
    conda:
        "../../envs/samtools.yaml"
    input:
        ref=config["reference"]["fasta"],
    output:
        "results/plot/satmut/input/{region}.fa.gz",
    params:
        region=lambda wc: "%s:%d-%d"
        % (
            getSatMutContig(wc.region),
            getSatMutStartPos(wc.region),
            getSatMutEndPos(wc.region),
        ),
    log:
        "logs/plot/satmut/get_region.{region}.log",
    shell:
        """
        samtools faidx {input.ref} {params.region} | bgzip -c > {output}
        """


rule plot_satmut_create_scores:
    """Rule to create ISM satmut scores"""
    conda:
        "../../envs/tensorflow.yml"
    input:
        satmut=lambda wc: getSatMutData(wc.region),
        sequence="results/plot/satmut/input/{region}.fa.gz",
        script=getScript("satMut_toISM.py"),
    output:
        output="results/plot/satmut/input/{region}.satmut.h5",
    params:
        startPos=lambda wc: getSatMutStartPos(wc.region),
        endPos=lambda wc: getSatMutEndPos(wc.region),
    log:
        "logs/plot/satmut/create_scores.{region}.log",
    shell:
        """
        python {input.script} --satmut {input.satmut}  \
        --start-position {params.startPos} --end-position {params.endPos} \
        --sequence {input.sequence} \
        --output {output} &> {log}
        """


rule plot_satmut_getRegionSplitList:
    """
    Rule to split a regions into smaller overlappig chunks of 200bp with a step of 150bp. 
    This is used to create the scores for the satmut regions. The scores are then combined 
    to create the final scores for the satmut regions.
    """
    output:
        "results/plot/satmut/input/ism/{region}.lst",
    params:
        contig=lambda wc: getSatMutContig(wc.region),
        startPos=lambda wc: getSatMutStartPos(wc.region)-1,
        endPos=lambda wc: getSatMutEndPos(wc.region),
        sequence_length=config["satmut"]["sequence_length"],
        step_size=config["satmut"]["step_size"],
    log:
        "logs/plot/satmut/getRegionSplitList.{region}.log",
    conda:
        "../../envs/samtools.yaml"
    shell:
        """
        bedtools makewindows \
        -b <(echo -e "{params.contig}\\t{params.startPos}\\t{params.endPos}") \
        -w {params.sequence_length} -s {params.step_size} | \
        awk '{{if ($3-$2 == 200) {{print $1":"$2+1"-"$3}} else {{print $1":"$2+1"-"$2+{params.sequence_length}}}}}' \
        > {output} 2> {log}
        """


rule plot_satmut_getRegionSplitFasta:
    """rule to collect the sequences for the satmut regions."""
    input:
        region="results/plot/satmut/input/ism/{region}.lst",
        ref=config["reference"]["fasta"],
    output:
        "results/plot/satmut/input/ism/{region}.fa.gz",
    log:
        "logs/plot/satmut/getRegionSplitFasta.{region}.log",
    conda:
        "../../envs/samtools.yaml"
    params:
        left=config["adapters"]["left"],
        right=config["adapters"]["right"],
        length=config["satmut"]["sequence_length"],
    shell:
        """
        samtools faidx {input.ref} -r {input.region} -n {params.length} | \
        sed -e '/^>/! s/^/{params.left}/' -e '/^>/!  s/$/{params.right}/'  | \
        bgzip -c > {output} 2> {log}
        """


rule plot_satmut_predict_region:
    """rule to generate ISM scores of a region"""
    input:
        model="results/training/model.regression.{test_fold}.{validation_fold}.json",
        weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
        sequences="results/plot/satmut/input/ism/{region}.fa.gz",
        script=getScript("ism.py"),
    output:
        scores="results/plot/satmut/input/ism/{region}.scores.{test_fold}.{validation_fold}.h5",
    params:
        sequence_length=230,
        mutation_length=200,
        mutation_start=16,
    log:
        "logs/plot/satmut/predict_region.{region}.{test_fold}.{validation_fold}.log",
    conda:
        "../../envs/tensorflow.yml"
    shell:
        """
        python {input.script} \
        --sequence {input.sequences} --sequence-length {params.sequence_length} --mutation-length {params.mutation_length} --mutation-start {params.mutation_start} \
        --model {input.model} --weights {input.weights} \
        --scores-output {output.scores} &> {log}
        """


rule plot_satmut_combine_predictions_val:
    conda:
        "../../envs/tfmodisco.yml"
    input:
        scores=lambda wc: expand(
            "results/plot/satmut/input/ism/{{region}}.scores.{{test_fold}}.{validation_fold}.h5",
            validation_fold=list(range(1, 11))[: int(wc.test_fold) - 1]
            + list(range(1, 11))[int(wc.test_fold) :],
        ),
        script=getScript("ism_concat.py"),
    output:
        hdf5="results/plot/satmut/input/ism/{region}.scores_comb_validation.{test_fold}.h5",
    log:
        "logs/plot/satmut/combine_predictions_val.{region}.{test_fold}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --output {output.hdf5}  &> {log}
        """


rule plot_satmut_combine_predictions_test:
    conda:
        "../../envs/tfmodisco.yml"
    input:
        scores=lambda wc: expand(
            "results/plot/satmut/input/ism/{{region}}.scores_comb_validation.{test_fold}.h5",
            test_fold=list(range(1, 11)),
        ),
        script=getScript("ism_concat.py"),
    output:
        hdf5="results/plot/satmut/input/{region}.ism_scores_split.h5",
    log:
        "logs/plot/satmut/combine_predictions_test.{region}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --output {output.hdf5}  &> {log}
        """


rule plot_satmut_combine_predictions_splits:
    conda:
        "../../envs/tfmodisco.yml"
    input:
        scores="results/plot/satmut/input/{region}.ism_scores_split.h5",
        script=getScript("ism_combine_splits.py"),
    output:
        "results/plot/satmut/input/{region}.ism_scores.h5",
    params:
        overlap=config["satmut"]["sequence_length"] - config["satmut"]["step_size"],
    log:
        "logs/plot/satmut/combine_predictions_splits.{region}.log",
    shell:
        """
        python {input.script} \
        --input {input.scores} --overlap {params.overlap} \
        --output {output}  &> {log}
        """


rule plot_satmut_plot_scores_sign:
    """Rule to plot ISM satmut scores"""
    conda:
        "../../envs/plot_ism_regions.yaml"
    input:
        satmut="results/plot/satmut/input/{region}.satmut.h5",
        scores="results/plot/satmut/input/{region}.ism_scores.h5",
        script=getScript("plot_satmut_ism.py"),
    output:
        heatmap="results/plot/satmut/{region}.{target}/heatmap.sign.pdf",
        scatter="results/plot/satmut/{region}.{target}/scatter.sign.pdf",
    params:
        target=lambda wc: wc.target,
        startPos=lambda wc: getSatMutStartPos(wc.region),
    log:
        "logs/plot/satmut/plot_scores_sign.{region}.{target}.log",
    shell:
        """
        python {input.script} --satmut {input.satmut} --score {input.scores} \
        --output-heatmap {output.heatmap} --output-scatter {output.scatter} \
        --num-bcs 10 --p-value 1e-5 \
        --start {params.startPos} \
         --target {params.target} &> {log}
        """

rule plot_satmut_plot_scores_all:
    """Rule to plot ISM satmut scores"""
    conda:
        "../../envs/plot_ism_regions.yaml"
    input:
        satmut="results/plot/satmut/input/{region}.satmut.h5",
        scores="results/plot/satmut/input/{region}.ism_scores.h5",
        script=getScript("plot_satmut_ism.py"),
    output:
        heatmap="results/plot/satmut/{region}.{target}/heatmap.all.pdf",
        scatter="results/plot/satmut/{region}.{target}/scatter.all.pdf",
    params:
        target=lambda wc: wc.target,
        startPos=lambda wc: getSatMutStartPos(wc.region),
    log:
        "logs/plot/satmut/plot_scores_all.{region}.{target}.log",
    shell:
        """
        python {input.script} --satmut {input.satmut} --score {input.scores} \
        --output-heatmap {output.heatmap} --output-scatter {output.scatter} \
        --start {params.startPos} \
         --target {params.target} &> {log}
        """

rule plot_satmut_get_satmut_ism_tsv:
    """Rule to generate a tsv with variants and score"""
    conda:
        "../../envs/plot_satmut_regions.yaml"
    input:
        satmut="results/plot/satmut/input/{region}.satmut.h5",
        scores="results/plot/satmut/input/{region}.ism_scores.h5",
        script=getScript("satmut_ism_tsv.py"),
    output:
        "results/plot/satmut/input/{region}.{target}.tsv.gz",
    params:
        target=lambda wc: wc.target,
        startPos=lambda wc: getSatMutStartPos(wc.region),
    log:
        "logs/plot/satmut/get_satmut_ism_tsv.{region}.{target}.log",
    shell:
        """
        python {input.script} --satmut {input.satmut} --score {input.scores} \
        --output {output} \
        --start {params.startPos} \
         --target {params.target} &> {log}
        """

rule plot_satmut:
    """Rule to plot ISM satmut scores"""
    conda:
        "../../envs/plot_satmut.yaml"
    input:
        scores="results/plot/satmut/input/{region}.{target}.tsv.gz",
        script=getScript("plot_satmut.R"),
    output:
        "results/plot/satmut/{region}.{target}/satmut.pdf",
    log:
        "logs/plot/satmut/plot_satmut.{region}.{target}.log",
    shell:
        """
        Rscript {input.script} --input {input.scores} \
        --output {output}  &> {log}
        """
