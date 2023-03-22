########################
### extract kernels ####
########################
rule model_interpretation_extract_kernels:
    """
    Extract kernels from the first layer using the snakemake wrapper
    """
    input:
        "results/training/weights.regression.{test_fold}.{validation_fold}.h5",
    output:
        "results/model_interpretation/kernels/conv1.{test_fold}.{validation_fold}.txt",
    params:
        "conv1",
    log:
        "logs/model_interpretation/extract_kernels.{test_fold}.{validation_fold}.log",
    wrapper:
        getWrapper("dnn/extract_convolutional_kernels")


rule model_interpretation_convertToMemeFormat:
    input:
        "results/model_interpretation/kernels/conv1.{test_fold}.{validation_fold}.txt",
    output:
        "results/model_interpretation/kernels/conv1.{test_fold}.{validation_fold}.meme",
    wrapper:
        getWrapper("meme/uniprobe2meme")


rule model_interpretation_findMotifsForConvKernels:
    input:
        "results/model_interpretation/kernels/conv1.{test_fold}.{validation_fold}.meme",
        "results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    output:
        "results/model_interpretation/kernels/conv1.{test_fold}.{validation_fold}.motifs_hits.JASPAR_CORE_NR.tsv",
    wrapper:
        getWrapper("meme/tomtom")


### ISM

rule model_interpretation_getTestData:
    conda:
        "../envs/default.yml"
    input:
        "results/regression_input/regression.test.{test_fold}.{validation_fold}.tsv.gz",
    output:
        "results/model_interpretation/input/regression.test.{test_fold}.{validation_fold}.fa.gz",
    shell:
        """
        zcat {input} | awk -v 'OFS=\\t' '{{print ">"$1"\\n"$2}}' | bgzip -c > {output}
        """

rule model_interpretation_ism:
    input:
        model=lambda wc: getModelPath()["model"],
        weights=lambda wc: getModelPath()["weights"],
        sequences=lambda wc: getTestSequences(wc.test_sequence_type),
        script=getScript("ism.py"),
    output:
        scores="results/model_interpretation/{test_sequence_type}/scores.{test_fold}.{validation_fold}.h5",
    params:
        sequence_length=config["prediction"]["input_size"],
        mutation_length=config["satmut"]["mutation_length"],
        mutation_start=config["satmut"]["mutation_start"],
    log:
        "logs/model_interpretation/ism.{test_sequence_type}.{test_fold}.{validation_fold}.log",
    conda:
        "../envs/tensorflow.yml"
    shell:
        """
        python {input.script} \
        --sequence {input.sequences} --sequence-length {params.sequence_length} --mutation-length {params.mutation_length} --mutation-start {params.mutation_start} \
        --model {input.model} --weights {input.weights} \
        --scores-output {output.scores} &> {log}
        """


## tf-modisco-ism
rule model_interpretation_ism_concat:
    conda:
        "../envs/tfmodisco.yml"
    input:
        scores=lambda wc: expand(
            "results/model_interpretation/{{test_sequence_type}}/scores.{{test_fold}}.{validation_fold}.h5",
            validation_fold=getValidationFoldsForTest(
                wc.test_fold, config["training"]["folds"]
            ),
        ),
        script=getScript("ism_concat.py"),
    output:
        hdf5="results/model_interpretation/{test_sequence_type}/scores_concat.{test_fold}.h5",
    log:
        "logs/model_interpretation/ism_concat.{test_sequence_type}.{test_fold}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --output {output.hdf5}  &> {log}
        """


rule model_interpretation_tfmodisco_ism:
    conda:
        "../envs/tfmodisco.yml"
    input:
        scores=expand(
            "results/model_interpretation/{{test_sequence_type}}/scores_concat.{test_fold}.h5",
            test_fold=list(set(getTestFolds(config["training"]["folds"]))),
        ),
        script=getScript("tfmodisco_single_task.py"),
    output:
        hdf5="results/model_interpretation/{test_sequence_type}/tfmodisco/tfmodisco.hdf5",
    log:
        "logs/model_interpretation/tfmodisco_ism.{test_sequence_type}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --output {output.hdf5}  &> {log}
        """


rule model_interpretation_tfmodisco_lite_ism:
    conda:
        "../envs/tfmodisco-lite.yml"
    input:
        scores=expand(
            "results/model_interpretation/{{test_sequence_type}}/scores_concat.{test_fold}.h5",
            test_fold=list(set(getTestFolds(config["training"]["folds"]))),
        ),
        script=getScript("tfmodisco-lite.py"),
    output:
        hdf5="results/model_interpretation/{test_sequence_type}/tfmodisco-lite/tfmodisco.{target_id}.h5",
    params:
        target_id=lambda wc: wc.target_id,
    log:
        "logs/model_interpretation/tfmodisco_lite_ism.{test_sequence_type}.{target_id}.log",
    shell:
        """
        scores=`for i in {input.scores}; do echo "--scores $i"; done`;
        python {input.script} \
        `echo $scores` \
        --target-id {params.target_id} \
        --output {output.hdf5} --config default  &> {log}
        """


rule model_interpretation_tfmodisco_lite_report_ism:
    conda:
        "../envs/tfmodisco-lite.yml"
    input:
        seqlets="results/model_interpretation/{test_sequence_type}/tfmodisco-lite/tfmodisco.{target_id}.h5",
        motifs="results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme_nice.txt",
    output:
        report="results/model_interpretation/{test_sequence_type}/tfmodisco-lite/report.{target_id}/motifs.html",
        out_dir=directory(
            "results/model_interpretation/{test_sequence_type}/tfmodisco-lite/report.{target_id}"
        ),
    log:
        "logs/model_interpretation/tfmodisco_lite_report_ism.{test_sequence_type}.{target_id}.log",
    shell:
        """
        modisco report -i {input.seqlets} -o {output.out_dir} -m {input.motifs} &> {log}
        """


rule model_interpretation_convertToMeme:
    input:
        "results/model_interpretation/motifClustering/figures.{choice_intgrad_deeplift}.{test_fold}.{validation_fold}/motifs.txt",
    output:
        "results/model_interpretation/motifClustering/figures.{choice_intgrad_deeplift}.{test_fold}.{validation_fold}/motifs.meme",
    wrapper:
        getWrapper("meme/uniprobe2meme")


rule model_interpretation_download_JASPAR2022_CORE_vertebrates_non_redundant:
    output:
        "results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    shell:
        """
        curl https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt > {output}
        """


rule model_interpretation_convert_nicer_JASPAR2022_CORE_vertebrates_non_redundant:
    input:
        "results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    output:
        "results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme_nice.txt",
    shell:
        """
        cat {input} |
        awk '{{if ($1=="MOTIF") {{print $1,$2"_"$3,$3}} else {{print $0}}}}' > {output}
        """
