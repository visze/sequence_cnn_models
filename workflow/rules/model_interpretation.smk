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


########################
#### use tf modisco ####
########################


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


rule model_interpretation_prepareNegativeWindows:
    input:
        genome_file=config["reference"]["genome"],
    output:
        "results/model_interpretation/input/background/regression.potentialRegions.bed.gz",
    params:
        length=230,
        sliding=50,
    wrapper:
        getWrapper("negative_training_sampler/create_windows_over_genome")


rule model_interpretation_runBackgroundSampler:
    conda:
        "../envs/negative_training_sampler.yml"
    input:
        label="results/model_interpretation/input/background/regression.potentialRegions.bed.gz",
        reference=config["reference"]["fasta"],
        genome_file=config["reference"]["genome"],
        positive_fasta="results/model_interpretation/input/regression.test.{test_fold}.{validation_fold}.fa.gz",
    output:
        "results/model_interpretation/input/background/regression.regions.{test_fold}.{validation_fold}.bed.gz",
    log:
        out="logs/model_interpretation/runBackgroundSample.{test_fold}.{validation_fold}.out.log",
        err="logs/model_interpretation/runBackgroundSample.{test_fold}.{validation_fold}.err.log",
    shell:
        """
        PYTHONPATH=/data/gpfs-1/users/schubacm_c/work/projects/software/negative_training_sampler python -m negative_training_sampler \
        -i {input.label} -f {input.positive_fasta} \
        -r {input.reference} -g {input.genome_file}  \
        --cores 4 --memory 19GB \
        --log {log.out} \
        -o {output} &> {log.err}
        """


rule model_interpretation_getBackgroundData:
    conda:
        "../envs/default.yml"
    input:
        "results/model_interpretation/input/background/regression.regions.{test_fold}.{validation_fold}.bed.gz",
    output:
        "results/model_interpretation/input/background/regression.background.{test_fold}.{validation_fold}.bed.gz",
    shell:
        """
        zcat {input} | awk -v 'OFS=\\t' '{{if ($4==0){{print $0,"."}}}}' | bgzip -c > {output}
        """


rule model_interpretation_deepexplain:
    input:
        model="results/training/model.regression.{test_fold}.{validation_fold}.json",
        weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
        reference=config["reference"]["fasta"],
        sequences="results/model_interpretation/input/regression.test.{test_fold}.{validation_fold}.fa.gz",
        background="results/model_interpretation/input/background/regression.background.{test_fold}.{validation_fold}.bed.gz",
        script=getScript("deepexplain.py"),
    output:
        intgrad="results/model_interpretation/deepexplain/intgrad.{test_fold}.{validation_fold}.tsv.gz",
        deeplift="results/model_interpretation/deepexplain/deeplift.{test_fold}.{validation_fold}.tsv.gz",
    params:
        inputLayer="input",
        task=0,
    log:
        out="logs/model_interpretation/deepexplain.{test_fold}.{validation_fold}.out.log",
        err="logs/model_interpretation/deepexplain.{test_fold}.{validation_fold}.err.log",
    conda:
        "../envs/deepexplain_gpu.yml"
    shell:
        """
        python {input.script} \
        --reference {input.reference} \
        --sequence {input.sequences} --background {input.background} \
        --model {input.model} --weights {input.weights} \
        --input-layer input \
        --log {log.out} \
        --output-intgrad {output.intgrad} --output-deeplift {output.deeplift} &> {log.err}
        """


rule model_interpretation_deeplift:
    input:
        model="results/training/model.regression.{test_fold}.{validation_fold}.json",
        weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
        reference=config["reference"]["fasta"],
        sequences="results/model_interpretation/input/regression.test.{test_fold}.{validation_fold}.fa.gz",
        background="results/model_interpretation/input/background/regression.background.{test_fold}.{validation_fold}.bed.gz",
        script=getScript("deeplift.py"),
    output:
        rescale_conv_revealcancel_fc="results/model_interpretation/deeplift/rescale_conv_revealcancel_fc.{test_fold}.{validation_fold}.tsv.gz",
        grad_times_inp="results/model_interpretation/deeplift/grad_times_inp.{test_fold}.{validation_fold}.tsv.gz",
        guided_backprop="results/model_interpretation/deeplift/guided_backprop.{test_fold}.{validation_fold}.tsv.gz",
        integrated_gradients10="results/model_interpretation/deeplift/integrated_gradients10.{test_fold}.{validation_fold}.tsv.gz",
        rescale_all_layers="results/model_interpretation/deeplift/rescale_all_layers.{test_fold}.{validation_fold}.tsv.gz",
        revealcancel_all_layers="results/model_interpretation/deeplift/revealcancel_all_layers.{test_fold}.{validation_fold}.tsv.gz",
        rescale_conv_revealcancel_fc_multiref_10="results/model_interpretation/deeplift/rescale_conv_revealcancel_fc_multiref_10.{test_fold}.{validation_fold}.tsv.gz",
    params:
        inputLayer="input",
        task=0,
    log:
        out="logs/model_interpretation/deeplift.{test_fold}.{validation_fold}.out.log",
        err="logs/model_interpretation/deeplift.{test_fold}.{validation_fold}.err.log",
    conda:
        "../envs/deeplift_gpu.yml"
    shell:
        """
        python {input.script} \
        --reference {input.reference} \
        --sequence {input.sequences} --background {input.background} \
        --model {input.model} --weights {input.weights}
        --input-layer input \
        --log {log.out} \
        --output rescale_conv_revealcancel_fc {output.rescale_conv_revealcancel_fc} \
        --output grad_times_inp {output.grad_times_inp} \
        --output guided_backprop {output.guided_backprop} \
        --output integrated_gradients10 {output.integrated_gradients10} \
        --output rescale_all_layers {output.rescale_all_layers} \
        --output revealcancel_all_layers {output.revealcancel_all_layers} \
        --output rescale_conv_revealcancel_fc_multiref_10 {output.rescale_conv_revealcancel_fc_multiref_10} \
         &> {log.err}
        """


### ISM
rule model_interpretation_ism:
    input:
        model="results/training/model.regression.{test_fold}.{validation_fold}.json",
        weights="results/training/weights.regression.{test_fold}.{validation_fold}.h5",
        sequences="results/model_interpretation/input/regression.test.{test_fold}.{validation_fold}.fa.gz",
        script=getScript("ism.py"),
    output:
        scores="results/model_interpretation/ism/scores.{test_fold}.{validation_fold}.h5",
    params:
        sequence_length=230,
        mutation_length=200,
        mutation_start=16,
    log:
        "logs/model_interpretation/ism.{test_fold}.{validation_fold}.log",
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
            "results/model_interpretation/ism/scores.{{test_fold}}.{validation_fold}.h5",
            validation_fold=list(range(1, 11))[: int(wc.test_fold) - 1]
            + list(range(1, 11))[int(wc.test_fold) :],
        ),
        script=getScript("ism_concat.py"),
    output:
        hdf5="results/model_interpretation/ism/scores_concat.{test_fold}.h5",
    log:
        "logs/model_interpretation/ism_concat.{test_fold}.log",
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
            "results/model_interpretation/ism/scores_concat.{test_fold}.h5",
            test_fold=list(range(1, 11)),
        ),
        script=getScript("tfmodisco_single_task.py"),
    output:
        hdf5="results/model_interpretation/ism/tfmodisco/tfmodisco.hdf5",
    log:
        "logs/model_interpretation/tfmodisco_ism.log",
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
            "results/model_interpretation/ism/scores_concat.{test_fold}.h5",
            test_fold=list(range(1, 11)),
        ),
        script=getScript("tfmodisco-lite.py"),
    output:
        hdf5="results/model_interpretation/ism/tfmodisco-lite/tfmodisco.{target_id}.h5",
    params:
        target_id=lambda wc: wc.target_id,
    log:
        "logs/model_interpretation/tfmodisco_lite_ism.{target_id}.log",
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
        seqlets="results/model_interpretation/ism/tfmodisco-lite/tfmodisco.{target_id}.h5",
        motifs="results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    output:
        report="results/model_interpretation/ism/tfmodisco-lite/report.{target_id}/motifs.html",
        out_dir=directory(
            "results/model_interpretation/ism/tfmodisco-lite/report.{target_id}"
        ),
    log:
        "logs/model_interpretation/tfmodisco_lite_report_ism.{target_id}.log",
    shell:
        """
        modisco report -i {input.seqlets} -o {output.out_dir} -m {input.motifs} &> {log}
        """


## tf-modisco1


rule model_interpretation_tfmodisco1:
    conda:
        "../envs/tfmodisco.yml"
    input:
        scores="results/model_interpretation/ism/scores.{test_fold}.{validation_fold}.h5",
        script=getScript("tf_modisco.py"),
    output:
        hdf5="results/model_interpretation/ism/tfmodisco/tfModisco.{test_fold}.{validation_fold}.hdf5",
        fig_dir=directory(
            "results/model_interpretation/ism/tfmodisco/figures.{test_fold}.{validation_fold}/"
        ),
    log:
        "logs/model_interpretation/tfmodisco1.{test_fold}.{validation_fold}.log",
    shell:
        """
        python {input.script} \
        --input {input.scores} \
        --output {output.hdf5} \
        --output-figures-dir {output.fig_dir} &> {log}
        """


rule model_interpretation_tfmodisco2:
    input:
        ism="results/model_interpretation/ism/scores.{test_fold}.{validation_fold}.h5",
        hdf5="results/model_interpretation/ism/tfmodisco/tfModisco.{test_fold}.{validation_fold}.hdf5",
        reference=config["reference"]["fasta"],
        background="results/model_interpretation/input/background/regression.background.{test_fold}.{validation_fold}.bed.gz",
        script=getScript("tf_modisco_cluster.py"),
    output:
        heatmap="results/model_interpretation/ism/tfmodisco/heatmap.{test_fold}.{validation_fold}.png",
        motifs="results/model_interpretation/ism/tfmodisco/motifs.{test_fold}.{validation_fold}.txt",
        fig_dir=directory(
            "results/model_interpretation/ism/tfmodisco/figures.{test_fold}.{validation_fold}/"
        ),
    log:
        "logs/model_interpretation/tfmodisco2.{test_fold}.{validation_fold}.log",
    shell:
        """
        python {input.script} \
        --ism {input.ism} \
        --input {input.hdf5} \
        --reference {input.reference} \
        --background {input.background} \
        --output-motifs {output.motifs} \
        --output-log {log} \
        --output-heatmap {output.heatmap} \
        --output-figures-dir {output.fig_dir} &> {log}
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


rule model_interpretation_findMotifs:
    input:
        "results/model_interpretation/motifClustering/figures.{choice_intgrad_deeplift}.{test_fold}.{validation_fold}/motifs.meme",
        "results/model_interpretation/motifDB/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    output:
        "results/model_interpretation/motifComparison/{choice_intgrad_deeplift}.{test_fold}.{validation_fold}.tsv",
    wrapper:
        getWrapper("meme/tomtom")
