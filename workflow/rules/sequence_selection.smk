
rule sequence_selection_overlap_regions:
    conda:
        "../envs/default.yml"
    input:
        regions=getRegionFiles(),
        genome=config["reference"]["genome"],
    output:
        region="results/sequence_selection/regions.overlap.positives.bed.gz",
    params:
        merge=lambda wc: config["input"]["merge"],
        length=lambda wc: config["input"]["length"],
    log:
        "logs/sequence_selection/overlap_regions.log",
    shell:
        """
        bedtools merge -d {params.merge} -i <(
            zcat {input.regions} |  egrep "^chr([0-9]+|[XYM])\\s" | sort -k1,1 -k2,2n
        ) | \
        awk -v "OFS=\\t" '{{n=int(($3-$2)/2);$2=$2+n; $3=$3-n; print $1,$2,$3,".",".","+"}}' | \
        awk -v "OFS=\\t" '{{if ($2==$3) $3=$3+1; print $0}}' | \
        bedtools slop -g {input.genome} -s -l `echo "({params.length} / 2)" | \
        bc` -r `echo "(({params.length} / 2) - 1)" | bc`  | \
        sort -k1,1 -k2,2n | uniq | bgzip -c > {output.region}
        """


rule sequence_selection_annotate_overlap:
    conda:
        "../envs/default.yml"
    input:
        region=lambda wc: getRegionFileFromSample(wc.sample),
        genome=config["reference"]["genome"],
    output:
        region="results/sequence_selection/regions.overlap_sample.{sample}.bed.gz",
    params:
        length=lambda wc: config["input"]["length"],
    log:
        "logs/sequence_selection/annotate_overlap.{sample}.log",
    shell:
        """
        zcat {input.region} | egrep "^chr([0-9]+|[XYM])\\s" | \
        awk -v "OFS=\\t" '{{n=int(($3-$2)/2);$2=$2+n; $3=$3-n; print $1,$2,$3,".",".","+"}}' | \
        awk -v "OFS=\\t" '{{if ($2==$3) $3=$3+1; print $0}}' | \
        bedtools slop -g {input.genome} -s -l `echo "({params.length} / 2)" | \
        bc` -r `echo "(({params.length} / 2) - 1)" | bc`  | \
        sort -k1,1 -k2,2n | uniq | bgzip -c > {output.region}
        """


# negative background data


rule sequence_selection_negative_background_widnows:
    input:
        genome=config["reference"]["genome"],
    output:
        "results/sequence_selection/negatives.windows.{length}l.{sliding}s.bed.gz",
    params:
        length=lambda wc: wc.length,
        sliding=lambda wc: wc.sliding,
    log:
        "logs/sequence_selection/negative_background_widnows.{length}.{sliding}.log",
    wrapper:
        getWrapper("negative_training_sampler/create_windows_over_genome")


rule sequence_selection_negative_background_input:
    input:
        positives="results/sequence_selection/regions.overlap.positives.bed.gz",
        negatives=lambda wc: expand(
            "results/sequence_selection/negatives.windows.{length}l.{sliding}s.bed.gz",
            length=config["input"]["length"],
            sliding=config["input"]["sliding"],
        ),
    output:
        "results/sequence_selection/negatives.regions.input.bed.gz",
    log:
        "logs/sequence_selection/negative_background_input.log",
    wrapper:
        getWrapper("negative_training_sampler/create_input")


import random


rule sequence_selection_negative_background_sampler:
    input:
        label="results/sequence_selection/negatives.regions.input.bed.gz",
        reference=config["reference"]["fasta"],
        genome_file=config["reference"]["genome"],
    output:
        "results/sequence_selection/negatives.regions.selected.bed.gz",
    threads: 5
    params:
        seed=lambda wc: random.Random(config["seed"]).randint(0, 100000),
        memory="10GB",
    log:
        "results/logs/sequence_selection/negative_background_sampler.log",
    wrapper:
        getWrapper("negative_training_sampler/0.3.0")


rule sequence_selection_negative_background_final:
    conda:
        "../envs/default.yml"
    input:
        "results/sequence_selection/negatives.regions.selected.bed.gz",
    output:
        "results/sequence_selection/negatives.regions.final.bed.gz",
    log:
        "logs/sequence_selection/negative_background_final.log",
    shell:
        """
        zcat {input} | \
        awk -v 'OFS=\\t' '{{if ($4==0){{print $1,$2,$3,".",".","+"}}}}' | \
        bgzip -c > {output}
        """


# create regions file


rule sequence_selection_create_regions_file:
    conda:
        "../envs/default.yml"
    input:
        "results/sequence_selection/negatives.regions.final.bed.gz",
        "results/sequence_selection/regions.overlap.positives.bed.gz",
    output:
        "results/sequence_selection/regions.overlap.all.bed.gz",
    log:
        "logs/sequence_selection/create_regions_file.log",
    shell:
        """
        zcat {input} | sort -k1,1 -k2,2n | uniq | bgzip -c > {output}
        """


rule sequence_selection_annotate_regions:
    conda:
        "../envs/default.yml"
    input:
        regions="results/sequence_selection/regions.overlap.all.bed.gz",
        samples=lambda wc: expand(
            "results/sequence_selection/regions.overlap_sample.{sample}.bed.gz",
            sample=list(samples.index),
        ),
        negative="results/sequence_selection/negatives.regions.final.bed.gz",
        genome=config["reference"]["genome"],
    output:
        regions="results/sequence_selection/regions.annotated.all.bed.gz",
    params:
        names=lambda wc: "\t".join(list(samples.index) + ["negatives"]),
    log:
        "logs/sequence_selection/annotate_regions.log",
    shell:
        """
        bedtools annotate -i {input.regions} -files {input.samples} {input.negative} -names "{params.names}" | \
        sort -k1,1 -k2,2n | uniq | bgzip -c > {output.regions}
        """


# train test validate regions


rule sequence_selection_training_regions:
    conda:
        "../envs/default.yml"
    input:
        "results/sequence_selection/regions.annotated.all.bed.gz",
    output:
        "results/sequence_selection/regions.annotated.training.bed.gz",
    params:
        positiveLabelThreshold=0.8,
    log:
        "logs/sequence_selection/training_regions.log",
    shell:
        """
        zcat {input} | egrep -v '^chr8' | egrep -v '^chr18' | egrep -v '^chrM' | \
        awk -v "OFS=\\t" '{{for(i=7; i<=NF; i++) {{if($i>={params.positiveLabelThreshold}){{ $i=1 }} else {{$i=0}}}}; print $0}}' | \
        bgzip -c > {output}
        """


rule sequence_selection_validation_regions:
    conda:
        "../envs/default.yml"
    input:
        "results/sequence_selection/regions.annotated.all.bed.gz",
    output:
        "results/sequence_selection/regions.annotated.validation.bed.gz",
    params:
        positiveLabelThreshold=0.8,
    log:
        "logs/sequence_selection/validation_regions.log",
    shell:
        """
        zcat {input} | egrep '^chr18' | \
        awk -v "OFS=\\t" '{{for(i=7; i<=NF; i++) {{if($i>={params.positiveLabelThreshold}){{ $i=1 }} else {{$i=0}}}}; print $0}}' | \
        bgzip -c > {output}
        """


rule sequence_selection_test_regions:
    conda:
        "../envs/default.yml"
    input:
        "results/sequence_selection/regions.annotated.all.bed.gz",
    output:
        "results/sequence_selection/regions.annotated.test.bed.gz",
    params:
        positiveLabelThreshold=0.8,
    log:
        "logs/sequence_selection/test_regions.log",
    shell:
        """
        zcat {input} | egrep '^chr8' | \
        awk -v "OFS=\\t" '{{for(i=7; i<=NF; i++) {{if($i>={params.positiveLabelThreshold}){{ $i=1 }} else {{$i=0}}}}; print $0}}' | \
        bgzip -c > {output}
        """


# bidirectional


rule sequence_selection_bidirectional:
    conda: "../envs/default.yml"
    input:
        "results/sequence_selection/regions.annotated.{dataset}.bed.gz",
    output:
        "results/sequence_selection/regions.annotated_bidirectional.{dataset}.bed.gz",
    log:
        "logs/sequence_selection/bidirectional..{dataset}.log",
    shell:
        """
        (
            zcat {input} | awk -v "OFS=\\t" '{{$6="+";print$0}}'; \
            zcat {input} | awk -v "OFS=\\t" '{{$6="-";print$0}}';
        ) | sort -k1,1 -k2,2n | bgzip -c > {output}
        """


# extract fasta


rule sequence_selection_extract_fasta:
    conda:
        "../envs/default.yml"
    input:
        regions="results/sequence_selection/regions.annotated_bidirectional.{dataset}.bed.gz",
        reference=config["reference"]["fasta"],
    output:
        sequences="results/sequence_selection/sequence.annotated_bidirectional.{dataset}.fa.gz",
    log:
        "logs/sequence_selection/extract_fasta.{dataset}.log",
    shell:
        """
        bedtools getfasta -s -fi {input.reference} -bed {input.regions} | bgzip -c > {output.sequences}
        """
