
rule sequence_selection_overlap_regions:
    input:
        regions=getRegionFiles(),
        genome=config["reference"]["genome"],
    output:
        region="results/sequence_selection/regions.overlap.positives.bed.gz",
    params:
        merge=lambda wc: config["input"]["merge"],
        length=lambda wc: config["input"]["length"],
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
    input:
        region=lambda wc: getRegionFileFromSample(wc.sample),
        genome=config["reference"]["genome"],
    output:
        region="results/sequence_selection/regions.overlap_sample.{sample}.bed.gz",
    params:
        length=lambda wc: config["input"]["length"],
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
        length="{length}",
        sliding="{sliding}",
    wrapper:
        getWrapper("negative_training_sampler/create_windows_over_genome")


rule sequence_selection_negative_background_input:
    input:
        positives="results/sequence_selection/regions.overlap.positives.bed.gz",
        negatives=lambda wc: expand(
            "results/sequence_selection/negatives.windows.{length}l.{sliding}s.bed.gz",
            length=lambda wc: config["input"]["length"],
            sliding=lambda wc: config["input"]["sliding"],
        ),
    output:
        "results/sequence_selection/negatives.regions.input.bed.gz",
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
    wrapper:
        getWrapper("negative_training_sampler/0.3.0")


rule sequence_selection_negative_background_final:
    input:
        "results/sequence_selection/negatives.regions.selected.bed.gz",
    output:
        "results/sequence_selection/negatives.regions.final.bed.gz",
    shell:
        """
        zcat {input} | \
        awk -v 'OFS=\\t' '{{if ($4==0){{print $1,$2,$3,".",".","+"}}}}' | \
        bgzip -c > {output}
        """


# create regions file


rule sequence_selection_create_regions_file:
    input:
        "results/sequence_selection/negatives.regions.final.bed.gz",
        "results/sequence_selection/regions.overlap.positives.bed.gz",
    output:
        "results/sequence_selection/regions.overlap.all.bed.gz",
    shell:
        """
        zcat {input} | sort -k1,1 -k2,2n | uniq | bgzip -c > {output}
        """


rule sequence_selection_annotate_regions:
    input:
        regions="results/sequence_selection/regions.overlap.all.bed.gz",
        samples=lambda wc: expand(
            "results/sequence_selection/regions.overlap_sample.{sample}.bed.gz",
            sample=list(samples.keys()),
        ),
        negative="results/sequence_selection/negatives.regions.final.bed.gz",
        genome=config["genome"],
    output:
        regions="results/sequence_selection/regions.annotated.all.bed.gz",
    params:
        names=lambda wc: "\t".join(list(samples.keys()) + ["negatives"]),
    shell:
        """
        bedtools annotate -i {input.regions} -files {input.samples} {input.negative} -names "{params.names}" | \
        sort -k1,1 -k2,2n | uniq | bgzip -c > {output.regions}
        """


# train test validate regions


rule sequence_selection_training_regions:
    input:
        "results/sequence_selection/regions.annotated.all.bed.gz",
    output:
        "results/sequence_selection/regions.annotated.training.bed.gz",
    shell:
        "zcat {input} | egrep -v '^chr8' | egrep -v '^chr18' | egrep -v '^chrM' | bgzip -c > {output}"


rule sequence_selection_validation_regions:
    input:
        "results/sequence_selection/regions.annotated.all.bed.gz",
    output:
        "results/sequence_selection/regions.annotated.validation.bed.gz",
    shell:
        "zcat {input} | egrep '^chr18' | bgzip -c > {output}"


rule sequence_selection_test_regions:
    input:
        "results/sequence_selection/regions.annotated.all.bed.gz",
    output:
        "results/sequence_selection/regions.annotated.test.bed.gz",
    shell:
        "zcat {input} | egrep '^chr8' | bgzip -c > {output}"


rule sequence_selection_extract_fasta:
    input:
        regions="results/sequence_selection/regions.annotated.{dataset}.bed.gz",
        reference=config["reference"]["fasta"],
    output:
        sequences="results/sequence_selection/sequence.annotated.{dataset}.fa.gz",
    shell:
        """
        bedtools getfasta -s -fi {input.reference} -bed {input.regions} | bgzip -c > {output.sequences}
        """
