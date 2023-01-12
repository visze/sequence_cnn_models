rule regression_input_combine_labels:
    """
    Combine labels file including folds and regression data with sequences fromfasta file
    """
    input:
        fasta=config["input"]["fasta"],
        labels=config["input"]["labels"],
    output:
        "results/regression_input/regression.tsv.gz",
    log:
        "logs/regression_input/combine_labels.log",
    shell:
        """
        join -t $'\\t' \
        <(cat {input.labels} | sed 's/\\r//' | awk -v "OFS=\\t" '{{printf "%s\\t", $2; for(i=3; i<=NF; i++) {{ printf "%s\\t", $i}}; printf "%d\\n", $1}}' | sort -k1,1) \
        <(
            cat {input.fasta} | \
            awk -v "OFS=\\t" '/^>/ {{printf("%s\\t",substr($1,2));next; }} {{printf("%s\\n",$1);}}' | \
            sed 's/_F\\t/\\t/g' | \
            sort -k1,1;
        ) | gzip -c > {output} 2> {log}
        """


rule regression_input_get_training:
    """
    Get Training data from regression data defined by validation/test bin
    """
    input:
        "results/regression_input/regression.tsv.gz",
    output:
        "results/regression_input/regression.training.{test_fold}.{validation_fold}.tsv.gz",
    params:
        test_fold=lambda wc: wc.test_fold,
        validation_fold=lambda wc: wc.validation_fold,
    log:
        "logs/regression_input/get_training.{test_fold}.{validation_fold}.log",
    shell:
        """
        zcat {input} | \
        awk -v "OFS=\\t" '{{
            if ($(NF-1) != {params.test_fold} && $(NF-1) != {params.validation_fold}) {{
                printf "%s\\t%s",$1,$NF;
                for(i=2; i<(NF-1); i++) {{ printf "\\t%s", $i}};
                printf "\\n"
            }}
        }}' | \
        sed 's/\\r//' | \
        gzip -c > {output} 2> {log}
        """


rule regression_input_get_validation:
    """
    Get Validation data from regression data defined by validation/test bin
    """
    input:
        "results/regression_input/regression.tsv.gz",
    output:
        "results/regression_input/regression.validation.{test_fold}.{validation_fold}.tsv.gz",
    params:
        test_fold=lambda wc: wc.test_fold,
        validation_fold=lambda wc: wc.validation_fold,
    log:
        "logs/regression_input/get_validation.{test_fold}.{validation_fold}.log",
    shell:
        """
        zcat {input} | \
        awk -v "OFS=\\t" '{{
            if ($(NF-1) == {params.validation_fold}) {{
                printf "%s\\t%s",$1,$NF;
                for(i=2; i<(NF-1); i++) {{ printf "\\t%s", $i}};
                printf "\\n"
            }}
        }}' | \
        sed 's/\\r//' | \
        gzip -c > {output} 2> {log}
        """


rule regression_input_get_test:
    """
    Get Test data from regression data defined by validation/test bin
    """
    input:
        "results/regression_input/regression.tsv.gz",
    output:
        "results/regression_input/regression.test.{test_fold}.{validation_fold}.tsv.gz",
    params:
        test_fold=lambda wc: wc.test_fold,
        validation_fold=lambda wc: wc.validation_fold,
    log:
        "logs/regression_input/get_test.{test_fold}.{validation_fold}.log",
    shell:
        """
        zcat {input} | \
        awk -v "OFS=\\t" '{{
            if ($(NF-1) == {params.test_fold}) {{
                printf "%s\\t%s",$1,$NF;
                for(i=2; i<(NF-1); i++) {{ printf "\\t%s", $i}};
                printf "\\n"
            }}
        }}' | \
        sed 's/\\r//' | \
        gzip -c > {output} 2> {log}
        """
