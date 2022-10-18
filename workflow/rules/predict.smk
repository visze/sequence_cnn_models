rule predict_fromFasta:
    input:
        test_file=lambda wc: tests.loc[wc.test_name]["file"],
        model="results/training/model.json",
        weights="results/training/weights.h5",
        script=getScript("predict.py"),
    output:
        "results/predictions/{test_name}.tsv.gz",
    log:
        "logs/predict/fromFasta.{test_name}.log",
    params:
        length=lambda wc: config["input"]["length"],
    conda:
        "../envs/tensorflow.yml"
    threads: 1
    shell:
        """
        python {input.script} \
        --test {input.test_file} \
        --model {input.model} --weights {input.weights} \
        --sequence-length {params.length} \
        --output {output} > {log}
        """
