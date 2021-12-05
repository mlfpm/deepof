#!/bin/sh

mkdir snakemake_logs

module load cuda/11.2
module load tensorflow/gpu-cuda-11.2/2.6.0
module load tensorflow-probability/0.14.1

snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

pipenv run snakemake --snakefile deepof_experiments.smk deepof_experiments -j 25 --latency-wait 15 --cluster-config cluster.json --cluster "sbatch --time={cluster.time} --mem={cluster.mem} -o {cluster.stdout} -e {cluster.stderr} --job-name={cluster.jobname} --cpus-per-task={cluster.cpus} --constraint="gpu" --gres=gpu:gtx980:2" > deepof_experiments.out 2> deepof_experiments.err
