#!/bin/sh

mkdir snakemake_logs

snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

module load anaconda/3
module load intel/18.0.5
module load cuda/10.1
module load cudnn/7.6.5
module load nccl/2.6.4
module load tensorrt/6.0.1

pipenv run snakemake --snakefile deepof_experiments.smk deepof_experiments -j 15 --latency-wait 15 --cluster-config cluster.json --cluster "sbatch --time={cluster.time} --mem={cluster.mem} -o {cluster.stdout} -e {cluster.stderr} --job-name={cluster.jobname} --cpus-per-task={cluster.cpus} --constraint="gpu" --gres=gpu:gtx980:1" > deepof_experiments.out 2> deepof_experiments.err
