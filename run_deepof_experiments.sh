#!/bin/sh

mkdir snakemake_logs

module load anaconda/3/2021.05
module load cuda/11.2
module load nccl/2.11.4
module load cudnn/8.2.4
module load tensorrt/8.2.0
module load gcc/10

source activate deepof

snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

pipenv run snakemake --snakefile deepof_experiments.smk deepof_experiments -j 100 --latency-wait 15 --cluster-config cluster.json --cluster "sbatch --time={cluster.time} --mem={cluster.mem} -o {cluster.stdout} -e {cluster.stderr} --job-name={cluster.jobname} --cpus-per-task={cluster.cpus} --constraint="gpu" --gres=gpu:gtx980:1" > deepof_experiments.out 2> deepof_experiments.err
