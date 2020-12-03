mkdir snakemake_logs

snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

source /mpcdf/soft/distribution/obs_modules.sh
module load anaconda/3
module load intel/18.0.5
module load cuda/10.2
module load cudnn/8.0.4
module load nccl/2.7.8
module load tensorrt/7.1.3
module load tensorflow/gpu/2.3.0
module load tensorboard/2.3.0

pipenv run snakemake --snakefile deepof_experiments.smk deepof_experiments -j 8 --latency-wait 15 --cluster-config cluster.json --cluster "sbatch --time={cluster.time} --mem={cluster.mem} --exclude={cluster.excl} -o {cluster.stdout} -e {cluster.stderr} --job-name={cluster.jobname} --cpus-per-task={cluster.cpus} --constraint="gpu" --gres=gpu:gtx980:1" > deepof_experiments.out 2> deepof_experiments.err
