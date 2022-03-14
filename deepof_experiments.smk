# @authors lucasmiranda42
# encoding: utf-8
# deepof_experiments

"""

Snakefile for data and imputation.
Execution: sbatch snakemake
Plot DAG: snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
Plot rule graph: snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

"""

outpath = "/u/lucasmir/Projects/DLC/DeepOF/deepof/"

warmup_epochs = [15]
automatic_changepoints = ["rbf"]
animal_to_preprocess = ["B"]
gram_loss = [1.0]
encodings = [6]
cluster_numbers = [6, 8, 10, 12, 14, 16]
input_types = ["coords"]
run = [1]


rule deepof_experiments:
    input:
        # Train a variety of models
        expand(
            outpath
            + "deepof_unsupervised_VQVAE_encodings_input={input_type}_k={k}_latdim={latdim}_gram_loss={gram_loss}_run={run}.pkl",
            input_type=input_types,
            k=cluster_numbers,
            latdim=encodings,
            n_cluster=n_cluster_loss,
            gram_loss=gram_loss,
            run=run,
        ),


rule train_models:
    input:
        data_path=ancient(
            "/u/lucasmir/Projects/DLC/DeepOF/Projects/DeepOF_Stress_paper/Tagged_videos/Data_for_deepof_SI/JB08_files_SI",
        ),
    output:
        trained_models=outpath
        + "deepof_unsupervised_VQVAE_encodings_input={input_type}_k={k}_latdim={latdim}_gram_loss={gram_loss}__run={run}.pkl",
    shell:
        "pipenv run python -m deepof.deepof_train_unsupervised "
        "--train-path {input.data_path} "
        "--val-num 5 "
        "--animal-id B,W "
        "--animal-to-preprocess B "
        "--exclude-bodyparts Tail_1,Tail_2,Tail_tip "
        "--n-components {wildcards.k} "
        "--input-type {wildcards.input_type} "
        "--gram-loss {wildcards.gram_loss} "
        "--encoding-size {wildcards.latdim} "
        "--batch-size 64 "
        "--window-size 5 "
        "--window-step 1 "
        "--run {wildcards.run} "
        "--output-path {outpath}train_models"
