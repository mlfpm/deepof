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

automatic_changepoints = ["rbf"]
animal_to_preprocess = ["B"]
kmeans_loss = [1.0]
encodings = [16]
cluster_numbers = [12] #list(range(5, 26, 1))
input_types = ["coords"]
run = list(range(5))
embedding_model = ["VQVAE", "GMVAE"]
encoder_model = ["recurrent", "TCN", "transformer"]

rule deepof_experiments:
    input:
        # Train a variety of models
        expand(
            outpath
            + "train_models/deepof_unsupervised_{embedding_model}_encodings_input={input_type}_k={k}_latdim={latdim}_kmeans_loss={kmeans_loss}_run={run}.pkl",
            embedding_model=embedding_model,
            input_type=input_types,
            k=cluster_numbers,
            latdim=encodings,
            kmeans_loss=kmeans_loss,
            run=run,
        ),


rule train_models:
    input:
        data_path=ancient(
            "/u/lucasmir/Projects/DLC/DeepOF/Projects/DeepOF_Stress_paper/Tagged_videos/Data_for_deepof_SI/JB08_files_SI",
        ),
    output:
        trained_models=outpath
        + "train_models/deepof_unsupervised_{embedding_model}_encodings_input={input_type}_k={k}_latdim={latdim}_kmeans_loss={kmeans_loss}_run={run}.pkl",
    shell:
        "pipenv run python -m deepof.deepof_train_unsupervised "
        "--train-path {input.data_path} "
        "--embedding-model {wildcards.embedding_model}"
        "--val-num 15 "
        "--animal-id B,W "
        "--animal-to-preprocess B "
        "--exclude-bodyparts Tail_1,Tail_2,Tail_tip "
        "--n-components {wildcards.k} "
        "--input-type {wildcards.input_type} "
        "--kmeans-loss {wildcards.kmeans_loss} "
        "--encoding-size {wildcards.latdim} "
        "--batch-size 256 "
        "--window-size 5 "
        "--window-step 1 "
        "--run {wildcards.run} "
        "--output-path {outpath}train_models"
