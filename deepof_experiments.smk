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
warmup_mode = ["linear"]
automatic_changepoints = ["rbf"]
animal_to_preprocess = ["B"]
losses = ["VQVAE", "SELBO", "SIWAE"]
n_cluster_loss = [1.0]
gram_loss = [1.0]
encodings = [6]
cluster_numbers = [6, 8, 10, 12, 14, 16]
latent_reg = ["categorical+variance"]
entropy_knn = [10]
next_sequence_pred_weights = [0.0]
phenotype_pred_weights = [0.0]
supervised_pred_weights = [0.0]
input_types = ["coords"]
run = [1]


rule deepof_experiments:
    input:
        # Train a variety of models
        expand(
            outpath
            + "deepof_unsupervised_{latent_loss}_encodings_input={input_type}_k={k}_latdim={latdim}_latreg={latreg}_gram_loss={gram_loss}_n_cluster_loss={n_cluster}_run={run}.pkl",
            latent_loss=losses,
            input_type=input_types,
            k=cluster_numbers,
            latdim=encodings,
            latreg=latent_reg,
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
        + "deepof_unsupervised_{latent_loss}_encodings_input={input_type}_k={k}_latdim={latdim}_latreg={latreg}_gram_loss={gram_loss}_n_cluster_loss={n_cluster}_run={run}.pkl",
    shell:
        "pipenv run python -m deepof.deepof_train_unsupervised "
        "--train-path {input.data_path} "
        "--val-num 5 "
        "--animal-id B,W "
        "--animal-to-preprocess B "
        "--exclude-bodyparts Tail_1,Tail_2,Tail_tip "
        "--n-components {wildcards.k} "
        "--input-type {wildcards.input_type} "
        "--next-sequence-prediction 0.0 "
        "--phenotype-prediction 0.0 "
        "--supervised-prediction 0.0 "
        "--latent-reg {wildcards.latreg} "
        "--latent-loss {wildcards.latent_loss} "
        "--n-cluster-loss {wildcards.n_cluster} "
        "--gram-loss {wildcards.gram_loss} "
        "--kl-annealing-mode sigmoid "
        "--kl-warmup 15 "
        "--mmd-annealing-mode sigmoid "
        "--mmd-warmup 15 "
        "--montecarlo-kl 10 "
        "--encoding-size {wildcards.latdim} "
        "--entropy-knn 10 "
        "--batch-size 64 "
        "--window-size 5 "
        "--window-step 1 "
        "--run {wildcards.run} "
        "--output-path {outpath}train_models"
