# @authors lucasmiranda42
# encoding: utf-8
# deepof_experiments

"""

Snakefile for data and imputation.
Execution: sbatch snakemake
Plot DAG: snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
Plot rule graph: snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

"""

import os

outpath = "/psycl/g/mpsstatgen/lucas/DLC/DLC_autoencoders/DeepOF/deepof/logs/"
losses = ["ELBO"]  # , "MMD", "ELBO+MMD"]
encodings = [6]  # [2, 4, 6, 8, 10, 12, 14, 16]
cluster_numbers = [25]  # [1, 5, 10, 15, 20, 25]
latent_reg = ["none", "categorical", "variance", "categorical+variance"]
entropy_knn = [20, 50, 80, 100]
pheno_weights = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0]


rule deepof_experiments:
    input:
        "/psycl/g/mpsstatgen/lucas/DLC/DLC_autoencoders/DeepOF/deepof/supplementary_notebooks/regognise_elliptical_arena.ipynb",
        # expand(
        #     os.path.join(
        #         outpath,
        #         "coarse_hyperparameter_tuning/trained_weights/GMVAE_loss={loss}_k={k}_encoding={enc}_final_weights.h5",
        #     ),
        #     loss=losses,
        #     k=cluster_numbers,
        #     enc=encodings,
        # ),
        # expand(
        #     "/psycl/g/mpsstatgen/lucas/DLC/DLC_autoencoders/DeepOF/deepof/logs/latent_regularization_experiments/trained_weights/"
        #     "GMVAE_loss={loss}_encoding={encs}_k={k}_latreg={latreg}_entropyknn={entknn}_final_weights.h5",
        #     loss=losses,
        #     encs=encodings,
        #     k=cluster_numbers,
        #     latreg=latent_reg,
        #     entknn=entropy_knn,
        # ),
        # expand(
        #     "/psycl/g/mpsstatgen/lucas/DLC/DLC_autoencoders/DeepOF/deepof/logs/pheno_classification_experiments/trained_weights/"
        #     "GMVAE_loss={loss}_encoding={encs}_k={k}_pheno={phenos}_run_1_final_weights.h5",
        #     loss=losses,
        #     encs=encodings,
        #     k=cluster_numbers,
        #     phenos=pheno_weights,
        # ),


rule elliptical_arena_detector:
    input:
        to_exec="supplementary_notebooks/regognise_elliptical_arena_blank.ipynb",
    output:
        exec="supplementary_notebooks/regognise_elliptical_arena.ipynb",
    shell:
        "papermill {input.to_exec} "
        "-p vid_path './supplementary_notebooks' "
        "-p log_path ./logs' "
        "-p out_path './logs' "
        "{output.exec}"


rule coarse_hyperparameter_tuning:
    input:
        data_path="/psycl/g/mpsstatgen/lucas/DLC/DLC_models/deepof_single_topview/",
    output:
        trained_models=os.path.join(
            outpath,
            "coarse_hyperparameter_tuning/trained_weights/GMVAE_loss={loss}_k={k}_encoding={enc}_final_weights.h5",
        ),
    shell:
        "pipenv run python -m deepof.train_model "
        "--train-path {input.data_path} "
        "--val-num 25 "
        "--components {wildcards.k} "
        "--input-type coords "
        "--predictor 0 "
        "--phenotype-classifier 0 "
        "--variational True "
        "--loss {wildcards.loss} "
        "--kl-warmup 20 "
        "--mmd-warmup 0 "
        "--encoding-size {wildcards.enc} "
        "--batch-size 256 "
        "--window-size 24 "
        "--window-step 12 "
        "--output-path {outpath}coarse_hyperparameter_tuning "
        "--hyperparameter-tuning hyperband "
        "--hpt-trials 1"


rule latent_regularization_experiments:
    input:
        data_path=ancient(
            "/psycl/g/mpsstatgen/lucas/DLC/DLC_models/deepof_single_topview/"
        ),
    output:
        trained_models=os.path.join(
            outpath,
            "latent_regularization_experiments/trained_weights/GMVAE_loss={loss}_encoding={encs}_k={k}_latreg={latreg}_entropyknn={entknn}_final_weights.h5",
        ),
    shell:
        "pipenv run python -m deepof.train_model "
        "--train-path {input.data_path} "
        "--val-num 5 "
        "--components {wildcards.k} "
        "--input-type coords "
        "--predictor 0 "
        "--phenotype-classifier 0 "
        "--variational True "
        "--latent-reg {wildcards.latreg} "
        "--loss {wildcards.loss} "
        "--kl-warmup 20 "
        "--mmd-warmup 20 "
        "--montecarlo-kl 10 "
        "--encoding-size {wildcards.encs} "
        "--entropy-knn {wildcards.entknn} "
        "--batch-size 256 "
        "--window-size 24 "
        "--window-step 12 "

        "--output-path {outpath}latent_regularization_experiments"
        # "--exclude-bodyparts Tail_base,Tail_1,Tail_2,Tail_tip "


rule explore_phenotype_classification:
    input:
        data_path="/psycl/g/mpsstatgen/lucas/DLC/DLC_models/deepof_single_topview/",
    output:
        trained_models=os.path.join(
            outpath,
            "pheno_classification_experiments/trained_weights/GMVAE_loss={loss}_encoding={encs}_k={k}_pheno={phenos}_run_1_final_weights.h5",
        ),
    shell:
        "pipenv run python -m deepof.train_model "
        "--train-path {input.data_path} "
        "--val-num 15 "
        "--components {wildcards.k} "
        "--input-type coords "
        "--predictor 0 "
        "--phenotype-classifier {wildcards.phenos} "
        "--variational True "
        "--loss {wildcards.loss} "
        "--kl-warmup 20 "
        "--mmd-warmup 20 "
        "--montecarlo-kl 10 "
        "--encoding-size {wildcards.encs} "
        "--batch-size 256 "
        "--window-size 11 "
        "--window-step 11 "
        "--stability-check 3  "
        "--output-path {outpath}pheno_classification_experiments"
