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

outpath = "/u/lucasmir/DLC/DLC_autoencoders/DeepOF/deepof/logs/"
losses = ["ELBO"]  # , "MMD", "ELBO+MMD"]
encodings = [4, 6, 8]  # [2, 4, 6, 8, 10, 12, 14, 16]
cluster_numbers = [10, 15, 20]  # [1, 5, 10, 15, 20]
pheno_weights = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0]


rule deepof_experiments:
    input:
        expand( "/u/lucasmir/DLC/DLC_autoencoders/DeepOF/deepof/logs/hyperparameter_tuning/trained_weights/"
                "GMVAE_loss={loss}_encoding=2_k=15_run_1_final_weights.h5",
                loss=losses,
        )

        # expand(
        #     "/u/lucasmir/DLC/DLC_autoencoders/DeepOF/deepof/logs/dimension_and_loss_experiments/trained_weights/"
        #     "GMVAE_loss={loss}_encoding={encs}_k={k}_run_1_final_weights.h5",
        #     loss=losses,
        #     encs=encodings,
        #     k=cluster_numbers,
        # ),
        # expand(
        #     "/u/lucasmir/DLC/DLC_autoencoders/DeepOF/deepof/logs/pheno_classification_experiments/trained_weights/"
        #     "GMVAE_loss={loss}_encoding={encs}_k={k}_pheno={phenos}_run_1_final_weights.h5",
        #     loss=losses,
        #     encs=encodings,
        #     k=cluster_numbers,
        #     phenos=pheno_weights,
        # ),


rule coarse_hyperparameter_tuning:
    input:
        data_path="/u/lucasmir/DLC/DLC_models/deepof_single_topview/",
    output:
        trained_models=os.path.join(
            outpath,
            "hyperparameter_tuning/trained_weights/GMVAE_loss={loss}_encoding=2_run_1_final_weights.h5",
        ),
    shell:
        "pipenv run python -m deepof.train_model "
        "--train-path {input.data_path} "
        "--val-num 25 "
        "--components 15 "
        "--input-type coords "
        "--predictor 0 "
        "--phenotype-classifier 0 "
        "--variational True "
        "--loss {wildcards.loss} "
        "--kl-warmup 20 "
        "--mmd-warmup 0 "
        "--encoding-size 2 "
        "--batch-size 256 "
        "--window-size 24 "
        "--window-step 12 "
        "--output-path {outpath}coarse_hyperparameter_tuning "
        "--hyperparameter-tuning hyperband "
        "--hpt-trials 3"


# rule explore_encoding_dimension_and_loss_function:
#     input:
#         data_path=ancient("/u/lucasmir/DLC/DLC_models/deepof_single_topview/"),
#     output:
#         trained_models=os.path.join(
#             outpath,
#             "dimension_and_loss_experiments/trained_weights/GMVAE_loss={loss}_encoding={encs}_k={k}_run_1_final_weights.h5",
#         ),
#     shell:
#         "pipenv run python -m deepof.train_model "
#         "--train-path {input.data_path} "
#         "--val-num 5 "
#         "--components {wildcards.k} "
#         "--input-type coords "
#         "--predictor 0 "
#         "--phenotype-classifier 0 "
#         "--variational True "
#         "--loss {wildcards.loss} "
#         "--kl-warmup 20 "
#         "--mmd-warmup 20 "
#         "--montecarlo-kl 10 "
#         "--encoding-size {wildcards.encs} "
#         "--batch-size 256 "
#         "--window-size 24 "
#         "--window-step 6 "
#         "--exclude-bodyparts Tail_base,Tail_1,Tail_2,Tail_tip "
#         "--stability-check 3 "
#         "--output-path {outpath}dimension_and_loss_experiments"
#
#
# rule explore_phenotype_classification:
#     input:
#         data_path="/u/lucasmir/DLC/DLC_models/deepof_single_topview/",
#     output:
#         trained_models=os.path.join(
#             outpath,
#             "pheno_classification_experiments/trained_weights/GMVAE_loss={loss}_encoding={encs}_k={k}_pheno={phenos}_run_1_final_weights.h5",
#         ),
#     shell:
#         "pipenv run python -m deepof.train_model "
#         "--train-path {input.data_path} "
#         "--val-num 15 "
#         "--components {wildcards.k} "
#         "--input-type coords "
#         "--predictor 0 "
#         "--phenotype-classifier {wildcards.phenos} "
#         "--variational True "
#         "--loss {wildcards.loss} "
#         "--kl-warmup 20 "
#         "--mmd-warmup 20 "
#         "--montecarlo-kl 10 "
#         "--encoding-size {wildcards.encs} "
#         "--batch-size 256 "
#         "--window-size 11 "
#         "--window-step 11 "
#         "--stability-check 3  "
#         "--output-path {outpath}pheno_classification_experiments"




"pipenv run python -m deepof.train_model --train-path {input.data_path} --val-num 5 --components {wildcards.k} --input-type coords --predictor 0 --phenotype-classifier 0 --variational True --loss {wildcards.loss} --kl-warmup 20 --mmd-warmup 20 --montecarlo-kl 10 --encoding-size {wildcards.encs} --batch-size 256 --window-size 24 --window-step 6 --exclude-bodyparts Tail_base,Tail_1,Tail_2,Tail_tip --stability-check 3 --output-path {outpath}dimension_and_loss_experiments"