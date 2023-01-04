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

automatic_changepoints = [False] #, "rbf", "linear"]
animal_to_preprocess = ["B", None]
kmeans_loss = [0.0]
encodings = [8]
cluster_numbers = [25] #list(range(5, 26, 1))
input_types = ["graph"]#, "coords"]
run = range(3)
embedding_model = ["VQVAE", "VaDE"]
encoder_model = ["recurrent", "TCN", "transformer"]

rule deepof_experiments:
    input:
        # Train a variety of models
        expand(
            outpath
            + "train_models/deepof_unsupervised_{embedding_model}_encoder_{encoder}_encodings_input={input_type}_k={k}_latdim={latdim}_changepoints_{automatic_changepoints}_kmeans_loss={kmeans_loss}_run={run}.pkl",
            embedding_model=embedding_model,
            encoder=encoder_model,
            input_type=input_types,
            k=cluster_numbers,
            latdim=encodings,
            automatic_changepoints=automatic_changepoints,
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
        + "train_models/deepof_unsupervised_{embedding_model}_encoder_{encoder}_encodings_input={input_type}_k={k}_latdim={latdim}_changepoints_{automatic_changepoints}_kmeans_loss={kmeans_loss}_run={run}.pkl",
    shell:
        "python deepof/deepof_train_embeddings.py "
        "--train-path {input.data_path} "
        "--embedding-model {wildcards.embedding_model} "
        "--encoder-type {wildcards.encoder} "
        "--automatic-changepoints {wildcards.automatic_changepoints} "
        "--val-num 5 "
        "--animal-id B,W "
        "--animal-to-preprocess {wildcards.animal_to_preprocess} "
        "--exclude-bodyparts Tail_1,Tail_2,Tail_tip "
        "--n-components {wildcards.k} "
        "--input-type {wildcards.input_type} "
        "--kmeans-loss {wildcards.kmeans_loss} "
        "--encoding-size {wildcards.latdim} "
        "--batch-size 128 "
        "--window-size 25 "
        "--window-step 1 "
        "--run {wildcards.run} "
        "--output-path {outpath}train_models"
