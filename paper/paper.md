---
title: 'DeepOF: a Python package for supervised and unsupervised pattern recognition in mice motion tracking data'
tags:
  - Python
  - biology
  - neuroscience
  - behavioral annotation
authors:
  - name: Lucas Miranda
    orcid: 0000-0002-5484-6744
    affiliation: 1
  - name: Joeri Bordes
    orcid: 0000-0003-2909-2976
    affiliation: 2
  - name: Benno Pütz
    orcid: 0000-0002-2208-209X
    affiliation: 1
  - name: Mathias V Schmidt
    orcid: 0000-0002-3788-2268
    affiliation: 2
  - name: Bertram Müller-Myhsok
    orcid: 0000-0002-0719-101X
    corresponding: true
    affiliation: 1
affiliations:
 - name: Research Group Statistical Genetics, Max Planck Institute of Psychiatry, Munich, Germany
   index: 1
 - name: Research Group Neurobiology of Stress Resilience, Max Planck Institute of Psychiatry, Munich, Germany
   index: 2
 
date: 04 April 2023
bibliography: paper.bib
---

# Summary

DeepOF (Deep Open Field) is a Python package that provides a suite of tools for analyzing behavior in 
freely-moving rodents. Specifically, it focuses on postprocessing time-series data extracted from videos using 
[DeepLabCut](http://www.mackenziemathislab.org/deeplabcut#:~:text=DeepLabCut%E2%84%A2%20is%20an%20efficient,typically%2050%2D200%20frames) [@Mathis2018DeepLabCut:Learning]. 
The software encompasses a diverse set of capabilities, such as:

* Loading DeepLabCut data into custom objects and incorporating metadata related to experimental design.
* Processing data, including smoothing, imputation, and feature extraction.
* Annotating behavioral motifs in a supervised manner, such as recognizing huddling and climbing, and detecting fundamental social interactions between animals.
* Embedding motion tracking data in an unsupervised manner using neural network models, which also facilitate end-to-end deep clustering.
* Conducting post-hoc analysis of results and visualization to compare patterns across animals under different experimental conditions.

The package is designed to work with various types of DeepLabCut input (single and multi-animal projects), includes comprehensive documentation, 
and offers interactive tutorials. Although many of its primary functionalities (particularly the supervised annotation pipeline) 
were developed with top-down mice videos in mind, tagged with a specific set of labels, most essential functions operate 
without constraints. As demonstrated in the accompanying scientific application paper [@Bordes2022.06.23.497350], DeepOF has the potential to enable systematic and thorough 
behavioral assessments in a wide range of preclinical research settings.

# Statement of need

The field of behavioral research has experienced significant advancements in recent years, particularly in the 
quantification and analysis of animal behavior. Historically, behavioral quantification relied heavily on tests that were designed with
either one or a few readouts in mind. However, the advent of deep learning for computer vision and the development of packages such as DeepLabCut, which enable 
pose estimation without the need for physical markers, have rapidly expanded the possibilities for non-invasive animal 
tracking [@Mathis2020APerspectives].

By transforming raw video footage into time series data of tracked body parts, these approaches have paved the way for 
the development of software packages capable of automatically annotating behavior following a plethora of different 
approaches, increasing the number of patterns that can be studied per experiment with little burden on the experimenters. 

For example, several tools offer options to detect predefined behaviors using supervised machine learning. Along these lines, programs like SimBA [@Nilsson2020.04.19.049452], MARS [@MARS2021], or TREBA [@sun2021task], allow 
users to label a set of behaviors and train classifiers to detect them in new videos. They employ different labelling schemes which require different amounts of user input, and offer high flexibility in terms of the number of behaviors that can be detected.
On the other hand, packages such as B-SOiD [@Hsu2021], VAME [@Luxem2022IdentifyingMotion], and Keypoint-MoSeq [@Weinreb2023.03.16.532307], aim for a more exploratory approach that does not require user labelling, but instead relies on unsupervised learning to segment time series into different behaviors.
These packages are particularly useful when the user is interested in detecting novel behaviors, or when the number of behaviors is too large to be annotated manually. Moreover, some approaches have been developed to combine the best of both worlds, such as the
the A-SOiD active learning framework [@Schweihoff2022.11.04.515138], and the semi-supervised DAART [@Whiteway2021.06.16.448685]. While a thorough discussion on the advantages and disadvantages of each package is beyond the scope of this paper, further information can be found in this recent review [@Bordes2023]. 

In contrast to other available options, DeepOF offers both supervised and unsupervised annotation pipelines, 
that allow researchers to test hypotheses regarding experimental conditions such as stress, gene mutations, and sex, 
in a flexible way (\autoref{fig:intro}).

![Scheme representing DeepOF workflow. Upon creating a project, DLC data can be loaded and preprocessed before annotating it with either a supervised pipeline (which uses a set of pre-trained models and rule-based annotators) or an unsupervised pipeline, which relies on custom deep clustering algorithms. Patterns retrieved with either pipeline can be passed to downstream post-hoc analysis tools and visualization functions.\label{fig:intro}](../docs/source/_static/deepof_pipelines.png){width=80%}

The included supervised pipeline uses a series of rule-based annotators and pre-trained machine learning classifiers to detect when each animal is 
displaying a set of pre-defined behavioral motifs. The unsupervised workflow uses state-of-the-art deep clustering models to extract novel motifs without prior definition. DeepOF then provides an interpretability pipeline to explore what these retrieved clusters are in terms of behavior, which uses both Shapley Additive Explanations (SHAP) [@Goodwin2022] and direct mappings from clusters to video.
Moreover, regardless of whether the user chose the supervised annotation pipeline, the unsupervised one, or both, DeepOF provides an extensive set of post-hoc analysis and visualization tools.

When it comes to comparing it to other individual packages that use supervised and unsupervised annotation, DeepOF stands out in several ways. First of all, it is the first package, to the best of our knowledge, to offer both options. Second,
the supervised pipeline in DeepOF follows an opinionated philosophy, in the sense that it provides a set of pre-trained models that cannot be customized, but do not require user labels. This trades flexibility for ease of use, aiming at being a quick exploratory tool that can provide information on key individual and social behaviors with just a few commands.
Furthermore, when it comes to the unsupervised pipeline, DeepOF provides three custom deep clustering algorithms capable of segmenting the behavioral time series, as well as the aforementioned built-in interpretability pipeline. If a user runs both pipelines, supervised annotations can be incorporated into this interpretability pipeline in quite a unique way, to detect associations between supervised and unsupervised patterns.

All in all, DeepOF is a comprehensive, end-to-end tool designed to transform DeepLabCut output into relatively quick, exploratory insights on behavioral shifts between experimental conditions, and pinpoint which behaviors are driving them.

# Related literature

The DeepOF package has been used to characterize differences in behavior associated with Chronic Social Defeat Stress (CSDS)
in mice, as presented in our preprint (currently in revision at the time of writing [@Bordes2022.06.23.497350]). There are several other ongoing projects
involving the software, although none of them are published to this date.

# Acknowledgements

We acknowledge contributions from Felix Agakov and Karsten Borgwardt.

# Funding

This project has received funding from the European Union’s Framework
Programme for Research and Innovation Horizon 2020 (2014-2020) under
the Marie Skłodowska-Curie Grant Agreement No. 813533-MSCA-ITN-2018.

# References
