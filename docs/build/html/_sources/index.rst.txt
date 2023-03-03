
|Pipeline| |Coverage| |Docs| |CodeFactor| |Version| |MLFPM| |Black|

.. |Pipeline| image:: https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/pipeline.svg
   :target: https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/pipelines
.. |Coverage| image:: https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/coverage.svg
   :target: https://coverage.readthedocs.io/en/coverage-5.3/
.. |Docs| image:: https://readthedocs.org/projects/deepof/badge/?version=latest
   :target: https://deepof.readthedocs.io/en/latest
   :alt: Documentation Status
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/lucasmiranda42/deepof/badge
   :target: https://www.codefactor.io/repository/github/lucasmiranda42/deepof
.. |Version| image:: https://img.shields.io/badge/release-v0.1.6-informational
   :target: https://pypi.org/project/deepof/
.. |MLFPM| image:: https://img.shields.io/badge/funding-MLFPM-informational
   :target: https://pypi.org/project/deepof/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-black
   :target: https://github.com/psf/black

Welcome to DeepOF!
==================

A suite for postprocessing time-series extracted from videos of freely moving rodents using `DeepLabCut <http://www.mousemotorlab.org/deeplabcut>`_

.. image:: https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/raw/master/logos/deepOF_logo_w_text.png
  :width: 400
  :align: center
  :alt: DeepOF logo

Getting started
===============
You can use this package to either extract pre-defined motifs from the time series (such as time-in-zone, climbing,
basic social interactions) or to embed your data into a sequence-aware latent space to extract meaningful motifs in an
unsupervised way! Both of these can be used within the package, for example, to automatically
compare user-defined experimental groups.

Installation
------------

The easiest way to install DeepOF is to use `pip <https://pypi.org/project/deepof/>`_:

.. code:: bash

   pip install deepof

Alternatively, you can download our pre-built `Docker image <https://hub.docker.com/repository/docker/lucasmiranda42/deepof>`_,
which contains all compatible dependencies:

.. code:: bash

   # download the latest available image
   docker pull lucasmiranda42/deepof:latest
   # run the image in interactive mode, enabling you to open python and import deepof
   docker run -it lucasmiranda42/deepof

What you need
-------------
DeepOF relies heavily on DeepLabCut's output. Thorough tutorials on how to get started with DLC for pose estimation can be found `here <https://www.mousemotorlab.org/deeplabcut>`_.
Once your videos are processed and tagged, you can use DeepOF to extract and annotate your motion-tracking time-series. Currently, DeepOF requires videos to be filmed from a top-view perspective, and follow a set of labels
equivalent to the ones shown in the figure below. A pre-trained model capable of recognizing **C57Bl6** and **CD1** mice can be downloaded from `our repository <https://gitlab.mpcdf.mpg.de/lucasmir/deepof/tree/master/models>`_.

.. image:: _static/deepof_DLC_tagging.png
   :width: 400
   :align: center
   :alt: DeepOF label scheme

**NOTE**: Some DeepOF functions (such as climbing detection) currently require the user to film their animals in a round arena. This is scheduled to be
updated in future releases.

Basic usage
-----------
To start, create a folder for your project with at least two subdirectories inside, called 'Videos' and 'Tables'. The former should contain the videos you're
working with (either you original data or the labeled ones obtained from DLC); the latter should have all the tracking
tables you got from DeepLabCut, either in .h5 or .csv format. If you don't want to use DLC yourself, don't worry:
a compatible pre-trained model for mice will be released soon!

.. code:: bash

   my_project
   ├── Videos -> all tagged videos
   ├── Tables -> all tracking tables (.h5 or .csv)

IMPORTANT: You should make sure that the tables and videos correspond to the same experiments. While the names should
be compatible, this is handled by DLC by default.

The main module with which you'll interact is called ```deepof.data```. Let's import it and create a project:

.. code:: python

   import deepof.data
   my_project = deepof.data.Project(path="./my_project",
                                    arena_dims=380,        # diameter of the arena in milimeters
                                    arena_type="circular", # type of the filmed arena (optional). So far, only "circular" is valid
                                    smooth_alpha=2,        # smoothing coefficient (optional)
                                    frame_rate=25)         # frame rate of the videos in Hz (optional)

This command will create a ```deepof.data.Project``` object storing all the necessary information to start. The ```smooth_alpha```
parameter will control how much smoothing will be applied to your trajectories, using an exponentially weighted average.
Values close to 0 apply a stronger smoothing, and values close to 1 a very light one. In practice, we recommend values
between 0.95 and 0.99 if your trajectories are not too noisy. There are other things you can do here, but let's stick to
the basics for now.

One you have this, you can run you project using the ```.run()``` method, which will do quite a lot of computing under
the hood (load your data, smooth your trajectories, compute distances and angles). The returned object belongs to the
```deepof.data.Coordinates``` class.

.. code:: python

   my_project = my_project.run(verbose=True)

Once you have this, you can do several things! But let's first explore how the results of those computations I mentioned
are stored. To extract trajectories, distances and/or angles, you can respectively type:

.. code:: python

   my_project_coords = my_project.get_coords(center=True, polar=False, speed=0, align="Nose", align_inplace=True)
   my_project_dists  = my_project.get_distances(speed=0)
   my_project_angles = my_project.get_angles(speed=0)

Here, the data are stored as ```deepof.data.table_dict``` instances. These are very similar to python dictionaries
with experiment IDs as keys and pandas.DataFrame objects as values, with a few extra methods for convenience. Peeping
into the parameters you see in the code block above, ```center``` centers your data (it can be either a boolean or
one of the body parts in your model! in which case the coordinate origin will be fixed to the position of that point);
```polar``` makes the ```.get_coords()``` method return polar instead of Cartesian coordinates, and ```speed```
indicates the derivation level to apply (0 is position-based, 1 speed, 2 acceleration, 3 jerk, etc). Regarding
```align``` and ```align-inplace```, they take care of aligning the animal position to the y Cartesian axis: if we
center the data to "Center" and set ```align="Nose", align_inplace=True```, all frames in the video will be aligned in a
way that will keep the Center-Nose axis fixed. This is useful to constrain the set of movements that one can extract
with out unsupervised methods.

As mentioned above, the two main analyses that you can run are supervised and unsupervised. They are executed by
the ```.supervised_annotation()``` method, and the ```.deep_unsupervised_embedding()``` methods of the ```deepof.data.Coordinates```
class, respectively.

.. code:: python

   supervised_annot = my_project.supervised_annotation()
   gmvae_embedding  = my_project.deep_unsupervised_embedding()

The former returns a ```deepof.data.TableDict``` object, with a pandas.DataFrame per experiment containing a series of
annotations. The latter is a bit more complicated: it returns an array containing the encoding of the data per animal,
another one with motif membership per time point (probabilities of the animal doing whatever is represented by each of
the clusters at any given time), an abstract distribution (a multivariate Gaussian mixture) representing the extracted
components, and a decoder you can use to generate samples from each of the extracted components (yeah,
you get a generative model for free).

That's it for this (very basic) introduction. Check out the tutorials below for more advanced examples!

Tutorials
=========

* `Formatting your data: feature extraction from DLC output <tutorial_notebooks/deepof_preprocessing_tutorial.ipynb>`_
* `DeepOF supervised pipeline: detecting pre-defined behaviors <tutorial_notebooks/deepof_supervised_tutorial.ipynb>`_
* `DeepOF unsupervised pipeline: exploring the behavioral space <tutorial_notebooks/deepof_unsupervised_tutorial.ipynb>`_

Full API reference
==================

* `deepof.data (main data-wrangling module) <deepof.data.html>`_
* `deepof.utils (data-wrangling auxiliary functions) <deepof.utils.html>`_
* `deepof.models (deep unsupervised models) <deepof.models.html>`_
* `deepof.hypermodels (deep unsupervised hypermodels for hyperparameter tuning) <deepof.hypermodels.html>`_
* `deepof.annotation_utils (deep rule-based annotation auxiliary functions) <deepof.annotation_utils.html>`_
* `deepof.model_utils (deep machine learning models' auxiliary functions) <deepof.model_utils.html>`_
* `deepof.visuals (auxiliary visualization functions) <deepof.visuals.html>`_
* `deepof.post_hoc (auxiliary annotation analysis functions) <deepof.post_hoc.html>`_

