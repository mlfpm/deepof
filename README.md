[![Pipeline](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/pipeline.svg)](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/pipelines)
[![Coverage](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/coverage.svg)](https://coverage.readthedocs.io/en/coverage-5.3/)
[![Documentation Status](https://readthedocs.org/projects/deepof/badge/?version=latest)](https://deepof.readthedocs.io/en/latest)
[![CodeFactor](https://www.codefactor.io/repository/github/lucasmiranda42/deepof/badge)](https://www.codefactor.io/repository/github/lucasmiranda42/deepof)
[![Version](https://img.shields.io/badge/release-v0.3-informational)](https://pypi.org/project/deepof/)
[![MLFPM](https://img.shields.io/badge/funding-MLFPM-informational)](https://mlfpm.eu/)
[![Black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

<br>

<div align="center">
  <img width="350" height="350" src="https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/raw/master/logos/deepOF_logo_w_text.png">
</div>

### A suite for postprocessing time-series extracted from videos of freely moving rodents using [DeepLabCut](http://www.mousemotorlab.org/deeplabcut)
#

You can use this package to either extract pre-defined motifs from the time series (such as time-in-zone, climbing, 
basic social interactions) or to embed your data into a sequence-aware latent space to extract meaningful motifs in an
unsupervised way! Both of these can be used within the package, for example, to automatically 
compare user-defined experimental groups.

### How do I start?
##### Installation:

The easiest way to install DeepOF is to use [pip](https://pypi.org/project/deepof). Create and activate a virtual environment with Python >=3.9 and <3.11, for example using conda:

```bash
conda create -n deepof python=3.9
```

Then, activate the environment and install DeepOF:

```bash
conda activate deepof
pip install deepof
```

Alternatively, you can download our pre-built [Docker image](https://hub.docker.com/repository/docker/lucasmiranda42/deepof),
which contains all compatible dependencies:

```bash
# download the latest available image
docker pull lucasmiranda42/deepof:latest
# run the image in interactive mode, enabling you to open python and import deepof
docker run -it lucasmiranda42/deepof
```

Or use [poetry](https://python-poetry.org/):

```bash
# after installing poetry and clonning the DeepOF repository, just run
poetry install # from the main directory
```

##### Before we delve in:
DeepOF relies heavily on DeepLabCut's output. Thorough tutorials on how to get started with DLC for pose estimation can be found [here](https://www.mousemotorlab.org/deeplabcut).
Once your videos are processed and tagged, you can use DeepOF to extract and annotate your motion-tracking time-series. While many features in DeepOF can work regardless of the set of labels used, we currently recommend using videos from a top-down perspective, and follow our recommended
set of labels (which can be found in the full documentation page). A pre-trained model capable of recognizing **C57Bl6** and **CD1** mice can be downloaded from [our repository](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/tree/master/models).

##### Basic usage:

The main module with which you'll interact is called ```deepof.data```. Let's import it and create a project:

```python
import deepof.data
my_deepof_project = deepof.data.Project(
  project_path=".", # Path where to create project files
  video_path="/path/to/videos", # Path to DLC tracked videos
  table_path="/path/to/tables", # Path to DLC output
  project_name="my_deepof_project", # Name of the current project
  exp_conditions={exp_ID: exp_condition} # Dictionary containing one or more experimental conditions per provided video
)
```

This command will create a ```deepof.data.Project``` object storing all the necessary information to start. There are
many parameters that we can set here, but let's stick to the basics for now.

One you have this, you can run you project using the ```.create()``` method, which will do quite a lot of computing under
the hood (load your data, smooth your trajectories, compute distances, angles, and areas between body parts, and save all
results to disk). The returned object belongs to the ```deepof.data.Coordinates``` class.

```python
my_project = my_project.create(verbose=True)
```

Once you have this, you can do several things! But let's first explore how the results of those computations mentioned
are stored. To extract trajectories, distances, angles and/or areas, you can respectively type:

```python
my_project_coords = my_project.get_coords(center="Center", polar=False, align="Nose", speed=0)
my_project_dists  = my_project.get_distances(speed=0)
my_project_angles = my_project.get_angles(speed=0)
my_project_areas = my_project.get_areas(speed=0)
```

Here, the data are stored as ```deepof.data.table_dict``` instances. These are very similar to python dictionaries
with experiment IDs as keys and pandas.DataFrame objects as values, with a few extra methods for convenience. Peeping
into the parameters you see in the code block above, ```center``` centers your data (it can be either a boolean or
one of the body parts in your model! in which case the coordinate origin will be fixed to the position of that point);
```polar``` makes the ```.get_coords()``` method return polar instead of Cartesian coordinates, and ```speed```
indicates the derivation level to apply (0 is position-based, 1 speed, 2 acceleration, 3 jerk, etc). Regarding
```align``` and ```align-inplace```, they take care of aligning the animal position to the y Cartesian axis: if we
center the data to "Center" and set ```align="Nose", align_inplace=True```, all frames in the video will be aligned in a
way that will keep the Center-Nose axis fixed. This is useful to constrain the set of movements that one can extract
with our unsupervised methods.

As mentioned above, the two main analyses that you can run are supervised and unsupervised. They are executed by
the ```.supervised_annotation()``` method, and the ```.deep_unsupervised_embedding()``` methods of the ```deepof.data.Coordinates```
class, respectively.

```python
supervised_annot = my_project.supervised_annotation()
gmvae_embedding  = my_project.deep_unsupervised_embedding()
```

The former returns a ```deepof.data.TableDict``` object, with a pandas.DataFrame per experiment containing a series of
annotations. The latter is a bit more complicated: it returns a series of objects that depend on the model selected (we 
offer three flavours of deep clustering models), and allow for further analysis comparing cluster expression and dynamics.

That's it for this (very basic) introduction. Check out the tutorials and [full documentation](https://deepof.readthedocs.io/en/latest/index.html) for details!

---
### Issues

If you encounter any problems while using this package, please open an issue in the [issue tracker](https://github.com/mlfpm/deepof/issues).

---
### Contributions

We welcome contributions from the community! If you want to contribute to this project, please check out our [contribution guidelines](https://github.com/mlfpm/deepof/blob/master/CONTRIBUTING.md).

---

 This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No.  813533
 <div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/255px-Flag_of_Europe.svg.png">
</div>

---
