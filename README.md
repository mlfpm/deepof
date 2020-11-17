[![Maintainability](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/pipeline.svg)](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/pipelines)
[![Maintainability](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/coverage.svg)](https://coverage.readthedocs.io/en/coverage-5.3/)
[![Maintainability](https://api.codeclimate.com/v1/badges/c8c92dd2cb077c1beaa3/maintainability)](https://codeclimate.com/github/lucasmiranda42/deepof)
[![Maintainability](https://img.shields.io/badge/release-v0.0.1-informational)](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/blob/master/LICENSE)
[![Maintainability](https://img.shields.io/badge/funding-MLFPM-informational)](https://mlfpm.eu/)

# DeepOF
### A suite for postprocessing time-series extracted from videos of freely moving animals using [DeepLabCut](http://www.mousemotorlab.org/deeplabcut)
#

You can use this package to either extract pre-defined motifs from the time series (such as time-in-zone, climbing, 
basic social interactions) or to embed your data into a sequence-aware latent space to extract meaningful motifs in an
unsupervised way! Both of these can be used within the package, for example, to automatically 
compare user-defined experimental groups.

### How do I start?
##### Installation: 
clone the repository, go to its containing folder and type 
``` pip install -e deepof ``` (Careful: this will change upon PyPI submission!).

##### Before we delve in:
To start, create a folder for your project
 with at least two subdirectories inside, called 'Videos' and 'Tables'. The former should contain the videos you're
 working with (either you original data or the labeled ones obtained from DLC); the latter should have all the tracking 
 tables you got from DeepLabCut, either in .h5 or .csv format. If you don't want to use DLC yourself, don't worry:
 a compatible pre-trained model for mice will be released soon!
```
my_project  -- Videos -> all tagged videos
            |
            |
            -- Tables -> all tracking tables (.h5 or .csv)
```
IMPORTANT: You should make sure that the tables and videos correspond to the same experiments. While the names should 
be compatible, this is handled by DLC by default.

##### Basic usage:

The main module with which you'll interact is called ```deepof.data```. Let's import it and create a project:
```
import deepof.data
my_project = deepof.data.project(path="./my_project",
                                 smooth_alpha=0.99)
```
This command will create a ```deepof.data.project``` object storing all the necessary information to start. The ```smooth_alpha```
parameter will control how much smoothing will be applied to your trajectories, using an exponentially weighted average.
Values close to 0 apply a stronger smoothing, and values close to 1 a very light one. In practice, we recommend values
between 0.95 and 0.99 if your trajectories are not too noisy. There are other things you can do here, but let's stick to
the basics for now.  
  
One you have this, you can run you project using the ```.run()``` method, which will do quite a lot of computing under
the hood (load your data, smooth your trajectories, compute distances and angles). The returned object belongs to the 
```deepof.data.coordinates``` class.
```
my_project = my_project.run(verbose=True)
```

Once you have this, you can do several things! But let's first explore how the results of those computations I mentioned
are stored. To extract trajectories, distances and/or angles, you can respectively type:
```
my_project_coords = my_project.get_coords(center=True, polar=False, speed=0, align="Nose", align_inplace=True)
my_project_dists  = my_project.get_distances(speed=0)
my_project_angles = my_project.get_angles(speed=0)
```
Here, the data are stored as ```deepof.data.table_dict``` instances. These are very similar to python dictionaries
with experiment IDs as keys and pandas.DataFrame objects as values, with a few extra methods for convinience. Peeping
into the parameters you see in the code block above, ```center``` centers your data (it can be either a boolean or
one of the body parts in your model! in which case the coordinate origin will be fixed to the position of that point);
```polar``` makes the ```.get_coords()``` method return polar instead of Cartesian coordinates, and ```speed``` 
indicates the derivation level to apply (0 is position-based, 1 speed, 2 acceleration, 3 jerk, etc). Regarding 
```align``` and ```align-inplace```, they take care of aligning the animal position to the y Cartesian axis: if we
center the data to "Center" and set ```align="Nose", align_inplace=True```, all frames in the video will be alignes in a
way that will keep the Center-Nose axis fixed. This is useful to constrain the set of movemets that one can extract
with out unsupervised methods. 