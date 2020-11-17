[![Maintainability](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/pipeline.svg)](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/pipelines)
[![Maintainability](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/badges/master/coverage.svg)](https://coverage.readthedocs.io/en/coverage-5.3/)
[![Maintainability](https://api.codeclimate.com/v1/badges/c8c92dd2cb077c1beaa3/maintainability)](https://codeclimate.com/github/lucasmiranda42/deepof)
[![Maintainability](https://img.shields.io/badge/release-v0.0.1-informational)](https://gitlab.mpcdf.mpg.de/lucasmir/deepof/-/blob/master/LICENSE)
[![Maintainability](https://img.shields.io/badge/funding-MLFPM-informational)](https://mlfpm.eu/)

# DeepOF
### A suite for postprocessing of time series extracted from videos of freely moving animals using DeepLabCut
#
### How do I start?
##### Installation: 
clone the repository, go to its containing folder and type 
``` pip install -e deepof ``` (Careful: this will change upon PyPI submission!).

##### Basic usage:
To start, create a folder for your project
 with at least two subdirectories inside, called 'Videos' and 'Tables'. The former should contain the videos you're
 working with (either you original data or the labeled ones obtained from DLC); the latter should have all the tracking 
 tables you got from DeepLabCut, either in .h5 or in .csv format. If you don't want to use DLC yourself, don't worry:
 a compatible pre-trained model for mice will be released soon!
```
my_project  -- Videos -> all tagged videos
            |
            |
            -- Tables -> all tracking tables (.h5 or .csv)
```
IMPORTANT: Your should make sure that the tables and videos correspond to the same experiments. While the names should 
be compatible, this is handled by DLC by default.

