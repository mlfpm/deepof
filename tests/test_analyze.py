# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.analyze

"""
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import range_indexes, columns, data_frames

import deepof.data
import deepof.pose_utils
