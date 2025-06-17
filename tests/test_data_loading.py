# @author NoCreativeIdeaForGoodUserName
# encoding: utf-8
# module deepof

"""

Testing module for deepof.data_loading

"""

import os
import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis import reproduce_failure
from shutil import rmtree

from deepof.data_loading import (
    get_dt, save_dt
)


@settings(max_examples=300, deadline=None)
@given(
    table_type=st.one_of(
        st.just("numpy"),
        st.just("panda"),
        st.just("tuple"),
    ),
    return_path=st.booleans(),
    only_metainfo=st.booleans(),
    load_index=st.booleans(),
    load_range=st.one_of(
        st.just(None),
        st.lists(st.integers(min_value=0, max_value=99),min_size=1,max_size=100, unique=True).map(sorted)
    )

)
def test_get_dt_and_subfunctions(table_type, return_path, only_metainfo, load_index, load_range):
    # Create directory for saving stuff
    save_path=os.path.join(".", "tests", "test_examples", "save_folder")

    #Clear out possible remaining folder from last test
    if os.path.exists(save_path):
        rmtree(save_path)
    os.mkdir(save_path)

    # Create objects to save
    save_dict={}
    processed_dict={}
    if table_type=="numpy":
        save_dict['1']=np.random.rand(100, 5)
    elif table_type=="panda":
        save_dict['1']=pd.DataFrame(np.random.rand(100, 5))
    else:
        save_dict['1']=(np.random.rand(100, 5),np.random.rand(100, 5))

    #save data, keep either full dataset or path
    processed_dict['1'] = save_dt(save_dict['1'],os.path.join(save_path,'file1'),return_path)

    #get data again
    path=''
    if return_path:
        data, path=get_dt(processed_dict,'1',return_path,only_metainfo,load_index,load_range)
    else:
        data=get_dt(processed_dict,'1',return_path,only_metainfo,load_index,load_range)

    #remove saving structure
    rmtree(save_path)

    #formatting load range to avoid multiple if-else cases later
    adj_load_range=load_range
    if  load_range is not None and len(load_range)==2 and load_range[1]-load_range[0]>1:
        adj_load_range=np.arange(load_range[0],load_range[1]+1)
    elif load_range is None:
        adj_load_range=np.arange(0,100)

    #check functionality
    assert isinstance(path, (str, dict))
    if only_metainfo:
        assert isinstance(data, dict)
        assert 'num_rows' in data
        assert data['num_rows'] == 100
        assert 'num_cols' in data
        assert data['num_cols'] == 5
    elif table_type=="numpy":
        assert (save_dict['1'][adj_load_range]==data).all()
    elif table_type=="panda":
        assert (np.array(save_dict['1'].iloc[adj_load_range])==np.array(data)).all()
    else:
        assert (save_dict['1'][0][adj_load_range]==data[0]).all()
        assert (save_dict['1'][1][adj_load_range]==data[1]).all()