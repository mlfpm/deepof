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
        st.just([]),
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
    if load_range is not None and len(load_range)==2 and load_range[1]-load_range[0]>1:
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
    # normal case
    if len(adj_load_range)>0:
        if only_metainfo:
            pass
        elif table_type=="numpy":
            assert (save_dict['1'][adj_load_range]==data).all()
        elif table_type=="panda":
            assert (np.array(save_dict['1'].iloc[adj_load_range])==np.array(data)).all()
        else:
            assert (save_dict['1'][0][adj_load_range]==data[0]).all()
            assert (save_dict['1'][1][adj_load_range]==data[1]).all()
    # an empty range was provided resulting in loading an empty object
    else:
        if only_metainfo:
            pass
        elif table_type=="numpy": 
            assert isinstance(data, np.ndarray) and data.shape==(0,0)
        elif table_type=="panda": 
            assert (save_dict['1'].columns.astype(str)==data.columns.astype(str)).all()
        else:   
            assert isinstance(data[0], np.ndarray) and data[0].shape==(0,0)
            assert isinstance(data[1], np.ndarray) and data[1].shape==(0,0)


def test_duckdb_table_handling(tmp_path):
    import pytest
    from deepof.data import _save_replacing_table
    from deepof.data_manager import DataManager

    key = "20191203_Test_5"
    os.makedirs(os.path.join(tmp_path, key))
    # Replacing a window table with the reshaped tuple does not accumulate type suffixes or keep the old table
    entry = save_dt(np.zeros((10, 4, 3)), os.path.join(tmp_path, key, key + "_preprocessed"), True)
    tab, path = get_dt({key: entry}, key, return_path=True)
    entry = _save_replacing_table((tab[:, :2], tab[:, 2:]), path, True)
    assert entry["table"] == f"t_{key}_preprocessed__npz"
    with DataManager(entry["duckdb_file"]) as manager:
        assert [r[0] for r in manager.conn.execute("SHOW TABLES").fetchall()] == [entry["table"]]
    assert [a.shape for a in get_dt({key: entry}, key)] == [(10, 2, 3), (10, 2, 3)]

    # Loading from a missing database raises instead of creating an empty one
    missing = os.path.join(tmp_path, "missing", "database.duckdb")
    with pytest.raises(FileNotFoundError):
        get_dt({key: {"duckdb_file": missing, "table": "x"}}, key)
    assert not os.path.exists(missing)

    # Duplicate columns are renamed in place
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "a", "a"])
    with DataManager(entry["duckdb_file"]) as manager:
        assert list(manager._prepare_dataframe(df).columns) == ["a", "b", "a_duplicate0", "a_duplicate1"]
