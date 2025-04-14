# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof

"""Data loading functionality for the deepof package."""
import copy
import os

import ast
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from typing import Any, List, NewType, Tuple, Union
import warnings
from .data_manager import DataManager, sanitize_table_name

# DEFINE WARNINGS FUNCTION
def _suppress_warning(warn_messages):
    def somedec_outer(fn):
        def somedec_inner(*args, **kwargs):
            # Some warnings do not get filtered when record is not True
            with warnings.catch_warnings(record=True) as caught_warnings:
                for k in range(0, len(warn_messages)):
                    pattern=f"(\n)?.*{warn_messages[k]}.*"
                    warnings.filterwarnings("ignore", message=pattern)
                response = fn(*args, **kwargs)
            #display caught warnings (all warnings that were not ignored)    
            for caught_warning in caught_warnings:
                warnings.warn(caught_warning.message)
            return response
        
        return somedec_inner
    
    return somedec_outer



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)

def get_dt(
    tab_dict: dict,
    key: str,
    return_path: bool = False,
    only_metainfo: bool = False,
    load_index: bool = False,
    load_range: np.ndarray = None
):
    raw_data = tab_dict.get(key)
    path=''

    # In-memory DataFrame
    if isinstance(raw_data, pd.DataFrame):
        if only_metainfo:
            return {
                "columns": raw_data.columns.tolist(),
                "num_cols": raw_data.shape[1],
                "num_rows": raw_data.shape[0],
                "shape": raw_data.shape,
                "index_column": raw_data.index if load_index else None,
                "start_time":raw_data.index[0] if load_index else None,
                "end_time":raw_data.index[-1] if load_index else None
            }
        if load_range is None:
            sliced = raw_data
        elif isinstance(load_range, list) and len(load_range) == 2:
            sliced = raw_data.iloc[load_range[0]:load_range[1]+1]
        else:
            sliced = raw_data.iloc[load_range]

        #sliced = raw_data if load_range is None else raw_data.iloc[load_range[0]:load_range[1]+1]
        return (sliced, '') if return_path else sliced

    # In-memory array or tuple
    if isinstance(raw_data, (np.ndarray, tuple)):
        shape = raw_data[0].shape if isinstance(raw_data, tuple) else raw_data.shape
        if only_metainfo:
            return {"shape": shape, "num_rows": shape[0], "num_cols": shape[1] if len(shape) > 1 else 1}

        if load_range is not None:
            if isinstance(raw_data, tuple):
                if len(load_range) == 2:
                    raw_data = tuple(arr[load_range[0]:load_range[1]+1] for arr in raw_data)
                else:
                    raw_data = tuple(arr[load_range] for arr in raw_data)
            else:
                if len(load_range) == 2:
                    raw_data = raw_data[load_range[0]:load_range[1]+1]
                else:
                    raw_data = raw_data[load_range]

        return (raw_data, '') if return_path else raw_data

    # DuckDB-stored
    if isinstance(raw_data, dict) and "duckdb_file" in raw_data:
        db_path = raw_data["duckdb_file"]
        table_name = sanitize_table_name(raw_data["table"])

        with DataManager(db_path) as manager:
            result = manager.load(
                table_name,
                return_path=return_path,
                only_metainfo=only_metainfo,
                load_index=load_index,
                load_range=load_range
            )      

        if only_metainfo:
            return (result, '') if return_path else result
        else:
            if len(result) == 2:
                return (result[0], result[1]) if return_path else result[0]
            else:
                return (result, '') if return_path else result
            
    return (raw_data, path) if return_path else raw_data


@_suppress_warning(
    warn_messages=[
        "Creating an ndarray from ragged nested sequences .* is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray."
    ]
)

def save_dt(
    dt: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]],
    folder_path: str,
    return_path: bool = False
):
    if not folder_path:
        return dt

    db_path = os.path.join(os.path.dirname(folder_path), "database.duckdb")
    key = os.path.basename(folder_path)
    #manager = DataManager(db_path)
    with DataManager(db_path) as manager:
        manager.save(key, dt)
    #manager.close()

    return {"duckdb_file": db_path, "table": sanitize_table_name(key)} if return_path else dt


