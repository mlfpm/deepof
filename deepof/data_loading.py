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
from typing import Any, List, NewType, Tuple, Union
import warnings


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
        tab_dict: [table_dict, dict],
        key: str,
        return_path: bool = False,
        only_metainfo: bool = False,
        load_index: bool = False,
        load_range: np.ndarray = None,
        ):
    """retrieves data table from table dict 
    (I use this a lot, so it gets its own function)
    
    Args:
        tab_dict ([table_dict, dict]): Table Dict or dictionary with data tables (or paths)
        key (str): key to dict entry
        return_path (bool): Additionally return the path to the saving location of the table
        only_metainfo (bool): Return only meta info like numbers of rows and columns without loading the full table
        load_index (bool): Return index data additionally to meta_info
         
    Returns:
        Data table after loading
        Path to data table (if requested) 
    """

    if tab_dict is None:
        return None
    
    raw_data = tab_dict.get(key)
    path=''

    # Extract data if dictionary entry is a string
    if isinstance(raw_data, str):
        path = raw_data

        if only_metainfo:
            raw_data = load_dt_metainfo(raw_data,load_index)
        else:
            raw_data = load_dt(raw_data, load_range)
    
    # Extract metainfo if dictionary entry is a data frame
    elif only_metainfo:
        raw_data = get_metainfo_from_loaded_dt(raw_data,load_index)

    # Cut data into shape if it is already loaded but a load range was given
    elif isinstance(raw_data, np.ndarray) and load_range is not None:
        if len(load_range) ==2 and load_range[1]-load_range[0]> 1:
            raw_data = raw_data[load_range[0]:load_range[1]+1]   
        else:
            raw_data = raw_data[load_range]   

    elif isinstance(raw_data, pd.DataFrame) and load_range is not None:
        if len(load_range) ==2 and load_range[1]-load_range[0]> 1:
            raw_data = raw_data.iloc[load_range[0]:load_range[1]+1]   
        else:
            raw_data = raw_data.iloc[load_range] 

    elif isinstance(raw_data, Tuple) and len(raw_data)==2 and load_range is not None:
        if len(load_range) ==2 and load_range[1]-load_range[0]> 1:
            raw_data = (raw_data[0][load_range[0]:load_range[1]+1],raw_data[1][load_range[0]:load_range[1]+1])
        else:
            raw_data = (raw_data[0][load_range], raw_data[1][load_range])

    # Invisible "else" case: if none of the above, do nothig, since this means 
    # raw_data is the full table in memory and hence already loaded 
      
    return (raw_data, path) if return_path else raw_data


@_suppress_warning(
    warn_messages=[
        "Creating an ndarray from ragged nested sequences .* is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray."
    ]
)
def save_dt(dt: pd.DataFrame, path: str, return_path: bool = False):
    """Saves a given data frame fast and efficient using parquet

    Args:
        dt (pd.DataFrame): dataframe for saving
        path (str): Where to save 
        keep_in_RAM (bool): False, If the object that gets saved should be returned, True and only the path to teh file location gets returned

    Returns:
        Either saved dataframe or path to save location
    """

    if path is None:
        #skip saving
        return dt
    
    path = os.path.splitext(path)[0]
    
    if isinstance(dt, np.ndarray): 
        path = path + '.npy'
        with open(path, 'wb') as file:
            np.lib.format.write_array(file, dt)

    elif (isinstance(dt, Tuple) and all(isinstance(dt_subset, np.ndarray) for dt_subset in dt)):
        path = path + '.npz'
        with open(path, 'wb') as file:
            np.savez(file, *[('arr_{}'.format(i), arr) for i, arr in enumerate(dt)]) 
    
    elif isinstance(dt, pd.DataFrame) and len(dt.columns)>0:
        # Convert column headers to str as parquet cannot save non-str column headers
        columns_in = copy.deepcopy(dt.columns)
        if not isinstance(dt.columns, pd.MultiIndex) and type(dt.columns[0]) != str:
            dt.columns = [str(column) for column in columns_in]
        
        # Add "_duplicate{i} to double column headers as parquet cannot save or load double columns 
        # (will be removed again during loading)" 
        if len(dt.columns)!=len(np.unique(dt.columns)):
            cols = pd.Series(dt.columns)
            for i, dup in enumerate(cols[cols.duplicated()]):  # Get unique duplicated names
                cols[i] = f"{dup}_duplicate{i}"
            dt.columns = cols

        #save table with parquet for fast loading later on
        path = path+'.pqt'
        dt.to_parquet(path, engine='pyarrow', index=True)
        dt.columns = columns_in

        #save path to data in dict for large amounts of data, save full table for small amounts
    else:
        dt = None

    if return_path:
        return path
    else:
        return dt


def load_dt(path: str, load_range: np.ndarray = None):
    """Saves a given data frame fast and efficient using parquet

    Args:
        path (str): Where to save 

    Returns:
        Data table after laoding
    """

    if path is None:
        return None

    if path.endswith('.npy') and load_range is not None:
        # Load pointer
        mmap_array = np.load(path, mmap_mode='r')
        # Get specific range
        if len(load_range) ==2 and load_range[1]-load_range[0]> 1:
            tab = mmap_array[load_range[0]:load_range[1]+1] 
        else:
            tab = mmap_array[load_range] 
    elif path.endswith('.pqt') and load_range is not None:
        #this obviously still loads the whole table and is only an intermediary solution
        if len(load_range) ==2 and load_range[1]-load_range[0]> 1:
            tab=pd.read_parquet(path, engine='pyarrow').iloc[load_range[0]:load_range[1]+1] 
        else:  
            tab=pd.read_parquet(path, engine='pyarrow').iloc[load_range]
    
    elif load_range is not None:

        with open(path, 'rb') as file:
            loaded = np.load(file, allow_pickle=True)
            tab = tuple(loaded[f'arr_{i}'][1] for i in range(len(loaded.files)))
            if len(load_range) ==2 and load_range[1]-load_range[0]> 1:
                tab1=tab[0][load_range[0]:load_range[1]+1] 
                tab2=tab[1][load_range[0]:load_range[1]+1] 
            else:
                tab1=tab[0][load_range]
                tab2=tab[1][load_range]
            tab=(tab1,tab2)

    elif path.endswith('.npy'):
        with open(path, 'rb') as file:
            # Load the array using pickle
            tab = np.lib.format.read_array(file)

    elif path.endswith('.pqt'):
        tab = pd.read_parquet(path, engine='pyarrow')

    elif path.endswith('.npz'):
        with open(path, 'rb') as file:
            loaded = np.load(file, allow_pickle=True)
            # Reconstruct the tuple of NumPy arrays
            tab = tuple(loaded[f'arr_{i}'][1] for i in range(len(loaded.files)))

    else:
        tab = None


    #special case if the columns may be retrieved from .pqt files
    if hasattr(tab, 'columns') and not isinstance(tab.columns , pd.MultiIndex):
        #remove column duplication caps
        cols_lit = tab.columns 
        cols_lit = cols_lit.str.replace(r'_duplicate\d+', '', regex=True)

        #restore tuple columns
        cols_lit=[
            ast.literal_eval(item)
            if type(item) == str
            and item.startswith("(")
            and item.endswith(")")
            else item 
            for item 
            in cols_lit
            ]
        
        #necessary due to legacy shenanigans with things being sometimes multi indices and sometimes not
        is_multi_index=False
        if isinstance(cols_lit[0], tuple) and cols_lit[0][1]=='x':
            is_multi_index=True

        tab.columns=pd.Index(cols_lit,tupleize_cols=is_multi_index)

    
    return tab


def load_dt_metainfo(path: str, load_index=True):
    """Loads the columns of data frame given as path

    Args:
        path (str): path to file
        load_index (bool): Also load index column and extract some additional information

    Returns:
        Dictionary of columns
    """

    if path is None:
        return None
    
    meta_info=_init_metainfo()

    if path.endswith('.npy'):
        with open(path, 'rb') as file:
            version = np.lib.format.read_magic(file)
            shape, _, _ = np.lib.format._read_array_header(file, version)

            meta_info['shape'] = shape
            if len (shape)==2:
                meta_info['num_rows'] = shape[0]
                meta_info['num_cols'] = shape[1]

    elif path.endswith('.pqt'):

        #read columns from metadata
        info_meta = pq.read_metadata(path)
        columns = [field.name for field in info_meta.schema if not field.name == '__index_level_0__']

        #remove column duplication caps
        columns = pd.Index(columns).str.replace(r'_duplicate\d+', '', regex=True)

        #adjust columns
        columns=[
            ast.literal_eval(item)
            if type(item) == str
            and item.startswith("(")
            and item.endswith(")")
            else item 
            for item 
            in columns
            ]
        
        if load_index:
            index_column = pq.read_table(path, columns=['__index_level_0__'])
            meta_info['index_column'] = pd.Index(index_column[0][:])
            meta_info['start_time'] = str(index_column[0][0])
            meta_info['end_time'] = str(index_column[0][-1])
        
        meta_info['columns'] = columns
        meta_info['num_cols'] = len(columns)
        meta_info['num_rows'] = info_meta.num_rows
        meta_info['shape'] = (info_meta.num_rows, len(columns))

    elif path.endswith('.npz'):

        with np.load(path, allow_pickle=True) as npz_file:

            first_array_name = npz_file.files[0]
            # Get the array without loading it into memory
            shape = npz_file[first_array_name][1].shape

            meta_info['shape'] = shape
            if len (shape)==2:
                meta_info['num_rows'] = shape[0]
                meta_info['num_cols'] = shape[1]

            
    return meta_info

    
def get_metainfo_from_loaded_dt(table: Union[np.ndarray,pd.DataFrame], load_index=True):
    """Extracts the columns of a given data frame

    Args:
        table (pd.DataFrame): dataFrame to extract meta info from
        load_index (bool): Also load index column and extract some additional information

    Returns:
        Dictionary of columns
    """

    if table is None:
        return None
    
    meta_info=_init_metainfo()

    if isinstance(table, np.ndarray):

        meta_info['shape'] = table.shape
        if len(table.shape)==2:
            meta_info['num_rows'] = table.shape[0]
            meta_info['num_cols'] = table.shape[1]

    elif isinstance(table, pd.DataFrame):

        if load_index:
            meta_info['index_column'] = table.index
            meta_info['start_time'] = table.index[0]
            meta_info['end_time'] = table.index[-1]
        
        meta_info['columns'] = list(table.columns)
        meta_info['num_cols'] = len(meta_info['columns'])
        meta_info['num_rows'] = table.shape[0]
        meta_info['shape'] = (table.shape[0], len(meta_info['columns']))

    elif (isinstance(table, Tuple) and all(isinstance(dt_subset, np.ndarray) for dt_subset in table)):

        meta_info['shape'] = table[0].shape
        if len(table[0].shape)==2:
            meta_info['num_rows'] = table[0].shape[0]
            meta_info['num_cols'] = table[0].shape[1]


    return meta_info

def _init_metainfo():
    meta_info={}

    meta_info['index_column'] = None
    meta_info['start_time'] = None
    meta_info['end_time'] = None
    meta_info['columns'] = None
    meta_info['num_cols'] = None
    meta_info['num_rows'] = None
    meta_info['shape'] = None

    return meta_info