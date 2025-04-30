import os
import io
import ast
import copy
import pickle
import warnings
import duckdb as db
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union
from functools import lru_cache
from pathlib import Path
import json
import re



def sanitize_table_name(table_name: str) -> str:
    import re
    if table_name[0].isdigit():
        table_name = f"t_{table_name}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", table_name)

def suppress_warnings(warn_messages):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as caught_warnings:
                for msg in warn_messages:
                    warnings.filterwarnings("ignore", message=f".*{msg}.*")
                result = fn(*args, **kwargs)
            for caught in caught_warnings:
                warnings.warn(caught.message)
            return result
        return wrapper
    return decorator


class DataManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, db_path: str):
        path = Path(db_path).resolve()
        self.db_path = str(path)
        self.save_dir = path.parent if db_path else Path(".").resolve()
        self.conn = db.connect(self.db_path)

    def close(self):
        self.conn.close()
    
    def save(self, key: str, data: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, ...]]):
        sanitized_key = sanitize_table_name(key)
        file_ext = None


        if isinstance(data, (np.ndarray, tuple)):
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_files (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    shape TEXT,
                    dtype TEXT,
                    num_arrays INTEGER,
                    num_rows INTEGER,
                    num_cols INTEGER
                )
            """)

            table_name = "dataset_files"
            save_dir = self.save_dir
            os.makedirs(save_dir, exist_ok=True)
            num_rows, num_cols = None, None          


            if isinstance(data, np.ndarray):
                arrays = [data]
                file_ext = "npy"
                
            elif isinstance(data, tuple) and all(isinstance(arr, np.ndarray) for arr in data):
                arrays = list(data)
                file_ext = "npz"
            else:
                raise TypeError("Tuples must only contain NumPy arrays")
            new_key = f"{sanitized_key}__{file_ext}"

            # Use base key and store the format internally
            save_path = os.path.join(save_dir, f"{sanitized_key}.{file_ext}")

            if file_ext == "npy":
                with open(save_path, 'wb') as file:
                    np.lib.format.write_array(file, arrays[0])
            else:
                with open(save_path, 'wb') as file:
                    np.savez(file, *[('arr_{}'.format(i), arr) for i, arr in enumerate(arrays)])

            shape = arrays[0].shape
            if len(shape) >= 2:
                num_rows = shape[0]
                num_cols = shape[1]

            meta_info = {
                "shape": shape ,
                "dtypes": [str(arr.dtype) for arr in arrays],
                "num_arrays": len(arrays),
                "num_rows": num_rows,
                "num_cols": num_cols,
            }

            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} (key, file_path, shape, dtype, num_arrays, num_rows, num_cols)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_key,
                    save_path,
                    json.dumps(meta_info["shape"]),
                    json.dumps(meta_info["dtypes"]),
                    meta_info["num_arrays"],
                    meta_info["num_rows"],
                    meta_info["num_cols"]
                )
            )

        elif isinstance(data, pd.DataFrame):
            df = self._prepare_dataframe(data)
            self.conn.register("df_temp", df)
            self.conn.execute(f'CREATE OR REPLACE TABLE "{sanitized_key}" AS SELECT * FROM df_temp')

        else:
            raise TypeError(f"Unsupported data type for save(): {type(data)}")

        self._get_table_columns.cache_clear()
        return file_ext

    def load(self,
         key: str,
         return_path: bool = False,
         only_metainfo: bool = False,
         load_index: bool = False,
         load_range: np.ndarray = None,
         filetype : str = None):

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"{self.db_path} does not exist")

        sanitized_key = sanitize_table_name(key)
        tab = None

        if filetype:
            new_key = f"{sanitized_key}__{filetype}"

        # Check if dataset_files table exists before querying
            table_exists = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dataset_files'"
            ).fetchone()[0] > 0

            result = None
            if table_exists:
                result = self.conn.execute(
                        "SELECT file_path, shape, dtype, num_arrays, num_rows, num_cols FROM dataset_files WHERE key = ?",
                        (new_key,)
                    ).fetchone()
                if result is None:
                    raise KeyError(f"No data found for key '{new_key}' in dataset_files")

                file_path, shape_json, dtype_json, num_arrays, num_rows, num_cols = result

                if only_metainfo:
                    return {
                            "file_path": file_path,
                            "shape": json.loads(shape_json),
                            "dtype": json.loads(dtype_json),
                            "num_arrays": num_arrays,
                            "num_rows": num_rows,
                            "num_cols": num_cols
                        }

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file {file_path} not found on disk")

                if file_path.endswith(".npy"):
                    tab = np.load(file_path, mmap_mode='r') if load_range is not None else np.load(file_path)
                    if load_range is not None:
                        if isinstance(load_range, (list, np.ndarray)) and len(load_range) == 2:
                            load_range = slice(load_range[0], load_range[1] + 1)
                        tab = tab[load_range]
                

                elif file_path.endswith(".npz"):
                    
                    with open(file_path, 'rb') as file:
                        loaded = np.load(file, allow_pickle=True)

                        # Reconstruct the tuple of NumPy arrays (assuming each entry is a (name, array) tuple)
                        arrays = tuple(loaded[f'arr_{i}'][1] for i in range(len(loaded.files)))

                        if load_range is not None:
                            # Handle slicing logic robustly
                            if isinstance(load_range, (list, np.ndarray)) and len(load_range) == 2 and load_range[1] - load_range[0] > 1:
                                tab1 = arrays[0][load_range[0]:load_range[1] + 1]
                                tab2 = arrays[1][load_range[0]:load_range[1] + 1]
                            else:
                                tab1 = arrays[0][load_range]
                                tab2 = arrays[1][load_range]
                            tab = (tab1, tab2)
                        else:
                            tab = arrays

                    #with np.load(file_path, allow_pickle=True) as loaded:
                     #   arrays = tuple(loaded[f'arr_{i}'] for i in range(len(loaded.files)))
                      #  if load_range is not None:
                       #     if isinstance(load_range, (list, np.ndarray)) and len(load_range) == 2:
                        #        load_range = slice(load_range[0], load_range[1] + 1)
                         #   arrays = tuple(arr[load_range] for arr in arrays)
                        #tab = arrays[0] if len(arrays) == 1 else arrays
        else :
            if only_metainfo:
                return self._get_metadata(sanitized_key, load_index)

            query = self._build_query(sanitized_key, load_range)
            arrow_table = self.conn.execute(query).fetch_arrow_table()
            tab = arrow_table.to_pandas(
                split_blocks=True,
                self_destruct=True,
            )

            data_part = tab.iloc[:, 1:]  
            tab = self._parse_columns_to_tuples(data_part)

        return (tab, {"duckdb_file": self.db_path, "table": sanitized_key, "datatype":filetype}) if return_path else tab


    def _get_metadata(self, table_name: str, load_index: bool) -> dict:
        

        def parse_col(c):
            tuple_pattern = re.compile(r'^\(([^()]+,[^()]+)\)$')
            if isinstance(c, str) and tuple_pattern.match(c):
                try:
                    parts = [x.strip().strip("'").strip('"') for x in tuple_pattern.match(c).group(1).split(",")]
                    return tuple(parts)
                except Exception:
                    return c
            return c
        


        raw_cols = self._get_table_columns(table_name)
        column_names = [row[1] for row in raw_cols]    
        
        parsed_columns = [parse_col(name) for name in column_names]
        parsed_columns_no_index = parsed_columns[1:]
        num_rows = self.conn.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]

        meta = {
            "columns": parsed_columns_no_index,
            "num_cols": len(parsed_columns_no_index),
            "num_rows": num_rows,
            "shape": (num_rows, len(parsed_columns_no_index)),
        }
        
        if load_index:
            try:
                idx_df = self.conn.execute(f'SELECT "{column_names[0]}" FROM "{table_name}"').fetchdf()
                meta["index_column"] = idx_df.iloc[:, 0]
                meta["start_time"] = str(idx_df.iloc[0, 0])
                meta["end_time"] = str(idx_df.iloc[-1, 0])
            except Exception:
                meta["index_column"] = None
        else:
            meta["index_column"] = None

        return meta

    def _get_metadata_blob(self, table_name: str, load_index: bool) -> dict:
       
        try:
            meta_json = self.conn.execute(f'SELECT metadata FROM "{table_name}"').fetchone()[0]
            meta_dict = json.loads(meta_json)

            # Taking the first shape to make it similar like original behavior
            first_shape = meta_dict["shapes"][0] if meta_dict["shapes"] else (0,)

            return {
                "columns": [],
                "num_cols": np.prod(first_shape[1:]) if len(first_shape) > 1 else 1,
                "num_rows": first_shape[0] if len(first_shape) > 0 else 0,
                "shape": first_shape,
                "index_column": None,
            }

           

            #return {
             #       "columns": [],
              #      "num_cols": sum(
               #         np.prod(shape[1:]) if len(shape) > 1 else 1
                #        for shape in meta_dict["shapes"]
                 #   ),
                  #  "num_rows": meta_dict["shapes"][0][0] if meta_dict["shapes"] else 0,
                   # "shape": meta_dict["shapes"] if meta_dict["num_arrays"] > 1 else meta_dict["shapes"][0],
                   # "index_column": None,
                #}


        except Exception as e:
                return {
                    "columns": [],
                    "num_cols": 0,
                    "num_rows": 0,
                    "shape": (),
                    "index_column": None,
                    "error": f"Failed to load blob: {str(e)}"
                }

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if not isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = [str(col) for col in df_copy.columns]

        if len(df_copy.columns) != len(set(df_copy.columns)):
            cols = pd.Series(df_copy.columns)
            for i, dup in enumerate(cols[cols.duplicated()]):
                cols[i] = f"{dup}_duplicate{i}"
            df_copy.columns = cols

        if isinstance(df_copy.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
            df_copy.index = df_copy.index.astype(str)

        return df_copy.reset_index(names="index")

    def _build_query(self, table_name: str, load_range: np.ndarray) -> str:
        quoted_cols = ", ".join(
            f'"{col[1]}"' for col in self._get_table_columns(table_name)
        )
        if load_range is None:
            return f'SELECT {quoted_cols} FROM "{table_name}"'
        elif len(load_range) == 2:
            return f'SELECT {quoted_cols} FROM "{table_name}" WHERE rowid BETWEEN {load_range[0]} AND {load_range[1]}'
        else:
            values = ", ".join(map(str, load_range))
            return f'SELECT {quoted_cols} FROM "{table_name}" WHERE rowid IN ({values})'

    def _restore_index(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col == "index" or (isinstance(col, tuple) and "index" in col):
                try:
                    df.index = pd.to_timedelta(df[col], errors="coerce")
                    df.drop(columns=[col], inplace=True)
                    break
                except Exception:
                    continue
        return df    
    

    
    @lru_cache(maxsize=128)
    def _get_table_columns(self, table_name: str) -> Tuple[Tuple]:
        return tuple(self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall())
    
    def _parse_columns_to_tuples(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed_cols = []
        axis_labels = {"x", "y"}
        tuple_pattern = re.compile(r'^\(([^)]+)\)$')
        for col in df.columns:
            if isinstance(col, tuple):
                parsed_cols.append(col)
            elif isinstance(col, str):
                match = tuple_pattern.match(col)
                if match:
                    parts = [x.strip().strip("'") for x in match.group(1).split(",")]
                    if len(parts) == 2:
                        parsed_cols.append((parts[0], parts[1]))  # <--- treat both as str
                        continue
                parsed_cols.append(col)
            else:
                parsed_cols.append(col)

        tuple_cols = [col for col in parsed_cols if isinstance(col, tuple)]
        is_multiindex = (
            tuple_cols and
            all(len(col) == 2 and str(col[1]) in axis_labels for col in tuple_cols)
        )

        if is_multiindex:
            df.columns = pd.MultiIndex.from_tuples(parsed_cols)
        else:
            df.columns = parsed_cols

        return df


    