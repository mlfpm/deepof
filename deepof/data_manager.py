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
        self.db_path = str(Path(db_path).resolve())
        self.conn = db.connect(self.db_path)

    def close(self):
        self.conn.close()

    def save(self, key: str, data: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]]):
        table_name = sanitize_table_name(key)
        filetype, is_blob = ("npz", True) if isinstance(data, tuple) else ("npy", True) if isinstance(data, np.ndarray) else (None, False)

        
               
       

        if is_blob:
            table_name = f"{table_name}__{filetype}"
            buffer = io.BytesIO()
            arrays = data if isinstance(data, tuple) else (data,)
            np.savez_compressed(buffer, *arrays)

            meta = {
                "shapes": [arr.shape for arr in arrays],
                "dtypes": [str(arr.dtype) for arr in arrays],
                "num_arrays": len(arrays),
            }
            meta_json = json.dumps(meta)

            try:
                self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT ? AS data , ? AS metadata', [buffer.getvalue(), meta_json])
            except Exception:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                self.conn.execute(f'CREATE TABLE "{table_name}" (data BLOB, metadata TEXT)')
                self.conn.execute(
                    f'INSERT INTO "{table_name}" VALUES (?, ?)',
                    [buffer.getvalue(), meta_json]
                )
                
        elif isinstance(data, pd.DataFrame):
            df = self._prepare_dataframe(data)
            self.conn.register("df_temp", df)
            try:
                self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_temp')
            except Exception:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                self.conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_temp')
            
        else:
            raise TypeError("Unsupported data type")
        self._get_table_columns.cache_clear()
        return table_name


    def load(self,
             key: str,      
             return_path: bool = False,
             only_metainfo: bool = False,
             load_index: bool = False,
             load_range: np.ndarray = None):
        table_name = sanitize_table_name(key)
        

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"{self.db_path} does not exist")
        
        def _is_blob():
            cols = {col[1] for col in self._get_table_columns(table_name)}
            return "data" in cols and "metadata" in cols and len(cols) == 2
        
       
        if _is_blob():
            if only_metainfo:
                return self._get_metadata_blob(table_name, load_index)
            blob = self.conn.execute(f'SELECT data FROM "{table_name}"').fetchone()[0]

            with np.load(io.BytesIO(blob)) as loaded:
                if isinstance(load_range, (list, np.ndarray)):
                    arr_load_range = np.array(load_range)
                    if (
                        arr_load_range.ndim == 1
                        and len(arr_load_range) == 2
                        and np.issubdtype(arr_load_range.dtype, np.integer)
                    ):
                        load_range = slice(arr_load_range[0], arr_load_range[1] + 1)

                if len(loaded.files) == 1:
                    arr = loaded[loaded.files[0]]
                    deserialized = arr[load_range] if load_range is not None else arr
                else:
                    deserialized = tuple(
                        loaded[key][load_range] if load_range is not None else loaded[key]
                        for key in loaded.files
                    )

           
            return (deserialized, {"duckdb_file": self.db_path, "table": table_name}) if return_path else deserialized
        if only_metainfo:
            return self._get_metadata(table_name, load_index)
        query = self._build_query(table_name, load_range)
        arrow_table = self.conn.execute(query).fetch_arrow_table()
        df = arrow_table.to_pandas(
            split_blocks=True,             
            self_destruct=True,      
        )

        #index_part = df.iloc[:, :1]
        data_part = df.iloc[:, 1:]
        df = self._parse_columns_to_tuples(data_part)
        return (df, {"duckdb_file": self.db_path, "table": table_name}) if return_path else df


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


    