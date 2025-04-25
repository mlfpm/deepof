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
        self.db_path = db_path
        self.conn = db.connect(db_path)

    def close(self):
        self.conn.close()

    def save(self, key: str, data: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]]):
        table_name = sanitize_table_name(key)
        is_blob = isinstance(data, (np.ndarray, tuple))

        if is_blob:
            buffer = io.BytesIO()
            arrays = data if isinstance(data, tuple) else (data,)
            np.savez_compressed(buffer, *arrays)

            try:
                self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT ? AS data', [buffer.getvalue()])
            except Exception:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                self.conn.execute(f'CREATE TABLE "{table_name}" (data BLOB)')
                self.conn.execute(f'INSERT INTO "{table_name}" VALUES (?)', [buffer.getvalue()])
        
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
            cols = self._get_table_columns(table_name)
            return len(cols) == 1 and cols[0][1] == "data"
        if only_metainfo:
            return self._get_metadata(table_name, load_index)
        if _is_blob():
            df = self.conn.execute(f'SELECT data FROM "{table_name}"').fetchdf()
            blob = df.iloc[0]["data"]

            with np.load(io.BytesIO(blob)) as loaded:
                arrays = [loaded[key] for key in loaded.files]
                deserialized = tuple(arrays) if len(arrays) > 1 else arrays[0]
                if load_range is not None:   
                    if isinstance(load_range, list) and len(load_range) == 2:
                        load_range = slice(load_range[0], load_range[1] + 1)

                    if isinstance(deserialized, tuple):
                        deserialized = tuple(arr[load_range] for arr in deserialized)
                    else:
                        deserialized = deserialized[load_range]

              
                    

                #deserialized = deserialized[:, 1:]
                return (deserialized, {"duckdb_file": self.db_path, "table": table_name}) if return_path else deserialized

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
    

    def _get_metadata(self, table_name: str, load_index: bool) -> dict:
        def parse_col(c):
            try:
                return ast.literal_eval(c)
            except:
                return c

        
        raw_cols = self._get_table_columns(table_name)
        column_names = [row[1] for row in raw_cols]

     
        is_blob = len(column_names) == 1 and column_names[0] == "data"
        if is_blob:
            try:
                df = self.conn.execute(f'SELECT data FROM "{table_name}"').fetchdf()
                blob = df.iloc[0]["data"]


                with np.load(io.BytesIO(blob)) as loaded:
                    arrays = [loaded[key] for key in loaded.files]
                    deserialized = arrays[0]
                    

                if isinstance(deserialized, tuple):
                    try:
                        if all(arr.ndim == 2 and arr.shape[0] == deserialized[0].shape[0] for arr in deserialized):
                            stacked = np.concatenate(deserialized, axis=1)
                            shape = stacked.shape
                            num_rows = shape[0]
                            num_cols = shape[1]
                        else:
                            raise ValueError("Cannot concatenate arrays with incompatible shapes.")
                    except Exception as e:
                        shape = tuple(arr.shape for arr in deserialized)
                        num_rows = deserialized[0].shape[0] if deserialized[0].ndim > 0 else 1
                        num_cols = sum(
                            arr.shape[1] * arr.shape[2] if arr.ndim == 3 else
                            arr.shape[1] if arr.ndim == 2 else 1
                            for arr in deserialized
                        )
                
                elif isinstance(deserialized, np.ndarray):
                    shape = deserialized.shape
                    num_rows = shape[0]
                    num_cols = shape[1] if len(shape) > 1 else 1                    
                else:
                    shape = ()
                    num_rows = 0
                    num_cols = 0

                return {
                    "columns": [],
                    "num_cols": num_cols,
                    "num_rows": num_rows,
                    "shape": shape,
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

    
    @lru_cache(maxsize=128)
    def _get_table_columns(self, table_name: str) -> Tuple[Tuple]:
        return tuple(self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall())
    def _parse_columns_to_tuples(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed_cols = []

        axis_labels = {"x", "y"}

        for col in df.columns:
            if isinstance(col, tuple):
                parsed_cols.append(col)
            elif isinstance(col, str) and col.startswith("(") and col.endswith(")"):
                try:
                    parsed = ast.literal_eval(col)
                    if isinstance(parsed, tuple):
                        parsed_cols.append(parsed)
                    else:
                        parsed_cols.append(col)
                except Exception:
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


    