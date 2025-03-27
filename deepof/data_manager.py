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
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = db.connect(db_path)

    def close(self):
        self.conn.close()

    def save(self, key: str, data: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]]):
        table_name = sanitize_table_name(key)
        is_blob = isinstance(data, (np.ndarray, tuple))

        if (table_name,) in self.conn.execute("SHOW TABLES").fetchall():
            self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        if is_blob:
            buffer = io.BytesIO()
            pickle.dump(data, buffer)
            self.conn.execute(f'CREATE TABLE "{table_name}" (data BLOB)')
            self.conn.execute(f'INSERT INTO "{table_name}" VALUES (?)', [buffer.getvalue()])
        elif isinstance(data, pd.DataFrame):
            df = self._prepare_dataframe(data)
            self.conn.register("df_temp", df)
            self.conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_temp')
        else:
            raise TypeError("Unsupported data type")

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
            cols = self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            return len(cols) == 1 and cols[0][1] == "data"

        if only_metainfo:
            return self._get_metadata(table_name, load_index)

        if _is_blob():
            df = self.conn.execute(f'SELECT data FROM "{table_name}"').fetchdf()
            deserialized = pickle.loads(df.iloc[0]["data"])
            return (deserialized, {"duckdb_file": self.db_path, "table": table_name}) if return_path else deserialized

        query = self._build_query(table_name, load_range)
        df = self.conn.execute(query).fetchdf()
        df = self._parse_columns_to_tuples(df)
        df = self._restore_index(df)

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
            f'"{row[1]}"' for row in self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
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

        raw_cols = self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [parse_col(row[1]) for row in raw_cols]
        num_rows = self.conn.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]

        meta = {
            "columns": columns,
            "num_cols": len(columns),
            "num_rows": num_rows,
            "shape": (num_rows, len(columns)),
        }

        if load_index:
            if "index" in columns:
                try:
                    idx = self.conn.execute(f'SELECT "index" FROM "{table_name}"').fetchdf()
                    meta["index_column"] = idx.iloc[:, 0]
                    meta["start_time"] = str(idx.iloc[0, 0])
                    meta["end_time"] = str(idx.iloc[-1, 0])
                except:
                    meta["index_column"] = None
            else:
                meta["index_column"] = None

        return meta

    def _parse_columns_to_tuples(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed_cols = []
        is_multiindex = False

        for col in df.columns:
            if isinstance(col, tuple):
                parsed_cols.append(col)
                is_multiindex = True
            elif isinstance(col, str) and col.startswith("("):
                try:
                    parsed = ast.literal_eval(col)
                    if isinstance(parsed, tuple):
                        parsed_cols.append(parsed)
                        is_multiindex = True
                    else:
                        parsed_cols.append(col)
                except:
                    parsed_cols.append(col)
            else:
                parsed_cols.append(col)

        if is_multiindex:
            df.columns = pd.MultiIndex.from_tuples(parsed_cols)
        else:
            df.columns = parsed_cols
        return df
