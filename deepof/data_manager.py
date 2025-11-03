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
            # Extract minimal time metadata (n_rows, end_ns) from index if possible
            time_meta = self._extract_time_meta_from_index(data.index)

            df = self._prepare_dataframe(data)  # numeric-only table
            self.conn.register("df_temp", df)
            try:
                self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_temp')
            except Exception:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                self.conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_temp')

            if time_meta is not None:
                n_rows, end_ns = time_meta
                self._ensure_time_meta_table()
                self.conn.execute(
                    'INSERT OR REPLACE INTO "_dm_time_meta" (table_name, n_rows, end_ns) VALUES (?, ?, ?)',
                    [table_name, int(n_rows), int(end_ns)]
                )
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

        # Parse MultiIndex-like string columns back to tuples if applicable
        df = self._parse_columns_to_tuples(df)

        # Reconstruct index if requested (no fallbacks — raise on any mismatch)
        if load_index:
            meta = self._get_time_meta(table_name)
            assert meta is not None, f"Time index metadata missing for table '{table_name}'."
            end_ns = int(meta["end_ns"])
            total_rows = int(meta["n_rows"])

            n_loaded = len(df)
            if n_loaded == 0:
                df.index = pd.Index([], dtype=object)
            else:
                if load_range is None:
                    load_range = [0, total_rows - 1]

                if isinstance(load_range, (list, np.ndarray)):
                    arr = np.asarray(load_range)

                    if arr.ndim == 1 and arr.size == 2:
                        # Contiguous [a, b] using 0-based positions (like iloc but closed intervals)
                        a = int(arr[0])
                        b = int(arr[1])
                        expected = b - a + 1
                        assert (b > a) and (n_loaded == expected), "Given load range is invalid!"

                        index_strings = self._make_legacy_index_strings_from_end(
                            end_ns=end_ns,
                            total_rows=total_rows,
                            start_offset=a,
                            length=expected
                        )
                        df.index = pd.Index(index_strings)

                    else:
                        # Arbitrary positions (e.g., [0, 4, 6])
                        pos = arr.astype(int).ravel()
                        assert pos.size == n_loaded and pos.min() >= 0 and pos.max() < total_rows, "Given load range is invalid!"

                        full = pd.timedelta_range(
                            "00:00:00",
                            pd.to_timedelta(int(end_ns), unit="ns"),
                            periods=total_rows + 1,
                            closed="left",
                        )
                        s = pd.Series(full.astype(str), copy=False).str.replace(r"^0 days ", "", regex=True)
                        df.index = pd.Index(s.iloc[pos].to_numpy(dtype=object))

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

        # Columns and basic shape
        raw_cols = self._get_table_columns(table_name)
        column_names = [row[1] for row in raw_cols]
        parsed_columns = [parse_col(name) for name in column_names]
        num_rows = self.conn.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]

        meta = {
            "columns": parsed_columns,
            "num_cols": len(parsed_columns),
            "num_rows": num_rows,
            "shape": (num_rows, len(parsed_columns)),
        }

        if load_index:
            time_meta = self._get_time_meta(table_name)
            assert time_meta is not None, f"Time index metadata missing for table '{table_name}'."
            n_rows_meta = int(time_meta["n_rows"])
            end_ns = int(time_meta["end_ns"])
            assert n_rows_meta == num_rows, (
                f"n_rows mismatch between table and time metadata for '{table_name}' "
                f"(table {num_rows}, meta {n_rows_meta})."
            )

            # Reconstruct full legacy index (HH:MM:SS.SSSSSSSSS)
            idx_strings = self._make_legacy_index_strings_from_end(
                end_ns=end_ns,
                total_rows=n_rows_meta,
                start_offset=0,
                length=n_rows_meta
            )
            meta["index_column"] = pd.Index(idx_strings)
            meta["start_time"] = idx_strings[0] if n_rows_meta > 0 else None
            meta["end_time"] = idx_strings[-1] if n_rows_meta > 0 else None
        else:
            meta["index_column"] = None

        return meta

    def _get_metadata_blob(self, table_name: str, load_index: bool) -> dict:
        try:
            meta_json = self.conn.execute(f'SELECT metadata FROM "{table_name}"').fetchone()[0]
            meta_dict = json.loads(meta_json)

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

        # Keep tables numeric-only: drop the index into a RangeIndex
        return df_copy.reset_index(drop=True)

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
                        parsed_cols.append((parts[0], parts[1]))
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

    # ---------- Minimal helpers ----------

    def _ensure_time_meta_table(self):
        self.conn.execute(
            'CREATE TABLE IF NOT EXISTS "_dm_time_meta" ('
            'table_name TEXT PRIMARY KEY, '
            'n_rows BIGINT, '
            'end_ns BIGINT)'
        )

    def _get_time_meta(self, table_name: str) -> Union[dict, None]:
        row = self.conn.execute(
            'SELECT n_rows, end_ns FROM "_dm_time_meta" WHERE table_name = ?',
            [table_name]
        ).fetchone()
        if row is None:
            return None
        return {"n_rows": int(row[0]), "end_ns": int(row[1])}

    def _extract_time_meta_from_index(self, index: pd.Index) -> Union[Tuple[int, int], None]:
        try:
            if isinstance(index, pd.TimedeltaIndex):
                td = index
            else:
                td = pd.to_timedelta(index, errors="coerce")
                if getattr(td, "isna", None) is not None and np.any(td.isna()):
                    return None
                if not isinstance(td, pd.TimedeltaIndex):
                    td = pd.to_timedelta(td)

            n = len(td)
            if n == 0:
                return None
            ns = td.to_numpy(dtype="timedelta64[ns]").astype("int64")
            if n == 1:
                end_ns = 0
            else:
                # Invert end boundary E from last value L: L ≈ floor((n-1)/n * E)
                # => E ≈ round(L * n / (n - 1))
                last_ns = int(ns[-1])
                end_ns = int(round(last_ns * n / (n - 1)))
            return (n, end_ns)
        except Exception:
            return None

    def _make_legacy_index_strings_from_end(self, end_ns: int, total_rows: int, start_offset: int, length: int) -> np.ndarray:
        if length <= 0:
            return np.array([], dtype=object)
        full = pd.timedelta_range(
            "00:00:00",
            pd.to_timedelta(int(end_ns), unit="ns"),
            periods=int(total_rows) + 1,
            closed="left",
        )
        s = pd.Series(full.astype(str), copy=False)
        s = s.str.replace(r"^0 days ", "", regex=True)
        return s.iloc[start_offset:start_offset + length].to_numpy(dtype=object)