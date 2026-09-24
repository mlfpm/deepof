# @author NoCreativeIdeForGoodUsername
# encoding: utf-8
# module deepof.clustering

import os
import hashlib
import h5py
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import torch.distributed as dist
from typing import Dict, Tuple, Optional
from deepof.data_loading import get_dt
from tqdm import tqdm

_DEFAULT_SHUFFLE_SEED = 42
_SHUFFLE_COPY_RAM = 256 * 1024 ** 2  # ~256 MiB per permute chunk


def reorder_and_reshape(data: np.ndarray) -> np.ndarray:
    assert data.shape[2] % 3 == 0, "Error! Number of columns is not a multiple of 3 (x, y, speed)!"
    D = data.shape[2]
    N = D // 3

    x = data[:, :, 0:N]
    y = data[:, :, N:2*N]
    s = data[:, :, 2*N:3*N]

    out = np.stack([x, y, s], axis=-1)  # (B, W, N, 3)
    return out


class BatchDictDataset:
    def __init__(
        self,
        preprocessed_dict: Dict,
        dataset_folder: str,
        dataset_name: str,
        force_rebuild: bool = False,
        h5_chunk_len: Optional[int] = None,
        return_angles: Optional[bool] = False,
        supervised_dict: Optional[Dict] = None,
        read_only: bool = False,
        global_shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
    ):
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name
        self.return_angles = return_angles
        self.supervised_dict = supervised_dict
        self.read_only = read_only
        self.global_shuffle = bool(global_shuffle)
        self.shuffle_seed = None if shuffle_seed is None else int(shuffle_seed)

        # Determine if the dataset has angles
        self.has_angles = False
        if get_dt(preprocessed_dict, list(preprocessed_dict.keys())[0])[2].size > 0:
            self.has_angles = True

        self.X_path = os.path.join(dataset_folder, dataset_name + 'X_data.h5')
        self.a_path = os.path.join(dataset_folder, dataset_name + 'a_data.h5')
        self.ang_path = os.path.join(dataset_folder, dataset_name + 'ang_data.h5')
        self.y_path = os.path.join(dataset_folder, dataset_name + 'y_data.h5')
        self.idx_path = os.path.join(dataset_folder, dataset_name + 'video_idx.npy')

        self._init_hdf5(preprocessed_dict, force_rebuild=force_rebuild, h5_chunk_len=h5_chunk_len, read_only=read_only)

    def _effective_shuffle_seed(self) -> int:
        return _DEFAULT_SHUFFLE_SEED if self.shuffle_seed is None else int(self.shuffle_seed)

    def _does_need_build(self, preprocessed_dict: Dict) -> Tuple[bool, str]:
        """Check if HDF5 dataset needs rebuild by comparing metadata."""
        # Quick file existence checks
        required = [self.X_path, self.a_path, self.idx_path]
        if self.has_angles:
            required.append(self.ang_path)
        if self.supervised_dict is not None:
            required.append(self.y_path)

        for fpath in required:
            if not os.path.exists(fpath):
                return True, f"Missing: {os.path.basename(fpath)}"

        # Compute expected metadata from preprocessed_dict
        keys = list(preprocessed_dict.keys())
        keys_hash = hashlib.md5(','.join(sorted(str(k) for k in keys)).encode()).hexdigest()

        X_first, a_first, ang_first = get_dt(preprocessed_dict, keys[0])
        expected_shapes = {
            'x': tuple(reorder_and_reshape(X_first[:1]).shape[1:]),
            'a': tuple(np.expand_dims(a_first[:1], -1).shape[1:]),
        }
        if self.has_angles and ang_first.size > 0:
            expected_shapes['ang'] = tuple(np.expand_dims(ang_first[:1], -1).shape[1:])
        if self.supervised_dict is not None:
            expected_shapes['y'] = tuple(self.supervised_dict[keys[0]][:1].shape[1:])

        try:
            # Check main X file and metadata
            with h5py.File(self.X_path, 'r') as f:
                if 'X' not in f:
                    return True, "Corrupted X_data.h5"
                if not f.attrs.get('build_complete', False):
                    return True, "Previous build incomplete"

                stored_hash = f.attrs.get('keys_hash', None)
                if stored_hash is not None and stored_hash != keys_hash:
                    return True, "Video keys changed"

                if tuple(f['X'].shape[1:]) != expected_shapes['x']:
                    return True, "X shape mismatch"
                n_samples = f['X'].shape[0]

                stored_shuffle = bool(f.attrs.get('global_shuffle', False))
                if stored_shuffle != bool(self.global_shuffle):
                    return True, "global_shuffle setting changed"

                if self.global_shuffle:
                    stored_seed = int(f.attrs.get('shuffle_seed', -1))
                    if stored_seed != self._effective_shuffle_seed():
                        return True, "shuffle_seed changed"

            # Check other HDF5 files
            checks = [('a', self.a_path, 'a')]
            if self.has_angles and 'ang' in expected_shapes:
                checks.append(('ang', self.ang_path, 'ang'))
            if self.supervised_dict is not None:
                checks.append(('y', self.y_path, 'y'))

            for key, path, ds_name in checks:
                with h5py.File(path, 'r') as f:
                    if ds_name not in f:
                        return True, f"Corrupted {os.path.basename(path)}"
                    if tuple(f[ds_name].shape[1:]) != expected_shapes[key]:
                        return True, f"{key.upper()} shape mismatch"

            # Check video index
            video_idx = np.load(self.idx_path)
            if len(np.unique(video_idx)) != len(keys):
                return True, "Video count mismatch"

            # Verify sample count if hash was missing (backward compat)
            if stored_hash is None:
                expected_n = sum(get_dt(preprocessed_dict, k)[0].shape[0] for k in keys)
                if n_samples != expected_n:
                    return True, "Sample count mismatch"

            return False, "Dataset up-to-date"

        except (OSError, KeyError) as e:
            return True, f"Error reading files: {e}"

    def _init_hdf5(self, preprocessed_dict: Dict, h5_chunk_len: Optional[int], force_rebuild: bool = False, read_only: bool = False):
        os.makedirs(self.dataset_folder, exist_ok=True)

        if not read_only:
            if force_rebuild:
                need_build, reason = True, "Force rebuild requested"
            else:
                need_build, reason = self._does_need_build(preprocessed_dict)

            if need_build:
                print(f"BatchDictDataset: building HDF5 at {self.dataset_folder}...")
                print(f"  Reason: {reason}")
                if self.global_shuffle:
                    print(
                        f"  global_shuffle=True (seed={self._effective_shuffle_seed()}); "
                        "build uses temporary files and may take a while."
                    )
                self._build_hdf5(preprocessed_dict, h5_chunk_len=h5_chunk_len)
            else:
                print(f"BatchDictDataset: reusing existing HDF5 at {self.dataset_folder}")
        else:

            print(f"BatchDictDataset: reusing existing HDF5 at {self.dataset_folder}")

        with h5py.File(self.X_path, 'r') as f:
            X_ds = f['X']
            self.x_shape = tuple(X_ds.shape[1:])
            self.length = int(X_ds.shape[0])
            if 'global_shuffle' in f.attrs:
                file_gs = bool(f.attrs.get('global_shuffle', False))
                if file_gs != self.global_shuffle:
                    print(
                        f"Warning: requested global_shuffle={self.global_shuffle} but "
                        f"HDF5 has global_shuffle={file_gs}. Using file value for loading."
                    )
                self.global_shuffle = file_gs
            if 'shuffle_seed' in f.attrs:
                stored = int(f.attrs.get('shuffle_seed', -1))
                if stored >= 0:
                    self.shuffle_seed = stored
        with h5py.File(self.a_path, 'r') as f:
            A_ds = f['a']
            self.a_shape = tuple(A_ds.shape[1:])
        if self.has_angles:
            with h5py.File(self.ang_path, 'r') as f:
                Ang_ds = f['ang']
                self.ang_shape = tuple(Ang_ds.shape[1:])
        else:
            Ang_ds = None
            self.ang_shape = None

        # Load Y shape
        self._h5_Y = None
        self._Y = None
        if os.path.exists(self.y_path) and self.supervised_dict is not None:
            with h5py.File(self.y_path, 'r') as f:
                Y_ds = f['y']
                self.y_shape = tuple(Y_ds.shape[1:])
        else:
            self.y_shape = None

        self._h5_X = None
        self._h5_A = None
        self._h5_Ang = None
        print(
            f"HDF5 dataset ready. Samples: {self.length}, x_shape: {self.x_shape}, "
            f"a_shape: {self.a_shape}, ang_shape: {self.ang_shape}, "
            f"global_shuffle: {self.global_shuffle}"
        )

    def _tmp_path(self, path: str) -> str:
        return path + '.tmp'

    def _cleanup_tmp_files(self):
        for p in (self.X_path, self.a_path, self.ang_path, self.y_path):
            tmp = self._tmp_path(p)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _mark_build_incomplete(self):
        """Ensure a crashed rebuild cannot be mistaken for a complete cached dataset."""
        if os.path.exists(self.X_path):
            try:
                with h5py.File(self.X_path, 'a') as f:
                    f.attrs['build_complete'] = False
            except Exception:
                pass

    def _write_h5_attrs(self, f_X, keys_hash: str, n_videos: int, total_samples: int, shuffled: bool, shuffle_seed: int):
        f_X.attrs['keys_hash'] = keys_hash
        f_X.attrs['n_videos'] = n_videos
        f_X.attrs['n_samples'] = total_samples
        f_X.attrs['global_shuffle'] = bool(shuffled)
        f_X.attrs['shuffle_seed'] = int(shuffle_seed) if shuffled else -1
        f_X.attrs['build_complete'] = True

    def _sequential_write(
        self,
        preprocessed_dict: Dict,
        keys,
        x_path: str,
        a_path: str,
        ang_path: str,
        y_path: Optional[str],
        total_samples: int,
        shapes_X,
        shapes_A,
        shapes_Ang,
        shapes_Y,
        h5_chunk_len: int,
        keys_hash: str,
        write_final_attrs: bool,
        shuffle_seed: int,
    ):
        f_X = h5py.File(x_path, 'w')
        f_A = h5py.File(a_path, 'w')
        f_Ang = h5py.File(ang_path, 'w')
        f_Y = h5py.File(y_path, 'w') if y_path is not None else None

        try:
            f_X.attrs['build_complete'] = False

            X_dset = f_X.create_dataset(
                'X', shape=(total_samples, *shapes_X), dtype='float32',
                chunks=(h5_chunk_len, *shapes_X), compression=None, shuffle=False, fletcher32=False,
                maxshape=(total_samples, *shapes_X),
            )
            A_dset = f_A.create_dataset(
                'a', shape=(total_samples, *shapes_A), dtype='float32',
                chunks=(h5_chunk_len, *shapes_A), compression=None, shuffle=False, fletcher32=False,
                maxshape=(total_samples, *shapes_A),
            )
            Ang_dset = None
            if self.has_angles:
                Ang_dset = f_Ang.create_dataset(
                    'ang', shape=(total_samples, *shapes_Ang), dtype='float32',
                    chunks=(h5_chunk_len, *shapes_Ang), compression=None, shuffle=False, fletcher32=False,
                    maxshape=(total_samples, *shapes_Ang),
                )

            Y_dset = None
            if f_Y is not None:
                Y_dset = f_Y.create_dataset(
                    'y', shape=(total_samples, *shapes_Y), dtype='float32',
                    chunks=(h5_chunk_len, *shapes_Y), compression=None, shuffle=False, fletcher32=False,
                    maxshape=(total_samples, *shapes_Y),
                )

            idx = 0
            for key in tqdm(keys, desc="BatchDictDataset: writing HDF5", unit="video"):
                X_batch, a_batch, ang_batch = get_dt(preprocessed_dict, key)
                n = int(X_batch.shape[0])
                X_re = reorder_and_reshape(X_batch).astype(np.float32, copy=False)
                A_re = np.expand_dims(a_batch, -1).astype(np.float32, copy=False)
                Ang_re = np.expand_dims(ang_batch, -1).astype(np.float32, copy=False)

                X_dset[idx:idx+n] = X_re
                A_dset[idx:idx+n] = A_re
                if self.has_angles and ang_batch.size > 0:
                    Ang_dset[idx:idx+n] = Ang_re

                if Y_dset is not None:
                    y_batch = self.supervised_dict[key]
                    assert y_batch.shape[0] == n, \
                        f"Shape mismatch for key {key}: X has {n} rows, Y has {y_batch.shape[0]}. Check windowing."

                    Y_re = y_batch.astype(np.float32, copy=False)
                    Y_dset[idx:idx+n] = Y_re

                idx += n

            if write_final_attrs:
                self._write_h5_attrs(
                    f_X,
                    keys_hash=keys_hash,
                    n_videos=len(keys),
                    total_samples=total_samples,
                    shuffled=False,
                    shuffle_seed=shuffle_seed,
                )
        finally:
            f_X.close()
            f_A.close()
            f_Ang.close()
            if f_Y is not None:
                f_Y.close()

    def _write_shuffled_from_sequential(
        self,
        src_x: str,
        src_a: str,
        src_ang: Optional[str],
        src_y: Optional[str],
        perm: np.ndarray,
        total_samples: int,
        shapes_X,
        shapes_A,
        shapes_Ang,
        shapes_Y,
        h5_chunk_len: int,
        keys_hash: str,
        n_videos: int,
        shuffle_seed: int,
    ):
        bytes_per = int(np.prod(shapes_X)) * 4 + int(np.prod(shapes_A)) * 4
        if self.has_angles and shapes_Ang is not None:
            bytes_per += int(np.prod(shapes_Ang)) * 4
        if src_y is not None and shapes_Y is not None:
            bytes_per += int(np.prod(shapes_Y)) * 4
        copy_chunk = max(1, min(total_samples, _SHUFFLE_COPY_RAM // max(int(bytes_per), 1)))

        rdcc_nbytes = _SHUFFLE_COPY_RAM
        rdcc_nslots = 1_000_000

        fx_s = h5py.File(src_x, 'r', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        fa_s = h5py.File(src_a, 'r', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        fang_s = (
            h5py.File(src_ang, 'r', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
            if (src_ang is not None and self.has_angles) else None
        )
        fy_s = (
            h5py.File(src_y, 'r', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
            if src_y is not None else None
        )

        fx_d = h5py.File(self.X_path, 'w')
        fa_d = h5py.File(self.a_path, 'w')
        fang_d = h5py.File(self.ang_path, 'w') if self.has_angles else None
        fy_d = h5py.File(self.y_path, 'w') if src_y is not None else None

        try:
            fx_d.attrs['build_complete'] = False

            Xd = fx_d.create_dataset(
                'X', shape=(total_samples, *shapes_X), dtype='float32',
                chunks=(h5_chunk_len, *shapes_X), compression=None, shuffle=False, fletcher32=False,
                maxshape=(total_samples, *shapes_X),
            )
            Ad = fa_d.create_dataset(
                'a', shape=(total_samples, *shapes_A), dtype='float32',
                chunks=(h5_chunk_len, *shapes_A), compression=None, shuffle=False, fletcher32=False,
                maxshape=(total_samples, *shapes_A),
            )
            Angd = None
            if fang_d is not None:
                Angd = fang_d.create_dataset(
                    'ang', shape=(total_samples, *shapes_Ang), dtype='float32',
                    chunks=(h5_chunk_len, *shapes_Ang), compression=None, shuffle=False, fletcher32=False,
                    maxshape=(total_samples, *shapes_Ang),
                )
            Yd = None
            if fy_d is not None:
                Yd = fy_d.create_dataset(
                    'y', shape=(total_samples, *shapes_Y), dtype='float32',
                    chunks=(h5_chunk_len, *shapes_Y), compression=None, shuffle=False, fletcher32=False,
                    maxshape=(total_samples, *shapes_Y),
                )

            Xs, As = fx_s['X'], fa_s['a']
            Angs = fang_s['ang'] if fang_s is not None else None
            Ys = fy_s['y'] if fy_s is not None else None

            n_chunks = (total_samples + copy_chunk - 1) // copy_chunk
            for s in tqdm(
                range(0, total_samples, copy_chunk),
                desc="BatchDictDataset: global shuffle",
                unit="chunk",
                total=n_chunks,
            ):
                e = min(s + copy_chunk, total_samples)
                src_idx = perm[s:e]
                # h5py integer fancy indexing requires monotonically increasing indices
                order = np.argsort(src_idx, kind='mergesort')
                sorted_idx = np.ascontiguousarray(src_idx[order], dtype=np.int64)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.shape[0])

                Xd[s:e] = np.asarray(Xs[sorted_idx])[inv]
                Ad[s:e] = np.asarray(As[sorted_idx])[inv]
                if Angs is not None:
                    Angd[s:e] = np.asarray(Angs[sorted_idx])[inv]
                if Ys is not None:
                    Yd[s:e] = np.asarray(Ys[sorted_idx])[inv]

            self._write_h5_attrs(
                fx_d,
                keys_hash=keys_hash,
                n_videos=n_videos,
                total_samples=total_samples,
                shuffled=True,
                shuffle_seed=shuffle_seed,
            )
        finally:
            fx_s.close()
            fa_s.close()
            fx_d.close()
            fa_d.close()
            if fang_s is not None:
                fang_s.close()
            if fang_d is not None:
                fang_d.close()
            if fy_s is not None:
                fy_s.close()
            if fy_d is not None:
                fy_d.close()

    def _build_hdf5(self, preprocessed_dict: Dict, h5_chunk_len: Optional[int]):
        keys = list(preprocessed_dict.keys())
        keys_hash = hashlib.md5(','.join(sorted(str(k) for k in keys)).encode()).hexdigest()
        shuffle_seed = self._effective_shuffle_seed()

        total_samples = 0
        shapes_X = None
        shapes_A = None
        shapes_Ang = None
        shapes_Y = None
        video_indices = []

        for i, key in enumerate(keys):
            X_batch, a_batch, ang_batch = get_dt(preprocessed_dict, key)
            if shapes_X is None:
                sample_X = reorder_and_reshape(X_batch[:1])
                sample_A = np.expand_dims(a_batch[:1], -1)
                sample_Ang = np.expand_dims(ang_batch[:1], -1)
                shapes_X = tuple(sample_X.shape[1:])
                shapes_A = tuple(sample_A.shape[1:])
                shapes_Ang = tuple(sample_Ang.shape[1:])

                if self.supervised_dict is not None:
                    sample_Y = self.supervised_dict[key][:1]
                    shapes_Y = tuple(sample_Y.shape[1:])

            n = int(X_batch.shape[0])
            total_samples += n
            video_indices.append(np.full(n, i, dtype=np.int32))

        if h5_chunk_len is None:
            h5_chunk_len = min(512, max(1, total_samples))

        video_indices = np.concatenate(video_indices, axis=0) if video_indices else np.zeros((0,), dtype=np.int32)

        self._cleanup_tmp_files()
        self._mark_build_incomplete()

        y_final = self.y_path if self.supervised_dict is not None else None
        do_shuffle = bool(self.global_shuffle) and total_samples > 1

        if do_shuffle:
            x_seq = self._tmp_path(self.X_path)
            a_seq = self._tmp_path(self.a_path)
            ang_seq = self._tmp_path(self.ang_path)
            y_seq = self._tmp_path(self.y_path) if y_final is not None else None
        else:
            x_seq, a_seq, ang_seq, y_seq = self.X_path, self.a_path, self.ang_path, y_final

        try:
            self._sequential_write(
                preprocessed_dict=preprocessed_dict,
                keys=keys,
                x_path=x_seq,
                a_path=a_seq,
                ang_path=ang_seq,
                y_path=y_seq,
                total_samples=total_samples,
                shapes_X=shapes_X,
                shapes_A=shapes_A,
                shapes_Ang=shapes_Ang,
                shapes_Y=shapes_Y,
                h5_chunk_len=h5_chunk_len,
                keys_hash=keys_hash,
                write_final_attrs=not do_shuffle,
                shuffle_seed=shuffle_seed,
            )

            if do_shuffle:
                rng = np.random.default_rng(shuffle_seed)
                perm = rng.permutation(total_samples).astype(np.int64, copy=False)
                print(
                    f"BatchDictDataset: permuting {total_samples} windows globally "
                    f"(temporary extra disk ≈ one full copy of the dataset)."
                )
                self._write_shuffled_from_sequential(
                    src_x=x_seq,
                    src_a=a_seq,
                    src_ang=ang_seq if self.has_angles else None,
                    src_y=y_seq,
                    perm=perm,
                    total_samples=total_samples,
                    shapes_X=shapes_X,
                    shapes_A=shapes_A,
                    shapes_Ang=shapes_Ang,
                    shapes_Y=shapes_Y,
                    h5_chunk_len=h5_chunk_len,
                    keys_hash=keys_hash,
                    n_videos=len(keys),
                    shuffle_seed=shuffle_seed,
                )
                video_indices = video_indices[perm]
        finally:
            if do_shuffle:
                self._cleanup_tmp_files()

        np.save(self.idx_path, video_indices)
        print(
            f"HDF5 built. Samples: {total_samples}, chunks: {h5_chunk_len}, "
            f"global_shuffle: {bool(do_shuffle)}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        if self._h5_X is None:
            self._h5_X = h5py.File(self.X_path, 'r')
            self._h5_A = h5py.File(self.a_path, 'r')
            self._X = self._h5_X['X']
            self._A = self._h5_A['a']
            self._vid = np.load(self.idx_path, mmap_mode='r')

            if self.return_angles:
                self._h5_Ang = h5py.File(self.ang_path, 'r')
                self._Ang = self._h5_Ang['ang']

            if self.supervised_dict is not None and os.path.exists(self.y_path):
                self._h5_Y = h5py.File(self.y_path, 'r')
                self._Y = self._h5_Y['y']
            else:
                self._Y = None

        x_np = self._X[idx]
        a_np = self._A[idx]
        x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
        a = torch.from_numpy(np.ascontiguousarray(a_np)).float()
        vid = int(self._vid[idx])

        ret = [x, a]
        if self.return_angles:
            ang_np = self._Ang[idx]
            ang = torch.from_numpy(np.ascontiguousarray(ang_np)).float()
            ret.append(ang)

        if self._Y is not None:
            y_np = self._Y[idx]
            y_val = torch.from_numpy(np.ascontiguousarray(y_np)).float()
            ret.append(y_val)

        ret.append(torch.tensor(idx, dtype=torch.long))
        ret.append(vid)
        return tuple(ret)

    def make_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        iterable_for_h5: bool = True,
        rdcc_nbytes: int = 64 * 1024**2,
        rdcc_nslots: int = 1_000_000,
        block_shuffle: bool = True,
        permute_within_block: bool = False,
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
        seed: Optional[int] = None,
        ddp_shard: bool = True,
        bootstrap_training: bool = False,
        bootstrap_block_len: int = 250,
    ) -> DataLoader:

        # get DDP identity (if needed)
        ddp_rank = 0
        ddp_world_size = 1
        if ddp_shard and dist.is_available() and dist.is_initialized():
            ddp_rank = dist.get_rank()
            ddp_world_size = dist.get_world_size()

        if persistent_workers is None:
            persistent_workers = num_workers > 0

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))

        if iterable_for_h5:
            iterable = _H5BatchIterableDataset(
                self,
                x_path=self.X_path,
                a_path=self.a_path,
                ang_path=self.ang_path,
                y_path=self.y_path if self.supervised_dict is not None else None,
                idx_path=self.idx_path,
                batch_size=batch_size,
                n_samples=self.length,
                shuffle=shuffle,
                drop_last=drop_last,
                rdcc_nbytes=rdcc_nbytes,
                rdcc_nslots=rdcc_nslots,
                block_shuffle=block_shuffle,
                permute_within_block=permute_within_block,
                return_angles=self.return_angles,
                seed=seed,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                bootstrap_training=bootstrap_training,
                bootstrap_block_len=bootstrap_block_len,
                global_shuffle=self.global_shuffle,
            )
            return DataLoader(
                iterable,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                generator=gen,
            )
        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                generator=gen,
            )


class _H5BatchIterableDataset(IterableDataset):
    def __init__(
        self,
        base_dataset: BatchDictDataset,
        x_path: str,
        a_path: str,
        ang_path: str,
        y_path: Optional[str],
        idx_path: str,
        batch_size: int,
        n_samples: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        rdcc_nbytes: int = 64 * 1024**2,
        rdcc_nslots: int = 1_000_000,
        block_shuffle: bool = True,
        permute_within_block: bool = False,
        return_angles: int = False,
        seed: Optional[int] = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        bootstrap_training: bool = False,
        bootstrap_block_len: int = 250,
        global_shuffle: bool = False,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.x_path = x_path
        self.a_path = a_path
        self.ang_path = ang_path
        self.y_path = y_path
        self.idx_path = idx_path
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.block_shuffle = block_shuffle
        self.permute_within_block = permute_within_block
        self.return_angles = return_angles
        self.seed = None if seed is None else int(seed)
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.bootstrap_training = bootstrap_training
        self.bootstrap_block_len = int(bootstrap_block_len)
        self.global_shuffle = bool(global_shuffle)

    def __getattr__(self, name):
        # Called only if attribute not found on self
        return getattr(self.base_dataset, name)

    def __len__(self) -> int:
        if self.n_samples is None:
            with h5py.File(self.x_path, 'r') as f:
                n = int(f['X'].shape[0])
        else:
            n = self.n_samples

        bs = self.batch_size
        total_batches = (n // bs) if self.drop_last else ((n + bs - 1) // bs)

        # If sharded across ranks, each rank sees exactly 1/world_size of batches (excess is dropped)
        if self.ddp_world_size > 1:
            total_batches = (total_batches // self.ddp_world_size) * self.ddp_world_size
            return total_batches // self.ddp_world_size

        return total_batches

    def _compute_video_ranges(self, video_idx: np.ndarray) -> tuple:
        """
        Returns:
        starts: (n_videos,) start index in [0,n)
        ends:   (n_videos,) end index (exclusive)
        Assumes each video is contiguous in the concatenated dataset (true without global_shuffle).
        """
        vid = np.asarray(video_idx)
        n = vid.shape[0]
        if n == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        change = np.flatnonzero(vid[1:] != vid[:-1]) + 1
        bounds = np.concatenate(([0], change, [n])).astype(np.int64)
        starts = bounds[:-1]
        ends = bounds[1:]
        return starts, ends

    def _block_bootstrap_batch_starts(
        self,
        rng: np.random.Generator,
        video_idx: np.ndarray,
        n: int,
        bs: int,
        target_batches: int,
    ) -> np.ndarray:
        """
        Build an array of length target_batches, containing *batch start indices*.
        Each consecutive group of starts comes from a contiguous block within one video.
        """
        v_starts, v_ends = self._compute_video_ranges(video_idx)
        v_lens = (v_ends - v_starts).astype(np.int64)

        # only videos that can supply at least one full batch
        ok = v_lens >= bs
        v_starts, v_ends, v_lens = v_starts[ok], v_ends[ok], v_lens[ok]
        if len(v_lens) == 0:
            raise RuntimeError("No video segment long enough to provide a full batch.")

        L = int(self.bootstrap_block_len)

        # For efficient batching without boundary issues, add starts at step=bs.
        # (L is not required to be a multiple of bs)
        batches_per_block = int(max(1, np.ceil(L / bs)))

        out = np.empty((target_batches,), dtype=np.int64)
        filled = 0

        while filled < target_batches:
            v = int(rng.choice(len(v_lens), p=None))
            vs, ve, m = int(v_starts[v]), int(v_ends[v]), int(v_lens[v])

            # How many full batches fit in the selected video
            max_batches_in_video = m // bs
            bpb = min(batches_per_block, max_batches_in_video)

            # Choose a block start such that bpb full batches fit (so s+bs*bpb <= ve)
            max_start = ve - (bpb * bs)
            if max_start < vs:
                # extremely short video edge case; skip
                continue

            s0 = int(rng.integers(vs, max_start + 1))

            # write batch starts for this block
            for j in range(bpb):
                if filled >= target_batches:
                    break
                out[filled] = s0 + j * bs
                filled += 1

        return out

    def __iter__(self):
        X_h5 = h5py.File(self.x_path, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        A_h5 = h5py.File(self.a_path, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)

        Y_h5 = None
        Y = None
        if self.y_path is not None and os.path.exists(self.y_path):
            Y_h5 = h5py.File(self.y_path, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
            Y = Y_h5['y']

        video_idx = np.load(self.idx_path, mmap_mode='r')

        X = X_h5['X']
        A = A_h5['a']
        n = int(X.shape[0])
        bs = self.batch_size

        if self.drop_last:
            n_batches = n // bs
            starts = np.arange(0, n_batches * bs, bs, dtype=np.int64)
        else:
            starts = np.arange(0, n, bs, dtype=np.int64)

        w = get_worker_info()
        worker_id = w.id if w is not None else 0
        num_workers = w.num_workers if w is not None else 1

        # epoch counter (increments once per __iter__ call in each worker)
        self._epoch = getattr(self, "_epoch", 0) + 1

        # using self.seed from make_loader so all DDP ranks shuffle identically, then shard by DDP rank
        base_seed = self.seed if self.seed is not None else 0
        epoch_seed = (base_seed + self._epoch) % (2**32)
        rng = np.random.default_rng(epoch_seed)

        if self.shuffle and self.block_shuffle:
            rng.shuffle(starts)

        # Ensure divisible by world_size so every rank gets the same number of batches
        if self.ddp_world_size > 1:
            n_full = (len(starts) // self.ddp_world_size) * self.ddp_world_size
            starts = starts[:n_full]

        # Bootstrap: temporal blocks if storage is video-contiguous; i.i.d. batch
        # resampling if the HDF5 was globally shuffled at build time.
        if self.bootstrap_training:
            if self.global_shuffle:
                if (
                    self.bootstrap_block_len != 1
                    and worker_id == 0
                    and self.ddp_rank == 0
                ):
                    print(
                        "BatchDictDataset: global_shuffle=True; using i.i.d. batch "
                        f"bootstrap (ignoring bootstrap_block_len={self.bootstrap_block_len})."
                    )
                if len(starts) > 0:
                    starts = rng.choice(starts, size=len(starts), replace=True)
            else:
                target_batches = len(starts)
                starts = self._block_bootstrap_batch_starts(
                    rng=rng,
                    video_idx=video_idx,
                    n=n,
                    bs=bs,
                    target_batches=target_batches,
                )

        # DDP shard first
        if self.ddp_world_size > 1:
            starts = starts[self.ddp_rank :: self.ddp_world_size]

        # Then worker shard
        if w is not None:
            starts = starts[worker_id :: num_workers]

        Ang_h5 = None
        Ang = None
        if self.return_angles:
            Ang_h5 = h5py.File(self.ang_path, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
            Ang = Ang_h5['ang']

        for s in starts:
            e = min(s + bs, n)
            x_np = X[s:e]
            a_np = A[s:e]
            vid = video_idx[s:e]

            ang_np = Ang[s:e] if Ang is not None else None
            y_np = Y[s:e] if Y is not None else None

            if self.shuffle and self.permute_within_block:
                perm = rng.permutation(e - s)
                x_np = x_np[perm]
                a_np = a_np[perm]
                if ang_np is not None:
                    ang_np = ang_np[perm]
                if y_np is not None:
                    y_np = y_np[perm]
                vid = vid[perm]

            x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
            a = torch.from_numpy(np.ascontiguousarray(a_np)).float()
            vid_t = torch.from_numpy(np.ascontiguousarray(vid.astype(np.int32)))
            idx_np = np.arange(s, e, dtype=np.int64)
            idx_t = torch.from_numpy(idx_np)

            batch = [x, a]
            if ang_np is not None:
                batch.append(torch.from_numpy(np.ascontiguousarray(ang_np)).float())

            if y_np is not None:
                batch.append(torch.from_numpy(np.ascontiguousarray(y_np)).float())

            batch.append(idx_t)
            batch.append(vid_t)
            yield tuple(batch)

        X_h5.close()
        A_h5.close()
        if Ang_h5 is not None:
            Ang_h5.close()
        if Y_h5 is not None:
            Y_h5.close()