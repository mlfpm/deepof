import os
import hashlib
import h5py
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from typing import Dict, Tuple, Optional
from deepof.data_loading import get_dt

def reorder_and_reshape(data: np.ndarray) -> np.ndarray:
    assert data.shape[2] % 3 == 0, "Error! Number of columns is not a multiple of 3 (x, y, likelihood)!"
    final_shape = (*data.shape[:-1], int(data.shape[2]/3), 3)
    return data.reshape(final_shape)


class BatchDictDataset:
    def __init__(
        self,
        preprocessed_dict: Dict,
        dataset_folder: str,
        dataset_name: str,
        in_memory: bool = False,
        force_rebuild: bool = False,
        h5_chunk_len: Optional[int] = None,
        return_angles: Optional[bool] = False,
        supervised_dict: Optional[Dict] = None,
    ):
        self.in_memory = in_memory
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name
        self.return_angles = return_angles
        self.supervised_dict = supervised_dict
        
        # Determine if the dataset has angles
        self.has_angles = False
        if get_dt(preprocessed_dict, list(preprocessed_dict.keys())[0])[2].size > 0:
            self.has_angles = True

        self.X_path = os.path.join(dataset_folder, dataset_name + 'X_data.h5')
        self.a_path = os.path.join(dataset_folder, dataset_name + 'a_data.h5')
        self.ang_path = os.path.join(dataset_folder, dataset_name + 'ang_data.h5')
        self.y_path = os.path.join(dataset_folder, dataset_name + 'y_data.h5')
        self.idx_path = os.path.join(dataset_folder, dataset_name + 'video_idx.npy')

        if in_memory:
            self._load_all_to_memory(preprocessed_dict)
        else:
            self._init_hdf5(preprocessed_dict, force_rebuild=force_rebuild, h5_chunk_len=h5_chunk_len)

    def _load_all_to_memory(self, preprocessed_dict: Dict):
        print("BatchDictDataset: loading to memory...")
        all_x, all_a, all_ang, all_y, all_video_idx = [], [], [], [], []
        keys = list(preprocessed_dict.keys())
        for i, key in enumerate(keys):
            X_batch, a_batch, ang_batch = get_dt(preprocessed_dict, key)
            all_x.append(reorder_and_reshape(X_batch))
            all_a.append(np.expand_dims(a_batch, -1))
            all_ang.append(np.expand_dims(ang_batch, -1))

            if self.supervised_dict is not None:
                y_batch = self.supervised_dict[key]
                assert y_batch.shape[0] == X_batch.shape[0], \
                    f"Shape mismatch for key {key}: X has {X_batch.shape[0]} rows, Y has {y_batch.shape[0]}"
                all_y.append(y_batch)

            all_video_idx.append(np.full(X_batch.shape[0], i, dtype=np.int32))

        X = np.concatenate(all_x, axis=0)
        A = np.concatenate(all_a, axis=0)
        Ang = np.concatenate(all_ang, axis=0)
        video_idx = np.concatenate(all_video_idx, axis=0)

        X = np.ascontiguousarray(X, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        Ang = np.ascontiguousarray(Ang, dtype=np.float32)

        self.X_tensor = torch.from_numpy(X)
        self.A_tensor = torch.from_numpy(A)
        if self.has_angles:
            self.Ang_tensor = torch.from_numpy(Ang)
        else:
            self.Ang_tensor = None
        self.video_idx = torch.from_numpy(video_idx.astype(np.int32))

        # Process supervised Y tensor
        if self.supervised_dict is not None:
            Y = np.concatenate(all_y, axis=0)
            Y = np.ascontiguousarray(Y, dtype=np.float32)
            self.Y_tensor = torch.from_numpy(Y)
            self.y_shape = tuple(self.Y_tensor.shape[1:])
        else:
            self.Y_tensor = None
            self.y_shape = None

        self.x_shape = tuple(self.X_tensor.shape[1:])
        self.a_shape = tuple(self.A_tensor.shape[1:])
        if self.has_angles:
            self.ang_shape = tuple(self.Ang_tensor.shape[1:])
        else:
            self.ang_shape = None
        self.length = int(self.X_tensor.shape[0])
        print(f"In-memory dataset ready. Samples: {self.length}, x_shape: {self.x_shape}, a_shape: {self.a_shape}, ang_shape: {self.ang_shape}")

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

    def _init_hdf5(self, preprocessed_dict: Dict, h5_chunk_len: Optional[int], force_rebuild: bool = False):
        os.makedirs(self.dataset_folder, exist_ok=True)

        if force_rebuild:
            need_build, reason = True, "Force rebuild requested"
        else:
            need_build, reason = self._does_need_build(preprocessed_dict)

        if need_build:
            print(f"BatchDictDataset: building HDF5 at {self.dataset_folder}...")
            print(f"  Reason: {reason}")
            self._build_hdf5(preprocessed_dict, h5_chunk_len=h5_chunk_len)
        else:
            print(f"BatchDictDataset: reusing existing HDF5 at {self.dataset_folder}")

        with h5py.File(self.X_path, 'r') as f:
            X_ds = f['X']
            self.x_shape = tuple(X_ds.shape[1:])
            self.length = int(X_ds.shape[0])
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
        print(f"HDF5 dataset ready. Samples: {self.length}, x_shape: {self.x_shape}, a_shape: {self.a_shape}, ang_shape: {self.ang_shape}")

    def _build_hdf5(self, preprocessed_dict: Dict, h5_chunk_len: Optional[int]):
        keys = list(preprocessed_dict.keys())
        keys_hash = hashlib.md5(','.join(sorted(str(k) for k in keys)).encode()).hexdigest()
        
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

                # Check Y shape
                if self.supervised_dict is not None:
                    sample_Y = self.supervised_dict[key][:1]
                    shapes_Y = tuple(sample_Y.shape[1:])

            n = int(X_batch.shape[0])
            total_samples += n
            video_indices.append(np.full(n, i, dtype=np.int32))

        if h5_chunk_len is None:
            h5_chunk_len = min(512, total_samples)

        f_X = h5py.File(self.X_path, 'w')
        f_A = h5py.File(self.a_path, 'w')
        f_Ang = h5py.File(self.ang_path, 'w')
        f_Y = h5py.File(self.y_path, 'w') if self.supervised_dict is not None else None

        try:
            # Mark build as incomplete at start
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
            if self.has_angles:
                Ang_dset = f_Ang.create_dataset(
                    'ang', shape=(total_samples, *shapes_Ang), dtype='float32',
                    chunks=(h5_chunk_len, *shapes_Ang), compression=None, shuffle=False, fletcher32=False,
                    maxshape=(total_samples, *shapes_Ang),
                )

            # Create Y dataset
            Y_dset = None
            if f_Y is not None:
                Y_dset = f_Y.create_dataset(
                    'y', shape=(total_samples, *shapes_Y), dtype='float32',
                    chunks=(h5_chunk_len, *shapes_Y), compression=None, shuffle=False, fletcher32=False,
                    maxshape=(total_samples, *shapes_Y),
                )

            idx = 0
            for key in keys:
                X_batch, a_batch, ang_batch = get_dt(preprocessed_dict, key)
                n = int(X_batch.shape[0])
                X_re = reorder_and_reshape(X_batch).astype(np.float32, copy=False)
                A_re = np.expand_dims(a_batch, -1).astype(np.float32, copy=False)
                Ang_re = np.expand_dims(ang_batch, -1).astype(np.float32, copy=False)

                X_dset[idx:idx+n] = X_re
                A_dset[idx:idx+n] = A_re
                if ang_batch.size > 0:
                    Ang_dset[idx:idx+n] = Ang_re

                # Write Y data with assertion
                if Y_dset is not None:
                    y_batch = self.supervised_dict[key]
                    assert y_batch.shape[0] == n, \
                        f"Shape mismatch for key {key}: X has {n} rows, Y has {y_batch.shape[0]}. Check windowing."

                    Y_re = y_batch.astype(np.float32, copy=False)
                    Y_dset[idx:idx+n] = Y_re

                idx += n

            # Store metadata and mark build complete
            f_X.attrs['keys_hash'] = keys_hash
            f_X.attrs['n_videos'] = len(keys)
            f_X.attrs['n_samples'] = total_samples
            f_X.attrs['build_complete'] = True
            
        finally:
            f_X.close()
            f_A.close()
            f_Ang.close()
            if f_Y is not None:
                f_Y.close()

        video_indices = np.concatenate(video_indices, axis=0)
        np.save(self.idx_path, video_indices)
        print(f"HDF5 built. Samples: {total_samples}, chunks: {h5_chunk_len}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if self.in_memory:
            x = self.X_tensor[idx]
            a = self.A_tensor[idx]
            vid = int(self.video_idx[idx].item() if torch.is_tensor(self.video_idx) else self.video_idx[idx])

            ret = [x, a]
            if self.return_angles:
                ret.append(self.Ang_tensor[idx])

            if self.Y_tensor is not None:
                ret.append(self.Y_tensor[idx])

            ret.append(vid)
            return tuple(ret)
        else:
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
    ) -> DataLoader:
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        if self.in_memory:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )

        if iterable_for_h5:
            iterable = _H5BatchIterableDataset(
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
            )
            return DataLoader(
                iterable,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
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
            )


class _H5BatchIterableDataset(IterableDataset):
    def __init__(
        self,
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
    ):
        super().__init__()
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

    def __len__(self) -> int:
        if self.n_samples is None:
            with h5py.File(self.x_path, 'r') as f:
                n = int(f['X'].shape[0])
        else:
            n = self.n_samples
        bs = self.batch_size
        return (n // bs) if self.drop_last else ((n + bs - 1) // bs)

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
        if self.shuffle and self.block_shuffle:
            worker_id = w.id if w is not None else 0
            base_seed = (torch.initial_seed() - worker_id) % (2**32)
            rng = np.random.default_rng(base_seed)
            rng.shuffle(starts)

        if w is not None:
            starts = starts[w.id::w.num_workers]

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
                perm = np.random.default_rng().permutation(e - s)
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

            batch = [x, a]
            if ang_np is not None:
                batch.append(torch.from_numpy(np.ascontiguousarray(ang_np)).float())

            if y_np is not None:
                batch.append(torch.from_numpy(np.ascontiguousarray(y_np)).float())

            batch.append(vid_t)
            yield tuple(batch)

        X_h5.close()
        A_h5.close()
        if Ang_h5 is not None:
            Ang_h5.close()
        if Y_h5 is not None:
            Y_h5.close()