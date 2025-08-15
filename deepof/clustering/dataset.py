import os
import h5py
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from typing import Dict, Tuple, Optional
from deepof.data_loading import get_dt

def reorder_and_reshape(data: np.ndarray) -> np.ndarray:
    xy_order = [
        'B_Spine_1', 'B_Center', 'B_Left_bhip', 'B_Left_ear', 'B_Left_fhip',
        'B_Nose', 'B_Right_bhip', 'B_Right_ear', 'B_Right_fhip', 'B_Spine_2',
        'B_Tail_base'
    ]
    likelihood_order = [
        'B_Center', 'B_Left_bhip', 'B_Left_ear', 'B_Left_fhip', 'B_Nose',
        'B_Right_bhip', 'B_Right_ear', 'B_Right_fhip', 'B_Spine_1',
        'B_Spine_2', 'B_Tail_base'
    ]
    p_map = {name: i + 22 for i, name in enumerate(likelihood_order)}
    new_indices = [val for i, name in enumerate(xy_order) for val in (i * 2, i * 2 + 1, p_map[name])]
    final_shape = (*data.shape[:-1], 11, 3)
    return data[:, :, new_indices].reshape(final_shape)

class BatchDictDataset:
    def __init__(
        self,
        preprocessed_dict: Dict,
        dataset_folder: str,
        dataset_name: str,
        in_memory: bool = False,
        force_rebuild: bool = False,
        h5_chunk_len: Optional[int] = None,
    ):
        self.in_memory = in_memory
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name

        self.X_path = os.path.join(dataset_folder, dataset_name + 'X_data.h5')
        self.a_path = os.path.join(dataset_folder, dataset_name + 'a_data.h5')
        self.idx_path = os.path.join(dataset_folder, dataset_name + 'video_idx.npy')

        if in_memory:
            self._load_all_to_memory(preprocessed_dict)
        else:
            self._init_hdf5(preprocessed_dict, force_rebuild=force_rebuild, h5_chunk_len=h5_chunk_len)

    def _load_all_to_memory(self, preprocessed_dict: Dict):
        print("BatchDictDataset: loading to memory...")
        all_x, all_a, all_video_idx = [], [], []
        keys = list(preprocessed_dict.keys())
        for i, key in enumerate(keys):
            X_batch, a_batch = get_dt(preprocessed_dict, key)
            all_x.append(reorder_and_reshape(X_batch))
            all_a.append(np.expand_dims(a_batch, -1))
            all_video_idx.append(np.full(X_batch.shape[0], i, dtype=np.int32))

        X = np.concatenate(all_x, axis=0)
        A = np.concatenate(all_a, axis=0)
        video_idx = np.concatenate(all_video_idx, axis=0)

        X = np.ascontiguousarray(X, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)

        self.X_tensor = torch.from_numpy(X)
        self.A_tensor = torch.from_numpy(A)
        self.video_idx = torch.from_numpy(video_idx.astype(np.int32))

        self.x_shape = tuple(self.X_tensor.shape[1:])
        self.a_shape = tuple(self.A_tensor.shape[1:])
        self.length = int(self.X_tensor.shape[0])
        print(f"In-memory dataset ready. Samples: {self.length}, x_shape: {self.x_shape}, a_shape: {self.a_shape}")

    def _init_hdf5(self, preprocessed_dict: Dict, force_rebuild: bool, h5_chunk_len: Optional[int]):
        os.makedirs(self.dataset_folder, exist_ok=True)
        need_build = force_rebuild or (not os.path.exists(self.X_path) or not os.path.exists(self.a_path) or not os.path.exists(self.idx_path))
        if need_build:
            print(f"BatchDictDataset: building HDF5 at {self.dataset_folder}...")
            self._build_hdf5(preprocessed_dict, h5_chunk_len=h5_chunk_len)
        else:
            print(f"BatchDictDataset: found existing HDF5 at {self.dataset_folder}")

        with h5py.File(self.X_path, 'r') as f:
            X_ds = f['X']
            self.x_shape = tuple(X_ds.shape[1:])
            self.length = int(X_ds.shape[0])
        with h5py.File(self.a_path, 'r') as f:
            A_ds = f['a']
            self.a_shape = tuple(A_ds.shape[1:])
        self._h5_X = None
        self._h5_A = None
        print(f"HDF5 dataset ready. Samples: {self.length}, x_shape: {self.x_shape}, a_shape: {self.a_shape}")

    def _build_hdf5(self, preprocessed_dict: Dict, h5_chunk_len: Optional[int]):
        keys = list(preprocessed_dict.keys())
        total_samples = 0
        shapes_X = None
        shapes_A = None
        video_indices = []

        for i, key in enumerate(keys):
            X_batch, a_batch = get_dt(preprocessed_dict, key)
            if shapes_X is None:
                sample_X = reorder_and_reshape(X_batch[:1])
                sample_A = np.expand_dims(a_batch[:1], -1)
                shapes_X = tuple(sample_X.shape[1:])
                shapes_A = tuple(sample_A.shape[1:])
            n = int(X_batch.shape[0])
            total_samples += n
            video_indices.append(np.full(n, i, dtype=np.int32))

        if h5_chunk_len is None:
            h5_chunk_len = min(512, total_samples)

        with h5py.File(self.X_path, 'w') as X_h5, h5py.File(self.a_path, 'w') as A_h5:
            X_dset = X_h5.create_dataset(
                'X',
                shape=(total_samples, *shapes_X),
                dtype='float32',
                chunks=(h5_chunk_len, *shapes_X),
                compression=None, shuffle=False, fletcher32=False,
                maxshape=(total_samples, *shapes_X),
            )
            A_dset = A_h5.create_dataset(
                'a',
                shape=(total_samples, *shapes_A),
                dtype='float32',
                chunks=(h5_chunk_len, *shapes_A),
                compression=None, shuffle=False, fletcher32=False,
                maxshape=(total_samples, *shapes_A),
            )

            idx = 0
            for key in keys:
                X_batch, a_batch = get_dt(preprocessed_dict, key)
                n = int(X_batch.shape[0])
                X_re = reorder_and_reshape(X_batch).astype(np.float32, copy=False)
                A_re = np.expand_dims(a_batch, -1).astype(np.float32, copy=False)
                X_dset[idx:idx+n] = X_re
                A_dset[idx:idx+n] = A_re
                idx += n

        video_indices = np.concatenate(video_indices, axis=0)
        np.save(self.idx_path, video_indices)
        print(f"HDF5 built. Samples: {total_samples}, chunks: {h5_chunk_len}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.in_memory:
            x = self.X_tensor[idx]
            a = self.A_tensor[idx]
            vid = int(self.video_idx[idx].item() if torch.is_tensor(self.video_idx) else self.video_idx[idx])
            return x, a, vid
        else:
            if self._h5_X is None:
                self._h5_X = h5py.File(self.X_path, 'r')
                self._h5_A = h5py.File(self.a_path, 'r')
                self._X = self._h5_X['X']
                self._A = self._h5_A['a']
                self._vid = np.load(self.idx_path, mmap_mode='r')
            x_np = self._X[idx]
            a_np = self._A[idx]
            x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
            a = torch.from_numpy(np.ascontiguousarray(a_np)).float()
            vid = int(self._vid[idx])
            return x, a, vid

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
                idx_path=self.idx_path,
                batch_size=batch_size,
                n_samples=self.length,
                shuffle=shuffle,
                drop_last=drop_last,
                rdcc_nbytes=rdcc_nbytes,
                rdcc_nslots=rdcc_nslots,
                block_shuffle=block_shuffle,
                permute_within_block=permute_within_block,
            )
            return DataLoader(
                iterable,
                batch_size=None,                 # already batched
                shuffle=False,                   # iterator controls order
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
        else:
            # slow map-style HDF5 (per-sample)
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
        idx_path: str,
        batch_size: int,
        n_samples: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        rdcc_nbytes: int = 64 * 1024**2,
        rdcc_nslots: int = 1_000_000,
        block_shuffle: bool = True,
        permute_within_block: bool = False,
    ):
        super().__init__()
        self.x_path = x_path
        self.a_path = a_path
        self.idx_path = idx_path
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.block_shuffle = block_shuffle
        self.permute_within_block = permute_within_block

    def __len__(self) -> int:
        if self.n_samples is None:
            with h5py.File(self.x_path, 'r') as f:
                n = int(f['X'].shape[0])
        else:
            n = self.n_samples
        bs = self.batch_size
        return (n // bs) if self.drop_last else ((n + bs - 1) // bs)

    def __iter__(self):
        # Each worker opens its own file handles
        X_h5 = h5py.File(self.x_path, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        A_h5 = h5py.File(self.a_path, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        video_idx = np.load(self.idx_path, mmap_mode='r')

        X = X_h5['X']
        A = A_h5['a']
        n = int(X.shape[0])
        bs = self.batch_size

        # Build global list of contiguous batch starts
        if self.drop_last:
            n_batches = n // bs
            starts = np.arange(0, n_batches * bs, bs, dtype=np.int64)
        else:
            starts = np.arange(0, n, bs, dtype=np.int64)

        # Worker-aware sharding
        w = get_worker_info()
        if self.shuffle and self.block_shuffle:
            # derive a global seed consistent across workers in this epoch
            # PyTorch seeds each worker with base_seed + worker_id
            # so base_seed ≈ torch.initial_seed() - worker_id
            worker_id = w.id if w is not None else 0
            base_seed = (torch.initial_seed() - worker_id) % (2**32)
            rng = np.random.default_rng(base_seed)
            rng.shuffle(starts)

        if w is not None:
            starts = starts[w.id::w.num_workers]  # disjoint strided split

        for s in starts:
            e = min(s + bs, n)
            x_np = X[s:e]
            a_np = A[s:e]
            vid = video_idx[s:e]

            if self.shuffle and self.permute_within_block:
                # optional extra randomness inside the batch
                perm = np.random.default_rng().permutation(e - s)
                x_np = x_np[perm]
                a_np = a_np[perm]
                vid = vid[perm]

            x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
            a = torch.from_numpy(np.ascontiguousarray(a_np)).float()
            vid_t = torch.from_numpy(np.ascontiguousarray(vid.astype(np.int32)))

            yield x, a, vid_t

        X_h5.close()
        A_h5.close()