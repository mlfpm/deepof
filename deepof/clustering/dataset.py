# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof.clustering

"""Creation of a dataset for model training."""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from deepof.data_loading import get_dt


class BatchDictDataset(Dataset):
    def __init__(self, preprocessed_dict, dataset_folder):
        """
        preprocessed_dict: dict-like object (could be a path to a dict on disk, or an opened dict)
        dataset_folder: folder to store the HDF5 files
        get_dt: function(preprocessed_dict, key) -> (X_batch, a_batch)
        """
        self.preprocessed_dict = preprocessed_dict
        self.keys = list(preprocessed_dict.keys())
        self.dataset_folder = dataset_folder
        self.X_path = os.path.join(dataset_folder, 'X_data.h5')
        self.a_path = os.path.join(dataset_folder, 'a_data.h5')
        self.idx_path = os.path.join(dataset_folder, 'video_idx.npy')

        # Delete old prepared dataset (if any)
        for path in [self.X_path, self.a_path, self.idx_path]:
            if os.path.exists(path):
                os.remove(path)

        # Create HDF5 datasets
        self._create_hdf5_datasets()

        # Open HDF5 files for reading
        self.X_h5 = h5py.File(self.X_path, 'r')
        self.a_h5 = h5py.File(self.a_path, 'r')
        self.X = self.X_h5['X']
        self.a = self.a_h5['a']
        self.video_idx = np.load(self.idx_path)
        self.length = self.X.shape[0]

    def _create_hdf5_datasets(self):
        os.makedirs(self.dataset_folder, exist_ok=True)
        total_samples = 0
        shapes_X = None
        shapes_a = None
        video_indices = []

        # First pass: determine total number of samples and shapes
        for key in self.keys:
            X_batch, a_batch = get_dt(self.preprocessed_dict, key)
            if shapes_X is None:
                shapes_X = X_batch.shape[1:]  # (Window length, Main features)
                shapes_a = a_batch.shape[1:]  # (Window length, Edge features)
            total_samples += X_batch.shape[0]
            video_indices.append(np.full(X_batch.shape[0], self.keys.index(key), dtype=np.int32))

        # Create HDF5 datasets
        with h5py.File(self.X_path, 'w') as X_h5, h5py.File(self.a_path, 'w') as a_h5:
            X_dset = X_h5.create_dataset('X', shape=(total_samples, *shapes_X), dtype='float32', chunks=True)
            a_dset = a_h5.create_dataset('a', shape=(total_samples, *shapes_a), dtype='float32', chunks=True)

            # Second pass: write data
            idx = 0
            for key in self.keys:
                X_batch, a_batch = get_dt(self.preprocessed_dict, key)
                n = X_batch.shape[0]
                X_dset[idx:idx+n] = X_batch
                a_dset[idx:idx+n] = a_batch
                idx += n

        # Save video indices
        video_indices = np.concatenate(video_indices, axis=0)
        np.save(self.idx_path, video_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        a = torch.from_numpy(self.a[idx]).float()
        video_idx = int(self.video_idx[idx])
        return x, a, video_idx

    def __del__(self):
        # Ensure files are closed
        try:
            if hasattr(self, 'X_h5') and self.X_h5:
                self.X_h5.close()
            if hasattr(self, 'a_h5') and self.a_h5:
                self.a_h5.close()
        except Exception:
            pass

    # For multi-worker DataLoader support
    def _lazy_init(self):
        if not hasattr(self, 'X_h5') or self.X_h5 is None:
            self.X_h5 = h5py.File(self.X_path, 'r')
            self.a_h5 = h5py.File(self.a_path, 'r')
            self.X = self.X_h5['X']
            self.a = self.a_h5['a']
            self.video_idx = np.load(self.idx_path)

    def __getitem__(self, idx):
        self._lazy_init()
        x = torch.from_numpy(self.X[idx]).float()
        a = torch.from_numpy(self.a[idx]).float()
        video_idx = int(self.video_idx[idx])
        return x, a, video_idx


    