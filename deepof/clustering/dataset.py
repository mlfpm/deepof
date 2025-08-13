import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
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


class BatchDictDataset(Dataset):
    def __init__(self, preprocessed_dict, dataset_folder, dataset_name, in_memory: bool = False):
        """
        Initializes the dataset.

        Args:
            preprocessed_dict: Dictionary-like object with data.
            dataset_folder (str): Path to store/load HDF5 files.
            dataset_name (str): Prefix for the dataset files.
            in_memory (bool): If True, loads all data into RAM. If False, uses HDF5 on disk.
        """
        self.in_memory = in_memory

        if self.in_memory:
            print("Initializing dataset in IN-MEMORY mode.")
            self._load_all_to_memory(preprocessed_dict)
        else:
            print("Initializing dataset in HDF5 mode.")
            self._init_hdf5_mode(preprocessed_dict, dataset_folder, dataset_name)
    
    def _load_all_to_memory(self, preprocessed_dict):
        """Loads all data from the preprocessed dict into RAM."""
        all_x = []
        all_a = []
        all_video_idx = []
        keys = list(preprocessed_dict.keys())

        for i, key in enumerate(keys):
            X_batch, a_batch = get_dt(preprocessed_dict, key)
            all_x.append(reorder_and_reshape(X_batch))
            all_a.append(np.expand_dims(a_batch, -1))
            all_video_idx.append(np.full(X_batch.shape[0], i, dtype=np.int32))

        # Concatenate all batches into single NumPy arrays
        self.X = np.concatenate(all_x, axis=0)
        self.a = np.concatenate(all_a, axis=0)
        self.video_idx = np.concatenate(all_video_idx, axis=0)

        # Store shapes and length directly
        self.x_shape = self.X.shape[1:]
        self.a_shape = self.a.shape[1:]
        self.length = len(self.X)
        print(f"Dataset loaded into memory. Total samples: {self.length}")

    def _init_hdf5_mode(self, preprocessed_dict, dataset_folder, dataset_name):
        """Initializes the dataset for on-disk HDF5 access."""
        self.preprocessed_dict = preprocessed_dict
        self.keys = list(preprocessed_dict.keys())
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name
        
        self.X_path = os.path.join(dataset_folder, dataset_name + 'X_data.h5')
        self.a_path = os.path.join(dataset_folder, dataset_name + 'a_data.h5')
        self.idx_path = os.path.join(dataset_folder, dataset_name + 'video_idx.npy')
        
        self.X_h5 = None
        self.a_h5 = None

        if not os.path.exists(self.X_path):
            print(f"Creating HDF5 dataset at {self.X_path}...")
            self._create_hdf5_datasets()
        
        with h5py.File(self.X_path, 'r') as f:
            self.x_shape = f['X'].shape[1:]
        with h5py.File(self.a_path, 'r') as f:
            self.a_shape = f['a'].shape[1:]
        
        self.video_idx = np.load(self.idx_path)
        self.length = len(self.video_idx)
        print(f"Dataset initialized. X shape per item: {self.x_shape}, A shape per item: {self.a_shape}")

    def _create_hdf5_datasets(self):
        """Creates HDF5 files from the preprocessed dictionary."""
        # This function remains unchanged and is only called in HDF5 mode.
        os.makedirs(self.dataset_folder, exist_ok=True)
        total_samples, shapes_X, shapes_a = 0, None, None
        video_indices = []

        for key in self.keys:
            X_batch, a_batch = get_dt(self.preprocessed_dict, key)
            if shapes_X is None:
                shapes_X = (X_batch.shape[1], 11, 3) 
                shapes_a = (a_batch.shape[1], a_batch.shape[2], 1)
            total_samples += X_batch.shape[0]
            video_indices.append(np.full(X_batch.shape[0], self.keys.index(key), dtype=np.int32))

        with h5py.File(self.X_path, 'w') as X_h5, h5py.File(self.a_path, 'w') as a_h5:
            X_dset = X_h5.create_dataset('X', shape=(total_samples, *shapes_X), dtype='float32', chunks=True)
            a_dset = a_h5.create_dataset('a', shape=(total_samples, *shapes_a), dtype='float32', chunks=True)
            
            idx = 0
            for key in self.keys:
                X_batch, a_batch = get_dt(self.preprocessed_dict, key)
                n = X_batch.shape[0]               
                X_dset[idx:idx+n] = reorder_and_reshape(X_batch)
                a_dset[idx:idx+n] = np.expand_dims(a_batch,-1)
                idx += n

        video_indices = np.concatenate(video_indices, axis=0)
        np.save(self.idx_path, video_indices)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.in_memory:
            # In-memory mode: access NumPy arrays directly
            x = torch.from_numpy(self.X[idx]).float()
            a = torch.from_numpy(self.a[idx]).float()
            video_idx = int(self.video_idx[idx])
        else:
            # HDF5 mode: lazy-open file handles
            if self.X_h5 is None:
                self.X_h5 = h5py.File(self.X_path, 'r')
                self.a_h5 = h5py.File(self.a_path, 'r')
                self.X = self.X_h5['X']
                self.a = self.a_h5['a']

            x = torch.from_numpy(self.X[idx]).float()
            a = torch.from_numpy(self.a[idx]).float()
            video_idx = int(self.video_idx[idx])
        
        return x, a, video_idx
