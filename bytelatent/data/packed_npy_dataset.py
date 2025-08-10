# In bytelatent/data/packed_npy_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PackedNpyDataset(Dataset):
    """
    A PyTorch dataset for reading pre-packed, tokenized data from a .npy file.
    """
    def __init__(self, data_path: str, sequence_length: int):
        self.sequence_length = sequence_length

        file_path = Path(data_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")

        # Load the numpy array using memory-mapping for efficiency with large files.
        self.data = np.load(file_path, mmap_mode='r')

        self.num_sequences = len(self.data)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Retrieve the pre-packed sequence
        tokens = self.data[idx]

        # Convert to PyTorch tensors that the model expects
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))

        return {"x": x, "y": y}