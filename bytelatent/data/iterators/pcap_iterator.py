from typing import Any, Generator

import numpy as np

from bytelatent.data.data_types import Batch
from bytelatent.data.iterators.abstract_iterator import PydanticIteratorState, StatefulIterator


class PcapIteratorState(PydanticIteratorState):
    data_path: str
    batch_size: int
    current_idx: int = 0
    split: str = "train"  # "train" or "validation"

    def build(self):
        return PcapIterator(
            data_path=self.data_path,
            batch_size=self.batch_size,
            current_idx=self.current_idx,
            split=self.split
        )


class PcapIterator(StatefulIterator):
    def __init__(self, data_path: str, batch_size: int, current_idx: int = 0, split: str = "train"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.current_idx = current_idx
        self.split = split

        # Load the prepared data
        if split == "train":
            self.data = np.load(f"{data_path}/train_packed.npy")
        else:
            self.data = np.load(f"{data_path}/validation_packed.npy")

    def get_state(self):
        return PcapIteratorState(
            data_path=self.data_path,
            batch_size=self.batch_size,
            current_idx=self.current_idx,
            split=self.split
        )

    def create_iter(self) -> Generator[Batch, Any, None]:
        while self.current_idx < len(self.data):
            batch_end = min(self.current_idx + self.batch_size, len(self.data))
            batch_data = self.data[self.current_idx:batch_end]

            # Create x (input) and y (target) for next-byte prediction
            x = batch_data[:, :-1]
            y = batch_data[:, 1:]

            yield Batch(
                x=x,
                y=y,
                mask=None,  # No masking needed for entropy model
                patch_lengths=None,  # Not used for entropy model
                ngram_ids=None
            )

            self.current_idx = batch_end