import numpy as np
from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    batch = np.array(batch)
    return pad_sequence(batch[::, 0], batch_first=True), pad_sequence(batch[::, 1], batch_first=True), batch[::, 2]
