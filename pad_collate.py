import numpy as np
from torch.nn.utils.rnn import pad_sequence

import const


def collate(batch):
    batch = np.array(batch)
    return pad_sequence(batch[::, 0], batch_first=True, padding_value=const.PAD_TOKEN), \
        pad_sequence(batch[::, 1], batch_first=True, padding_value=const.PAD_TOKEN), batch[::, 2]
