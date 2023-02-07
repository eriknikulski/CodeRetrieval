import torch
from torch.nn.utils.rnn import pad_sequence

import const


def collate(batch):
    return tuple(pad_sequence(list(elems), batch_first=True, padding_value=const.PAD_TOKEN)
                 if torch.is_tensor(elems[0]) else elems for elems in zip(*batch))
