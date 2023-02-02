import torch

import const


def collate(batch):
    batch = (torch.stack(elem) for elem in zip(*batch))
    return tuple(elem[:, :elem.size(1) - (elem == const.PAD_TOKEN).sum(axis=1).min()] if elem.dim() == 2 else elem
                 for elem in batch)
