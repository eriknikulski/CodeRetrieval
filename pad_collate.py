import const

import torch


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    device = vec.get_device() if torch.cuda.is_available() else None
    return torch.cat([vec, torch.full(pad_size, const.PAD_TOKEN, dtype=torch.long).to(device)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label, url)

        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        max_len = [max(x[0].shape[self.dim] for x in batch),
                   max(x[1].shape[self.dim] for x in batch)]

        batch = [(pad_tensor(x[0], pad=max_len[0], dim=self.dim),
                  pad_tensor(x[1], pad=max_len[1], dim=self.dim),
                  x[2])
                 for x in batch]

        xs = torch.stack([x[0] for x in batch], dim=0)
        ys = torch.stack([x[1] for x in batch], dim=0)
        zs = [x[2] for x in batch]
        return xs, ys, zs

    def __call__(self, batch):
        return self.pad_collate(batch)
